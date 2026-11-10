"use strict";

// The Record tab: capture the microphone, play it back, run the whole recording through the
// selected engine (the ESP32 or this computer), then show the result with graphs and
// measurements. The engines are the same ones the Live tab uses.
const Recorder = (() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const A = HushwaveAnalysis;
  const C = HushwaveCharts;
  const FS = 16000;
  const BLOCK = 256;
  const MAX_RECORD_S = 120;
  const MIN_RECORD_S = 1;
  const LIVE_SPAN_BLOCKS = 625;

  const state = {
    kind: "mcu",
    recording: false,
    busy: false,
    cancel: false,
    samples: null,
    specIn: null,
    results: { mcu: null, python: null },
    port: null,
    ctx: null,
    stream: null,
    blocks: [],
    peaks: [],
    level: -90,
    shown: -90,
    frame: 0,
    startedAt: 0,
    urls: [],
  };

  const NAMES = { mcu: "ESP32", python: "this computer" };
  const FILE_TAGS = { mcu: "esp32", python: "computer" };

  const text = (selector, value) => {
    $(selector).textContent = value;
  };
  const show = (selector, visible) => {
    $(selector).hidden = !visible;
  };
  const num = (value, digits = 1) => (Number.isFinite(value) ? value.toFixed(digits) : "n/a");
  const withUnit = (value, unit, digits = 1) => (Number.isFinite(value) ? `${value.toFixed(digits)} ${unit}` : "n/a");
  const signedUnit = (value, unit) => {
    if (!Number.isFinite(value)) return "n/a";
    const t = Math.abs(value).toFixed(1);
    return `${Number(t) === 0 ? "" : value < 0 ? "-" : "+"}${t} ${unit}`;
  };
  const clock = (seconds) => {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const meterWidth = (db) => `${Math.min(100, Math.max(0, ((db + 60) / 60) * 100)).toFixed(1)}%`;

  function showError(message) {
    text("#rec-error", message || "");
    show("#rec-error", Boolean(message));
  }

  function describe(error) {
    const serial = HushwaveEngines.explainSerialError(error);
    if (serial) return serial;
    if (error && error.name === "NotAllowedError") {
      return "Access was denied. Allow the microphone in the address bar, then try again.";
    }
    if (error && error.name === "NotFoundError") return "No microphone was found.";
    return error && error.message ? error.message : String(error);
  }

  // -------------------------------------------------------------------- charts
  const charts = [];
  let drawQueued = false;

  function register(id, draw, getData) {
    const el = $(`#${id}`);
    const canvas = $("canvas", el);
    const entry = { el, canvas, draw, getData, height: Number(el.dataset.h) };
    el.style.height = `${entry.height}px`;
    const head = $(".playhead", el);
    if (head) {
      head.style.top = `${C.PAD.t}px`;
      head.style.bottom = `${C.PAD.b}px`;
      entry.head = head;
    }
    if (el.classList.contains("seekable")) {
      const seekTo = (event) => {
        const rect = el.getBoundingClientRect();
        seek(C.timeFraction(event.clientX - rect.left, rect.width));
      };
      el.addEventListener("pointerdown", (event) => {
        el.setPointerCapture(event.pointerId);
        seekTo(event);
      });
      el.addEventListener("pointermove", (event) => {
        if (event.buttons) seekTo(event);
      });
    }
    if (typeof ResizeObserver === "function") new ResizeObserver(() => scheduleDraw()).observe(el);
    charts.push(entry);
    return entry;
  }

  function drawChart(entry) {
    const width = Math.floor(entry.el.clientWidth);
    if (width < 10) return;
    const data = entry.getData();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    entry.canvas.width = Math.round(width * dpr);
    entry.canvas.height = Math.round(entry.height * dpr);
    entry.canvas.style.width = `${width}px`;
    entry.canvas.style.height = `${entry.height}px`;
    if (!data) return;
    entry.draw(entry.canvas.getContext("2d"), width, entry.height, data, C.themeFrom(document.documentElement), dpr);
  }

  function scheduleDraw() {
    if (drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(() => {
      drawQueued = false;
      for (const entry of charts) if (entry.el.offsetParent !== null) drawChart(entry);
      renderPlayhead();
    });
  }

  function currentResult() {
    return state.results[state.kind];
  }

  function setupCharts() {
    register("chart-wave", C.drawWaveform, () =>
      state.samples ? { original: state.samples, denoised: currentResult() ? currentResult().run.output : null, fs: FS } : null
    );
    register("chart-spec-in", C.drawSpectrogram, () => {
      const r = currentResult();
      return r ? { spec: state.specIn, fs: FS, vmin: r.vmin, vmax: r.vmax } : null;
    });
    register("chart-spec-out", C.drawSpectrogram, () => {
      const r = currentResult();
      return r ? { spec: r.specOut, fs: FS, vmin: r.vmin, vmax: r.vmax } : null;
    });
    register("chart-levels", C.drawLevels, () => {
      const r = currentResult();
      return r ? { original: r.seg.energy, denoised: r.energyOut, runs: r.runs, floorDb: r.seg.floorDb, hop: BLOCK, fs: FS } : null;
    });
    register("chart-spectra", C.drawSpectra, () => {
      const r = currentResult();
      return r ? { series: r.series, fs: FS } : null;
    });
    register("chart-timing", C.drawTiming, () => {
      const r = currentResult();
      return r ? { compute: r.run.computeMs, rtt: r.run.rttMs, blockMs: r.blockMs } : null;
    });
    const live = $("#rec-live");
    const box = $("#rec-live-box");
    box.style.height = `${box.dataset.h}px`;
    state.liveCanvas = live;
    if (window.matchMedia) {
      const query = window.matchMedia("(prefers-color-scheme: dark)");
      if (query.addEventListener) query.addEventListener("change", scheduleDraw);
    }
  }

  function paintLive() {
    const box = $("#rec-live-box");
    const width = Math.floor(box.clientWidth);
    if (width < 10) return;
    const height = Number(box.dataset.h);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const canvas = state.liveCanvas;
    if (canvas.width !== Math.round(width * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    C.drawEnvelope(
      canvas.getContext("2d"),
      width,
      height,
      { peaks: state.peaks, spanBlocks: Math.max(state.peaks.length, LIVE_SPAN_BLOCKS) },
      C.themeFrom(document.documentElement),
      dpr
    );
  }

  // ------------------------------------------------------------------- playback
  const player = {
    ctx: null,
    buffers: {},
    variant: "original",
    nodes: null,
    playing: false,
    startedAt: 0,
    offset: 0,
    duration: 0,
    token: 0,
    frame: 0,
  };

  function playerContext() {
    if (!player.ctx) player.ctx = new (window.AudioContext || window.webkitAudioContext)();
    return player.ctx;
  }

  function position() {
    if (!player.playing) return player.offset;
    return Math.min(player.duration, player.ctx.currentTime - player.startedAt);
  }

  function setPlayState(playing) {
    $("#rec-play").dataset.state = playing ? "playing" : "paused";
    text("#rec-play-label", playing ? "Pause" : "Play");
  }

  function renderPlayhead() {
    const fraction = player.duration > 0 ? position() / player.duration : 0;
    text("#rec-time", `${clock(position())} / ${clock(player.duration)}`);
    for (const entry of charts) {
      if (!entry.head) continue;
      entry.head.hidden = !entry.getData();
      entry.head.style.left = `calc(${C.PAD.l}px + (100% - ${C.PAD.l + C.PAD.r}px) * ${fraction.toFixed(5)})`;
    }
    if (player.playing) player.frame = requestAnimationFrame(renderPlayhead);
  }

  function stopSources() {
    player.token++;
    if (!player.nodes) return;
    for (const { source } of Object.values(player.nodes)) {
      source.onended = null;
      try {
        source.stop();
      } catch (_) {
        /* already stopped */
      }
      source.disconnect();
    }
    player.nodes = null;
  }

  function begin(at) {
    stopSources();
    const ctx = playerContext();
    const token = player.token;
    const when = ctx.currentTime + 0.05;
    player.nodes = {};
    for (const [key, buffer] of Object.entries(player.buffers)) {
      const source = ctx.createBufferSource();
      const gain = ctx.createGain();
      source.buffer = buffer;
      gain.gain.value = key === player.variant ? 1 : 0;
      source.connect(gain).connect(ctx.destination);
      source.start(when, at);
      player.nodes[key] = { source, gain };
    }
    const first = Object.values(player.nodes)[0];
    first.source.onended = () => {
      if (token !== player.token) return;
      player.playing = false;
      player.offset = 0;
      stopSources();
      setPlayState(false);
      renderPlayhead();
    };
    player.startedAt = when - at;
    player.playing = true;
    setPlayState(true);
    cancelAnimationFrame(player.frame);
    renderPlayhead();
  }

  async function play() {
    const ctx = playerContext();
    if (ctx.state === "suspended") await ctx.resume();
    begin(player.offset >= player.duration - 0.05 ? 0 : player.offset);
  }

  function pausePlayback() {
    if (!player.playing) return;
    player.offset = position();
    player.playing = false;
    stopSources();
    setPlayState(false);
    renderPlayhead();
  }

  function toggle() {
    if (!state.samples || state.recording) return;
    if (player.playing) pausePlayback();
    else play().catch((error) => showError(describe(error)));
  }

  function seek(fraction) {
    if (!state.samples) return;
    player.offset = Math.min(Math.max(fraction, 0), 1) * player.duration;
    if (player.playing) begin(player.offset);
    else renderPlayhead();
  }

  function setVariant(key) {
    if (!player.buffers[key]) return;
    player.variant = key;
    for (const button of $("#rec-variants").children) {
      button.setAttribute("aria-pressed", String(button.dataset.key === key));
    }
    if (player.nodes && player.ctx) {
      for (const [name, { gain }] of Object.entries(player.nodes)) {
        gain.gain.setTargetAtTime(name === key ? 1 : 0, player.ctx.currentTime, 0.01);
      }
    }
  }

  function loadPlayer(denoised) {
    pausePlayback();
    stopSources();
    const ctx = playerContext();
    const make = (samples) => {
      const buffer = ctx.createBuffer(1, samples.length, FS);
      buffer.copyToChannel(samples, 0);
      return buffer;
    };
    player.buffers = { original: make(state.samples) };
    if (denoised) player.buffers.denoised = make(denoised);
    player.duration = state.samples.length / FS;
    player.offset = Math.min(player.offset, player.duration);
    const denoisedButton = $("#rec-variants").children[1];
    denoisedButton.disabled = !denoised;
    setVariant(denoised && player.variant === "denoised" ? "denoised" : "original");
    setPlayState(false);
  }

  // ------------------------------------------------------------------- capture
  function onBlock(block) {
    if (!state.recording) return;
    let peak = 0;
    let sum = 0;
    for (let i = 0; i < block.length; i++) {
      const v = Math.abs(block[i]);
      if (v > peak) peak = v;
      sum += block[i] * block[i];
    }
    state.blocks.push(block);
    state.peaks.push(peak);
    const rms = Math.sqrt(sum / block.length);
    state.level = rms <= 0 ? -90 : Math.max(-90, 20 * Math.log10(rms));
    state.shown = Math.max(state.shown, state.level);
    if (state.blocks.length * BLOCK >= MAX_RECORD_S * FS) stopRecording();
  }

  function paintRecording() {
    state.shown = Math.max(state.level, state.shown - 1.2);
    $("#rec-meter").style.width = meterWidth(state.shown);
    text("#rec-db", `${state.level.toFixed(0)} dBFS`);
    text("#rec-timer", clock((state.blocks.length * BLOCK) / FS));
    paintLive();
    if (state.recording) state.frame = requestAnimationFrame(paintRecording);
  }

  function releaseCapture() {
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    if (state.ctx) state.ctx.close().catch(() => {});
    state.stream = state.ctx = null;
  }

  function setRecordingUi(recording) {
    const button = $("#rec-toggle");
    button.dataset.state = recording ? "recording" : "idle";
    text("#rec-toggle-label", recording ? "Stop" : state.samples ? "Record again" : "Record");
    text("#rec-status", recording ? "Recording" : state.samples ? "Recorded" : "Ready");
    $("#rec-status").classList.toggle("live", recording);
    show("#rec-live-box", recording);
    for (const button of $("#rec-engines").children) button.disabled = recording || state.busy || (button.dataset.kind === "mcu" && !navigator.serial);
    $("#rec-run").disabled = recording || state.busy;
  }

  async function startRecording() {
    showError("");
    pausePlayback();
    try {
      state.ctx = new AudioContext({ sampleRate: FS, latencyHint: "interactive" });
      if (state.ctx.state === "suspended") await state.ctx.resume();
      if (state.ctx.sampleRate !== FS) {
        throw new Error(`This browser runs audio at ${state.ctx.sampleRate} Hz instead of ${FS} Hz. Use Chrome or Edge.`);
      }
      await state.ctx.audioWorklet.addModule("/web/worklets.js");
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const source = state.ctx.createMediaStreamSource(state.stream);
      const node = new AudioWorkletNode(state.ctx, "capture", { processorOptions: { block: BLOCK } });
      const mute = state.ctx.createGain();
      mute.gain.value = 0;
      source.connect(node).connect(mute).connect(state.ctx.destination);
      node.port.onmessage = (event) => onBlock(event.data);
      state.blocks = [];
      state.peaks = [];
      state.level = state.shown = -90;
      state.recording = true;
      setRecordingUi(true);
      paintRecording();
    } catch (error) {
      console.error(error);
      showError(describe(error));
      releaseCapture();
      state.recording = false;
      setRecordingUi(false);
    }
  }

  function clearDownloads() {
    for (const url of state.urls) URL.revokeObjectURL(url);
    state.urls = [];
  }

  function stopRecording() {
    if (!state.recording) return;
    state.recording = false;
    cancelAnimationFrame(state.frame);
    releaseCapture();
    $("#rec-meter").style.width = "0";
    text("#rec-db", "-90 dBFS");
    const blocks = state.blocks;
    state.blocks = [];
    const samples = new Float32Array(blocks.length * BLOCK);
    blocks.forEach((block, index) => samples.set(block, index * BLOCK));
    if (samples.length < MIN_RECORD_S * FS) {
      showError("The recording is too short. Record at least one second.");
      setRecordingUi(false);
      return;
    }
    state.samples = samples;
    state.specIn = null;
    state.results = { mcu: null, python: null };
    player.offset = 0;
    player.variant = "original";
    text("#rec-timer", clock(samples.length / FS));
    setRecordingUi(false);
    show("#rec-playback", true);
    show("#rec-denoise", true);
    showResult();
  }

  function newRecording() {
    pausePlayback();
    stopSources();
    state.samples = null;
    state.specIn = null;
    state.results = { mcu: null, python: null };
    player.buffers = {};
    player.duration = 0;
    player.offset = 0;
    show("#rec-playback", false);
    show("#rec-denoise", false);
    show("#rec-results", false);
    showError("");
    text("#rec-timer", "0:00");
    setRecordingUi(false);
  }

  // ------------------------------------------------------------------- denoising
  function setBusy(busy) {
    state.busy = busy;
    $("#rec-run").disabled = busy;
    $("#rec-toggle").disabled = busy;
    $("#rec-new").disabled = busy;
    show("#rec-cancel", busy);
    for (const button of $("#rec-engines").children) button.disabled = busy || (button.dataset.kind === "mcu" && !navigator.serial);
  }

  function setProgress(done, total) {
    show("#rec-progress", true);
    $("#rec-progress-bar").style.width = `${((100 * done) / Math.max(1, total)).toFixed(1)}%`;
    text("#rec-progress-text", `${done} of ${total} blocks`);
  }

  function buildResult(kind, info, run) {
    const y = state.samples;
    const z = run.output;
    const seg = A.segment(y);
    const metrics = A.measure(y, z, seg);
    if (!state.specIn) state.specIn = A.spectrogram(y);
    const specOut = A.spectrogram(z);
    const vmax = Math.max(A.dbPercentile(state.specIn, 99.9), A.dbPercentile(specOut, 99.9));
    const pauseFrames = A.frameMask(seg.pause, state.specIn.frames, "all");
    const series = [
      { db: A.averageSpectrumDb(state.specIn), color: "original" },
      { db: A.averageSpectrumDb(specOut), color: "denoised" },
    ];
    const pauseIn = A.averageSpectrumDb(state.specIn, pauseFrames);
    const pauseOut = A.averageSpectrumDb(specOut, pauseFrames);
    if (pauseIn && pauseOut) {
      series.push({ db: pauseIn, color: "original", dash: [5, 4] }, { db: pauseOut, color: "denoised", dash: [5, 4] });
    }
    return {
      kind,
      info,
      run,
      seg,
      metrics,
      specOut,
      vmax,
      vmin: vmax - 80,
      energyOut: A.blockEnergyDb(z),
      runs: A.runs(seg.pause),
      series,
      blockMs: (1000 * info.block) / info.fs,
      compute: A.describeValues(run.computeMs),
      rtt: A.describeValues(run.rttMs),
    };
  }

  async function runDenoise() {
    if (!state.samples || state.busy || state.recording) return;
    const kind = state.kind;
    showError("");
    show("#rec-warn", false);
    pausePlayback();
    setBusy(true);
    setProgress(0, 1);
    text("#rec-run-status", kind === "mcu" ? "Connecting to the board" : "Starting");
    state.cancel = false;
    let engine = null;
    let failure = null;
    try {
      engine = kind === "mcu" ? new HushwaveEngines.McuEngine(state.port ? { port: state.port } : {}) : new HushwaveEngines.PythonEngine();
      engine.onError = (error) => {
        failure = error;
      };
      const info = await engine.open();
      if (kind === "mcu") state.port = engine.port;
      text("#rec-run-status", `Processing on ${NAMES[kind]}`);
      const run = await A.denoiseWhole(engine, state.samples, info, {
        isCancelled: () => state.cancel,
        failed: () => failure,
        onProgress: setProgress,
      });
      text("#rec-run-status", "Analysing");
      await new Promise((resolve) => setTimeout(resolve, 30));
      state.results[kind] = buildResult(kind, info, run);
      text("#rec-run-status", `Done on ${NAMES[kind]}`);
      showResult();
    } catch (error) {
      if (error && error.name === "AbortError") {
        text("#rec-run-status", "Cancelled");
      } else {
        console.error(error);
        text("#rec-run-status", "");
        showError(describe(error));
        if (kind === "mcu") state.port = null;
      }
    } finally {
      show("#rec-progress", false);
      if (engine) await engine.close().catch(() => {});
      setBusy(false);
    }
  }

  // --------------------------------------------------------------------- results
  function showResult() {
    const r = currentResult();
    const denoised = r ? r.run.output : null;
    if (state.samples) loadPlayer(denoised);
    show("#rec-results", Boolean(r));
    show("#dl-denoised", Boolean(r));
    show("#dl-json", Boolean(r));
    show("#rec-legend-denoised", Boolean(r));
    if (r) {
      const m = r.metrics;
      const delayMs = (1000 * r.info.delay) / r.info.fs;
      text("#rec-summary", `${r.info.name}. ${num(m.durationS, 1)} s at ${r.info.fs / 1000} kHz. Algorithmic delay ${num(delayMs, 1)} ms.`);
      text("#m-nr", withUnit(m.noiseReductionPauseDb, "dB"));
      text("#m-nr-sub", `${m.pauseBlocks} of ${m.totalBlocks} blocks are pauses`);
      text("#m-speech", signedUnit(m.speechLevelChangeDb, "dB"));
      text("#m-speech-sub", `${m.speechBlocks} blocks are speech`);
      text("#m-resid", withUnit(m.residualPauseDbfs, "dBFS"));
      text("#m-resid-sub", `Before: ${withUnit(m.originalPauseDbfs, "dBFS")}`);
      text("#m-compute", withUnit(r.compute.mean, "ms", 2));
      text("#m-compute-sub", `Max ${num(r.compute.max, 2)} ms, ${num((100 * r.compute.mean) / r.blockMs, 1)} % of the ${num(r.blockMs, 0)} ms block`);
      text("#m-rtt", withUnit(r.rtt.mean, "ms"));
      text("#m-rtt-sub", `Max ${num(r.rtt.max, 1)} ms, 95th percentile ${num(r.rtt.p95, 1)} ms`);
      text("#m-blocks", `${r.run.received} of ${r.run.blocks}`);
      text("#m-blocks-sub", `${r.run.lost} lost, ${r.run.dropped} dropped`);
      if (r.run.lost + r.run.dropped > 0) {
        text("#rec-warn", `${r.run.lost + r.run.dropped} blocks did not come back from the engine. The output has gaps. Run the denoising again.`);
        show("#rec-warn", true);
      }
    }
    scheduleDraw();
  }

  function setKind(kind) {
    if (state.busy || state.recording) return;
    state.kind = kind;
    for (const button of $("#rec-engines").children) button.setAttribute("aria-pressed", String(button.dataset.kind === kind));
    text("#rec-run-label", `Denoise with ${NAMES[kind]}`);
    text(
      "#rec-engine-note",
      kind === "mcu"
        ? "Every block of the recording goes over USB to the ESP32, which denoises it and sends it back."
        : "The same denoiser runs in Python on this computer."
    );
    text("#rec-run-status", state.results[kind] ? `Result from ${NAMES[kind]}` : "");
    show("#rec-warn", false);
    if (state.samples) showResult();
  }

  // ------------------------------------------------------------------- downloads
  function download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function measurementsJson() {
    const r = currentResult();
    const m = r.metrics;
    const clean = (value) => (Number.isFinite(value) ? Number(value.toFixed(4)) : null);
    return {
      engine: r.info.name,
      duration_s: clean(m.durationS),
      sample_rate_hz: r.info.fs,
      block_samples: r.info.block,
      algorithmic_delay_samples: r.info.delay,
      noise_floor_db: clean(m.floorDb),
      pause_blocks: m.pauseBlocks,
      speech_blocks: m.speechBlocks,
      total_blocks: m.totalBlocks,
      noise_reduction_pause_db: clean(m.noiseReductionPauseDb),
      noise_level_pause_before_dbfs: clean(m.originalPauseDbfs),
      noise_level_pause_after_dbfs: clean(m.residualPauseDbfs),
      speech_level_change_db: clean(m.speechLevelChangeDb),
      peak_before_dbfs: clean(m.peakInDbfs),
      peak_after_dbfs: clean(m.peakOutDbfs),
      rms_before_dbfs: clean(m.rmsInDbfs),
      rms_after_dbfs: clean(m.rmsOutDbfs),
      blocks_received: r.run.received,
      blocks_lost: r.run.lost,
      blocks_dropped: r.run.dropped,
      processing_ms: { mean: clean(r.compute.mean), p95: clean(r.compute.p95), max: clean(r.compute.max) },
      round_trip_ms: { mean: clean(r.rtt.mean), p95: clean(r.rtt.p95), max: clean(r.rtt.max) },
    };
  }

  // ------------------------------------------------------------------------ init
  function key(event) {
    if (!state.samples || state.recording) return;
    if (event.key === " ") toggle();
    else if (event.key === "1") setVariant("original");
    else if (event.key === "2") setVariant("denoised");
    else if (event.key === "ArrowLeft") seek((position() - 2) / player.duration);
    else if (event.key === "ArrowRight") seek((position() + 2) / player.duration);
    else return;
    event.preventDefault();
  }

  function init() {
    setupCharts();
    const hasSerial = Boolean(navigator.serial);
    $("#rec-engines").children[0].disabled = !hasSerial;
    show("#rec-engine-hint", !hasSerial);
    setKind(hasSerial ? "mcu" : "python");
    for (const button of $("#rec-engines").children) button.addEventListener("click", () => setKind(button.dataset.kind));
    for (const button of $("#rec-variants").children) button.addEventListener("click", () => setVariant(button.dataset.key));
    $("#rec-toggle").addEventListener("click", () => (state.recording ? stopRecording() : startRecording()));
    $("#rec-play").addEventListener("click", toggle);
    $("#rec-run").addEventListener("click", runDenoise);
    $("#rec-cancel").addEventListener("click", () => {
      state.cancel = true;
    });
    $("#rec-new").addEventListener("click", newRecording);
    $("#dl-original").addEventListener("click", () => {
      if (state.samples) download(A.wavBlob(state.samples, FS), "recording-original.wav");
    });
    $("#dl-denoised").addEventListener("click", () => {
      const r = currentResult();
      if (r) download(A.wavBlob(r.run.output, FS), `recording-denoised-${FILE_TAGS[state.kind]}.wav`);
    });
    $("#dl-json").addEventListener("click", () => {
      if (!currentResult()) return;
      const blob = new Blob([JSON.stringify(measurementsJson(), null, 2) + "\n"], { type: "application/json" });
      download(blob, `recording-measurements-${FILE_TAGS[state.kind]}.json`);
    });
    window.addEventListener("pagehide", clearDownloads);
    setRecordingUi(false);
  }

  // Called when the tab is shown, so that the charts are drawn at their real size.
  function activate() {
    scheduleDraw();
  }

  // Called when the tab is left.
  function pause() {
    pausePlayback();
    if (state.recording) stopRecording();
  }

  return { init, activate, pause, key, isBusy: () => state.busy };
})();
