"use strict";

const $ = (selector, root = document) => root.querySelector(selector);

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function signed(value, digits = 1) {
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return text;
  return `${value < 0 ? "-" : "+"}${text}`;
}

function showError(id, message) {
  const el = $(id);
  el.textContent = message;
  el.hidden = !message;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

// ------------------------------------------------------------------ replay
const Replay = (() => {
  const state = {
    manifest: null,
    clip: null,
    variant: "b5",
    buffers: new Map(),
    nodes: null,
    ctx: null,
    playing: false,
    startedAt: 0,
    offset: 0,
    duration: 0,
    token: 0,
    frame: 0,
  };
  const images = new Map();

  function context() {
    if (!state.ctx) state.ctx = new (window.AudioContext || window.webkitAudioContext)();
    return state.ctx;
  }

  function position() {
    if (!state.playing) return state.offset;
    return Math.min(state.duration, state.ctx.currentTime - state.startedAt);
  }

  function render() {
    const pos = position();
    const fraction = state.duration > 0 ? pos / state.duration : 0;
    $("#playhead").style.left = `${(100 * fraction).toFixed(3)}%`;
    $("#time").textContent = `${formatTime(pos)} / ${formatTime(state.duration)}`;
    $("#spec").setAttribute("aria-valuenow", String(Math.round(100 * fraction)));
    if (state.playing) state.frame = requestAnimationFrame(render);
  }

  function stopSources() {
    state.token++;
    if (!state.nodes) return;
    for (const { source } of state.nodes.values()) {
      source.onended = null;
      try {
        source.stop();
      } catch (_) {
        /* already stopped */
      }
      source.disconnect();
    }
    state.nodes = null;
  }

  function begin(at) {
    stopSources();
    const ctx = context();
    const buffers = state.buffers.get(state.clip.id);
    const token = state.token;
    const when = ctx.currentTime + 0.05;
    state.nodes = new Map();
    for (const variant of state.manifest.variants) {
      const source = ctx.createBufferSource();
      const gain = ctx.createGain();
      source.buffer = buffers[variant.key];
      gain.gain.value = variant.key === state.variant ? 1 : 0;
      source.connect(gain).connect(ctx.destination);
      source.start(when, at);
      state.nodes.set(variant.key, { source, gain });
    }
    const first = state.nodes.values().next().value;
    first.source.onended = () => {
      if (token !== state.token) return;
      state.playing = false;
      state.offset = 0;
      stopSources();
      $("#play").textContent = "Play";
      render();
    };
    state.startedAt = when - at;
    state.playing = true;
    $("#play").textContent = "Pause";
    cancelAnimationFrame(state.frame);
    render();
  }

  async function play() {
    const ctx = context();
    if (ctx.state === "suspended") await ctx.resume();
    begin(state.offset >= state.duration - 0.05 ? 0 : state.offset);
  }

  function pause() {
    state.offset = position();
    state.playing = false;
    stopSources();
    $("#play").textContent = "Play";
    render();
  }

  function toggle() {
    if (!state.clip) return;
    if (state.playing) pause();
    else play().catch((error) => showError("#replay-error", error.message));
  }

  function seek(fraction) {
    if (!state.clip) return;
    state.offset = Math.min(Math.max(fraction, 0), 1) * state.duration;
    if (state.playing) begin(state.offset);
    else render();
  }

  function seekBy(seconds) {
    seek((position() + seconds) / state.duration);
  }

  function setVariant(key) {
    state.variant = key;
    for (const button of $("#variants").children) {
      button.setAttribute("aria-pressed", String(button.dataset.key === key));
    }
    if (state.nodes && state.ctx) {
      for (const [name, { gain }] of state.nodes) {
        gain.gain.setTargetAtTime(name === key ? 1 : 0, state.ctx.currentTime, 0.01);
      }
    }
    const file = state.clip.files[key].image;
    $("#spec-img").src = `/examples/${file}`;
    for (const row of $("#measures tbody").children) {
      row.classList.toggle("active", row.dataset.key === key);
    }
  }

  function fillMeasures() {
    const table = $("#measures");
    const body = $("tbody", table);
    body.replaceChildren();
    const data = state.clip.measurements || {};
    const rows = state.manifest.variants.filter((v) => data[v.key]);
    table.hidden = rows.length === 0;
    for (const variant of rows) {
      const row = document.createElement("tr");
      row.dataset.key = variant.key;
      const cells = [
        variant.label,
        `${data[variant.key].noise_reduction_pause_db.toFixed(1)} dB`,
        `${signed(data[variant.key].speech_level_change_db)} dB`,
      ];
      cells.forEach((text, index) => {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        else cell.className = "num";
        cell.textContent = text;
        row.append(cell);
      });
      body.append(row);
    }
  }

  async function load(clip) {
    pause();
    state.offset = 0;
    state.clip = clip;
    state.duration = clip.duration_s;
    $("#clip-note").textContent = clip.note;
    $("#x-end").textContent = `${formatTime(clip.duration_s)}`;
    for (const button of $("#clips").children) {
      button.setAttribute("aria-pressed", String(button.dataset.id === clip.id));
    }
    for (const variant of state.manifest.variants) {
      const image = new Image();
      image.src = `/examples/${clip.files[variant.key].image}`;
      images.set(`${clip.id}/${variant.key}`, image);
    }
    fillMeasures();
    setVariant(state.variant);
    render();
    if (!state.buffers.has(clip.id)) {
      const ctx = context();
      const decoded = {};
      for (const variant of state.manifest.variants) {
        const data = await (await fetch(`/examples/${clip.files[variant.key].audio}`)).arrayBuffer();
        decoded[variant.key] = await ctx.decodeAudioData(data);
      }
      if (state.clip !== clip) return;
      state.buffers.set(clip.id, decoded);
    }
  }

  function seekFromPointer(event) {
    const rect = $("#spec").getBoundingClientRect();
    seek((event.clientX - rect.left) / rect.width);
  }

  async function init() {
    try {
      state.manifest = await getJson("/examples/manifest.json");
    } catch (_) {
      showError("#replay-error", "No example clips found. Run: make examples");
      $("#play").disabled = true;
      return;
    }
    const clips = $("#clips");
    for (const clip of state.manifest.clips) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = clip.title;
      button.dataset.id = clip.id;
      button.addEventListener("click", () => load(clip).catch(report));
      clips.append(button);
    }
    const variants = $("#variants");
    state.manifest.variants.forEach((variant, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = variant.label;
      button.dataset.key = variant.key;
      button.title = `Key ${index + 1}`;
      button.addEventListener("click", () => setVariant(variant.key));
      variants.append(button);
    });
    $("#play").addEventListener("click", toggle);
    $("#spec").addEventListener("pointerdown", seekFromPointer);
    await load(state.manifest.clips[0]).catch(report);
  }

  function report(error) {
    showError("#replay-error", `Could not load the clip: ${error.message}`);
  }

  function key(event) {
    if (!state.manifest) return;
    if (event.key === " ") {
      toggle();
    } else if (event.key >= "1" && event.key <= String(state.manifest.variants.length)) {
      setVariant(state.manifest.variants[Number(event.key) - 1].key);
    } else if (event.key === "ArrowLeft") {
      seekBy(-2);
    } else if (event.key === "ArrowRight") {
      seekBy(2);
    } else {
      return;
    }
    event.preventDefault();
  }

  return { init, pause, key };
})();

// -------------------------------------------------------------------- live
const Live = (() => {
  const MAX_QUEUE = 8;
  const PREFILL_BLOCKS = 4;
  const state = {
    running: false,
    enhanced: true,
    ctx: null,
    stream: null,
    capture: null,
    playback: null,
    sid: null,
    info: null,
    queue: [],
    sending: false,
    rtt: 0,
    compute: 0,
    computeMax: 0,
    bufferMs: 0,
    underruns: 0,
    dropped: 0,
    levelIn: -90,
    levelOut: -90,
    shownIn: -90,
    shownOut: -90,
    frame: 0,
  };

  const meterWidth = (db) => `${Math.min(100, Math.max(0, ((db + 60) / 60) * 100)).toFixed(1)}%`;

  function paint() {
    state.shownIn = Math.max(state.levelIn, state.shownIn - 1.2);
    state.shownOut = Math.max(state.levelOut, state.shownOut - 1.2);
    $("#meter-in").style.width = meterWidth(state.shownIn);
    $("#meter-out").style.width = meterWidth(state.shownOut);
    $("#db-in").textContent = `${state.levelIn.toFixed(0)} dBFS`;
    $("#db-out").textContent = `${state.levelOut.toFixed(0)} dBFS`;
    $("#s-compute").textContent = `${state.compute.toFixed(2)} ms (max ${state.computeMax.toFixed(2)})`;
    $("#s-rtt").textContent = `${state.rtt.toFixed(1)} ms`;
    $("#s-buffer").textContent = `${state.bufferMs.toFixed(0)} ms`;
    $("#s-drop").textContent = `${state.underruns} underruns, ${state.dropped} dropped`;
    const blockMs = (1000 * state.info.block) / state.info.fs;
    const total = blockMs + state.info.delay_ms + state.rtt + state.bufferMs;
    $("#s-total").textContent = `${total.toFixed(0)} ms`;
    if (state.running) state.frame = requestAnimationFrame(paint);
  }

  async function pump() {
    if (state.sending) return;
    state.sending = true;
    try {
      while (state.running && state.queue.length > 0) {
        const block = state.queue.shift();
        const started = performance.now();
        const response = await fetch(`/api/process?sid=${state.sid}&enh=${state.enhanced ? 1 : 0}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: block.buffer,
        });
        if (!response.ok) throw new Error(`engine returned HTTP ${response.status}`);
        const out = new Float32Array(await response.arrayBuffer());
        const elapsed = performance.now() - started;
        state.rtt = state.rtt === 0 ? elapsed : 0.95 * state.rtt + 0.05 * elapsed;
        const compute = Number(response.headers.get("X-Compute-Ms"));
        state.compute = state.compute === 0 ? compute : 0.95 * state.compute + 0.05 * compute;
        state.computeMax = Math.max(state.computeMax, compute);
        state.levelIn = Number(response.headers.get("X-Level-In"));
        state.levelOut = Number(response.headers.get("X-Level-Out"));
        state.shownIn = Math.max(state.shownIn, state.levelIn);
        state.shownOut = Math.max(state.shownOut, state.levelOut);
        if (state.running) state.playback.port.postMessage(out, [out.buffer]);
      }
    } catch (error) {
      if (state.running) {
        showError("#live-error", `Stopped: ${error.message}`);
        stop().catch(() => {});
      }
    } finally {
      state.sending = false;
    }
  }

  function describe(error) {
    if (error && error.name === "NotAllowedError") {
      return "Microphone access was denied. Allow it in the address bar, then start again.";
    }
    if (error && error.name === "NotFoundError") return "No microphone was found.";
    return error && error.message ? error.message : String(error);
  }

  async function start() {
    showError("#live-error", "");
    $("#live-toggle").disabled = true;
    $("#live-status").textContent = "Starting";
    try {
      state.info = await getJson("/api/info");
      const info = state.info;
      state.ctx = new AudioContext({ sampleRate: info.fs, latencyHint: "interactive" });
      if (state.ctx.state === "suspended") await state.ctx.resume();
      if (state.ctx.sampleRate !== info.fs) {
        throw new Error(`This browser runs audio at ${state.ctx.sampleRate} Hz instead of ${info.fs} Hz. Use Chrome or Edge.`);
      }
      await state.ctx.audioWorklet.addModule("/web/worklets.js");
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const session = await getJson("/api/session", { method: "POST" });
      state.sid = session.sid;
      const source = state.ctx.createMediaStreamSource(state.stream);
      state.capture = new AudioWorkletNode(state.ctx, "capture", {
        processorOptions: { block: info.block },
      });
      const mute = state.ctx.createGain();
      mute.gain.value = 0;
      source.connect(state.capture).connect(mute).connect(state.ctx.destination);
      state.playback = new AudioWorkletNode(state.ctx, "playback", {
        outputChannelCount: [1],
        processorOptions: { prefill: PREFILL_BLOCKS * info.block, capacity: 16 * info.block },
      });
      state.playback.connect(state.ctx.destination);
      state.playback.port.onmessage = (event) => {
        state.bufferMs = (1000 * event.data.fill) / info.fs;
        state.underruns = event.data.underruns;
      };
      state.queue = [];
      state.rtt = state.compute = state.computeMax = state.bufferMs = 0;
      state.underruns = state.dropped = 0;
      state.levelIn = state.levelOut = state.shownIn = state.shownOut = -90;
      state.capture.port.onmessage = (event) => {
        if (!state.running) return;
        state.queue.push(event.data);
        if (state.queue.length > MAX_QUEUE) {
          state.queue.shift();
          state.dropped++;
        }
        pump();
      };
      state.running = true;
      $("#live-toggle").textContent = "Stop";
      $("#live-status").textContent = "Running";
      paint();
    } catch (error) {
      showError("#live-error", describe(error));
      await teardown();
      $("#live-status").textContent = "Stopped";
    } finally {
      $("#live-toggle").disabled = false;
    }
  }

  async function teardown() {
    state.running = false;
    cancelAnimationFrame(state.frame);
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    if (state.sid) {
      fetch(`/api/close?sid=${state.sid}`, { method: "POST", keepalive: true }).catch(() => {});
    }
    if (state.ctx) await state.ctx.close().catch(() => {});
    state.stream = state.ctx = state.capture = state.playback = state.sid = null;
    state.queue = [];
  }

  async function stop() {
    await teardown();
    $("#live-toggle").textContent = "Start";
    $("#live-status").textContent = "Stopped";
    $("#meter-in").style.width = $("#meter-out").style.width = "0";
  }

  function setEnhanced(value) {
    state.enhanced = value;
    $("#enhance").setAttribute("aria-checked", String(value));
  }

  function init() {
    $("#live-toggle").addEventListener("click", () => (state.running ? stop() : start()));
    $("#enhance").addEventListener("click", () => setEnhanced(!state.enhanced));
    window.addEventListener("pagehide", () => {
      if (state.sid) navigator.sendBeacon(`/api/close?sid=${state.sid}`);
    });
  }

  return { init, stop, isRunning: () => state.running };
})();

// --------------------------------------------------------------------- app
function describeSetup(info) {
  const rule = { wiener: "Wiener gain", lsa: "log-spectral amplitude gain", subtraction: "spectral subtraction" }[info.rule] || info.rule;
  const tracker = { mcra: "MCRA", minstats: "minimum statistics" }[info.estimator] || info.estimator;
  return `${rule}, ${tracker} noise tracking, ${info.frame_ms} ms frames, ${info.fs / 1000} kHz`;
}

async function setTab(name) {
  const tab = name === "live" ? "live" : "replay";
  for (const button of document.querySelectorAll("nav button")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === tab));
  }
  $("#panel-replay").hidden = tab !== "replay";
  $("#panel-live").hidden = tab !== "live";
  if (tab === "live") Replay.pause();
  else if (Live.isRunning()) await Live.stop();
}

async function main() {
  for (const button of document.querySelectorAll("nav button")) {
    button.addEventListener("click", () => {
      location.hash = button.dataset.tab;
    });
  }
  window.addEventListener("hashchange", () => setTab(location.hash.slice(1)));
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea")) return;
    if (event.key === " " && event.target.closest("button")) return;
    if (!$("#panel-replay").hidden) Replay.key(event);
  });
  Live.init();
  try {
    const info = await getJson("/api/info");
    $("#setup").textContent = describeSetup(info);
    $("#s-algo").textContent = `${info.delay_ms.toFixed(1)} ms (${info.delay_samples} samples)`;
    $("#s-block").textContent = `${((1000 * info.block) / info.fs).toFixed(1)} ms (${info.block} samples)`;
  } catch (_) {
    $("#setup").textContent = "Engine not reachable";
  }
  await setTab(location.hash.slice(1));
  await Replay.init();
}

main();
