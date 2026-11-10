"use strict";
// The plain wording for Web Serial errors.
const assert = require("node:assert/strict");
const { explainSerialError } = require("../../demo/web/engines.js");

function failure(name, message) {
  const error = new Error(message);
  error.name = name;
  return error;
}

const opened = explainSerialError(
  failure("NetworkError", "Failed to execute 'open' on 'SerialPort': Failed to open serial port.")
);
assert.match(opened, /Close the Arduino Serial Monitor/);
assert.match(opened, /unplug and replug/);
assert.match(explainSerialError(failure("NotFoundError", "Failed to execute 'requestPort' on 'Serial': No port selected by the user.")), /No port was selected/);
assert.match(explainSerialError(failure("InvalidStateError", "The port is already open.")), /already open/);
assert.match(explainSerialError(failure("SecurityError", "Must be handling a user gesture.")), /needs a click/);
assert.match(explainSerialError(failure("NetworkError", "The device has been lost.")), /port could not be opened|disconnected/);
assert.equal(explainSerialError(failure("NotFoundError", "Requested device not found")), null);
assert.equal(explainSerialError(new Error("something else")), null);
assert.equal(explainSerialError(null), null);
for (const text of [opened]) assert.ok(/^[\x20-\x7e]+$/.test(text) && !text.includes("\u2014"));
console.log("errors: ok");
