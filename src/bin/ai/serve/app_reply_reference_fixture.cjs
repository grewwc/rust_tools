// Load the production reference helpers shared by partial-page VM fixtures.
const assert = require("node:assert/strict");
const vm = require("node:vm");

function loadReplyReferenceHelpers(context, script) {
  if (vm.runInContext("typeof replyReferenceVersion", context) === "undefined") context.replyReferenceVersion = 0;
  for (const [start, end] of [
    ["function setMessageIdentity(", "function setAnswerRate("],
    ["function textOf(", "// Persisted runtime origin sidecar"],
  ]) {
    const from = script.indexOf(start), to = script.indexOf(end, from);
    assert.ok(from >= 0 && to > from, `reference fixture source: ${start}`);
    vm.runInContext(script.slice(from, to), context);
  }
}

module.exports = { loadReplyReferenceHelpers };
