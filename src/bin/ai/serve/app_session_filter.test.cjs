// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_session_filter.test.cjs
// Reproduce the stale composition state that leaves a visible filter unsearched.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const dir = __dirname;
const fixture = path.join(dir, "app_mark.test.cjs");
const exported = new vm.Script(
  fs.readFileSync(fixture, "utf8") + "\n;({ harness });",
  { filename: fixture },
).runInNewContext({
  require: (name) => name === "node:test" ? { test() {} } : require(name),
  __dirname: dir,
  __filename: fixture,
  AbortController,
  URL,
  setImmediate,
});

const { test } = require("node:test");
const assert = require("node:assert/strict");

test("a non-composing input commits the filter after a lost compositionend", async () => {
  const items = [
    { id: "match", summary: "优化 session list", marked: false, size_bytes: 1 },
    { id: "other", summary: "WebUI actions", marked: false, size_bytes: 1 },
  ];
  const h = exported.harness({ items });
  const filter = h.elements.sessfilter;

  h.dispatch(filter, "compositionstart");
  filter.value = "youhua";
  h.dispatch(filter, "input", { isComposing: true, inputType: "insertCompositionText", data: "youhua" });
  assert.equal(h.requests.length, 0, "preedit must not trigger a search");

  // Chrome can omit compositionend when focus leaves the field; the final
  // input still contains the committed candidate and is not marked composing.
  filter.value = "优化";
  h.dispatch(filter, "input", { isComposing: false, inputType: "insertText", data: "优化" });
  h.advance(250);
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].searchParams.get("query"), "优化");
  assert.deepEqual(JSON.parse(JSON.stringify(h.context.lastItems.map((item) => item.id))), ["match"]);
});