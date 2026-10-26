// Run with: node --test src/bin/ai/serve/app_fold.test.cjs
// The collapse gesture is a DOM decision: two quick taps at one spot, then the
// nearest open <details> above the finger. Single taps, slow taps, keyboard
// activation and controls inside the fold must keep their existing behavior.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function harness() {
  const clock = { now: 1000 };
  const listeners = new Map();
  const msgs = {
    addEventListener(type, fn) { listeners.set(type, (listeners.get(type) || []).concat(fn)); },
  };
  const context = vm.createContext({
    $: (id) => (id === "msgs" ? msgs : null),
    Date: { now: () => clock.now },
  });
  vm.runInContext(script.slice(script.indexOf("let lastFoldTap"), script.indexOf("// The agent sends a phone action")), context);
  // wire() itself needs the whole page, so only its transcript registrations run.
  // `$("jumpbottom")` also appears in earlier functions, so the slice ends at the
  // first occurrence after the transcript lines.
  const wireStart = script.indexOf('$("msgs").addEventListener("click", handlePreviewClick);');
  vm.runInContext("function handlePreviewClick() {}\nfunction scheduleJumpButtonUpdate() {}\n" +
    script.slice(wireStart, script.indexOf('$("jumpbottom")', wireStart)), context);
  return {
    clock,
    handlers: (type) => listeners.get(type) || [],
    tap(target, x, y, detail = 1) {
      for (const fn of listeners.get("click") || []) fn({ target, clientX: x, clientY: y, detail });
    },
  };
}

function node(tagName) {
  return {
    tagName, parentNode: null, open: false,
    closest(selector) {
      const names = selector.split(",").map((part) => part.trim().toUpperCase());
      for (let cur = this; cur; cur = cur.parentNode) if (names.includes(cur.tagName)) return cur;
      return null;
    },
  };
}

function makeFold(open) {
  const card = node("DIV");
  const details = node("DETAILS");
  details.open = open;
  details.parentNode = card;
  const body = node("DIV");
  body.parentNode = details;
  const summary = node("SUMMARY");
  summary.parentNode = details;
  return { card, details, body, summary };
}

function doubleTap(h, target) {
  h.tap(target, 100, 200);
  h.clock.now += 100;
  h.tap(target, 101, 201);
}

test("a double tap on an open fold collapses it", () => {
  const h = harness();
  const fold = makeFold(true);
  doubleTap(h, fold.body);
  assert.equal(fold.details.open, false);
});

test("single, slow and far-apart taps leave the fold open", () => {
  const h = harness();
  const fold = makeFold(true);
  h.tap(fold.body, 100, 200);
  assert.equal(fold.details.open, true);
  h.clock.now += 1200; // beyond the double-tap window
  h.tap(fold.body, 100, 200);
  assert.equal(fold.details.open, true);
  h.clock.now += 100;
  h.tap(fold.body, 400, 200); // inside the window, but elsewhere
  assert.equal(fold.details.open, true);
});

test("a closed fold stays closed and keyboard activation is ignored", () => {
  const h = harness();
  const closed = makeFold(false);
  doubleTap(h, closed.body);
  assert.equal(closed.details.open, false);
  const open = makeFold(true);
  h.clock.now += 100;
  h.tap(open.body, 100, 200, 0);
  h.clock.now += 100;
  h.tap(open.body, 100, 200, 0);
  assert.equal(open.details.open, true);
});

test("taps inside a fold's controls keep their own meaning", () => {
  const h = harness();
  const fold = makeFold(true);
  const button = node("BUTTON");
  button.parentNode = fold.body;
  const summary = fold.summary;
  doubleTap(h, button);
  doubleTap(h, summary);
  assert.equal(fold.details.open, true);
});

test("the innermost open fold collapses first", () => {
  const h = harness();
  const outer = makeFold(true);
  const inner = makeFold(true);
  inner.card.parentNode = outer.body;
  doubleTap(h, inner.body);
  assert.equal(inner.details.open, false);
  assert.equal(outer.details.open, true);
  doubleTap(h, inner.body);
  assert.equal(outer.details.open, false);
});

test("wire() registers the gesture and folds do not double-tap zoom", () => {
  const h = harness();
  const names = h.handlers("click").map((fn) => fn.name);
  assert.ok(names.includes("handlePreviewClick"), names.join(","));
  assert.ok(names.includes("collapseFoldForTap"), names.join(","));
  assert.match(html, /\.msg details \{[^}]*touch-action:\s*manipulation/);
});