// Run with: node --test src/bin/ai/serve/app_fold.test.cjs
// The collapse gesture is a DOM decision: two quick taps at one spot, then the
// nearest open <details> above the finger. Desktop subagent summaries keep the
// first native toggle, while phones retain both native summary clicks.
// Single taps, slow taps, keyboard activation and controls keep their behavior.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function viewportTest(name, fn) {
  for (const width of [390, 700, 701, 1280]) test(name + " (" + width + "px)", () => fn(width));
}

function harness({ width = 1280 } = {}) {
  const clock = { now: 1000 };
  const viewport = { width };
  const listeners = new Map();
  const msgs = {
    addEventListener(type, fn) { listeners.set(type, (listeners.get(type) || []).concat(fn)); },
  };
  const context = vm.createContext({
    $: (id) => (id === "msgs" ? msgs : null),
    Date: { now: () => clock.now },
    window: {
      addEventListener() {},
      matchMedia(query) {
        assert.equal(query, "(min-width:701px)");
        return { matches: viewport.width >= 701 };
      },
    },
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
    viewport,
    handlers: (type) => listeners.get(type) || [],
    tap(target, x, y, detail = 1) {
      const event = {
        target, clientX: x, clientY: y, detail, defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; },
      };
      for (const fn of listeners.get("click") || []) fn(event);
      // Summary activation is the click default action, after event bubbling.
      // Keyboard-generated clicks use detail=0 and activate the same action.
      const summary = target.closest("summary");
      if (summary && !target.closest("button, a, input, select, textarea") && !event.defaultPrevented) {
        summary.parentNode.open = !summary.parentNode.open;
      }
      return event;
    },
  };
}

function node(tagName) {
  return {
    tagName, parentNode: null, open: false, className: "", dataset: {},
    closest(selector) {
      for (let cur = this; cur; cur = cur.parentNode) {
        if (selector.split(",").some((part) => {
          const tag = /^[\w-]+/.exec(part.trim())?.[0];
          const classes = [...part.matchAll(/\.([\w-]+)/g)].map((match) => match[1]);
          return (!tag || cur.tagName === tag.toUpperCase()) &&
            classes.every((name) => cur.className.split(/\s+/).includes(name));
        })) return cur;
      }
      return null;
    },
  };
}

function makeFold(open, subagents = false) {
  const card = node("DIV");
  card.className = "msg asst";
  if (subagents) card.dataset.subagents = "true";
  const details = node("DETAILS");
  details.open = open;
  details.parentNode = card;
  const body = node("DIV");
  body.parentNode = details;
  const summary = node("SUMMARY");
  summary.parentNode = details;
  card.querySelector = (selector) => selector === "details" ? details : null;
  return { card, details, body, summary };
}

function doubleTap(h, target) {
  h.tap(target, 100, 200);
  h.clock.now += 100;
  h.tap(target, 101, 201);
}

function doubleClick(h, target) {
  const first = h.tap(target, 100, 200, 1);
  h.clock.now += 100;
  const second = h.tap(target, 100, 200, 2);
  const event = {
    target, clientX: 100, clientY: 200, detail: 2, defaultPrevented: false,
    preventDefault() { this.defaultPrevented = true; },
  };
  for (const fn of h.handlers("dblclick")) fn(event);
  return { first, second, double: event };
}

viewportTest("an expanded subagent summary keeps desktop collapse and native mobile toggles", (width) => {
  const h = harness({ width }), fold = makeFold(true, true), label = node("SPAN");
  label.parentNode = fold.summary;
  const events = doubleClick(h, label);
  assert.equal(fold.details.open, width <= 700, "only desktop suppresses the second native summary toggle");
  assert.equal(events.first.defaultPrevented, false);
  assert.equal(events.second.defaultPrevented, width >= 701);
  assert.equal(events.double.defaultPrevented, false, "double-click selection keeps its default action");
});

viewportTest("a closed subagent summary double-click toggles once on desktop and twice on mobile", (width) => {
  const h = harness({ width }), fold = makeFold(false, true);
  const events = doubleClick(h, fold.summary);
  assert.equal(fold.details.open, width >= 701);
  assert.equal(events.first.defaultPrevented, false);
  assert.equal(events.second.defaultPrevented, width >= 701);
});

viewportTest("an expanded task card collapses on a double tap in its report rows", (width) => {
  const h = harness({ width }), fold = makeFold(true);
  fold.card.dataset.taskOutput = "true";
  // Task rows live in .task-output next to <details>, not inside it.
  const row = node("DIV");
  row.parentNode = fold.card;
  doubleTap(h, row);
  assert.equal(fold.details.open, width <= 700, "desktop closes the card; phones keep it open");
});

viewportTest("subagent summary single clicks and keyboard activation retain native toggles", (width) => {
  const h = harness({ width }), fold = makeFold(false, true);
  for (const [detail, expected] of [[1, true], [1, false], [0, true], [0, false]]) {
    h.clock.now += 1000;
    assert.equal(h.tap(fold.summary, 100, 200, detail).defaultPrevented, false);
    assert.equal(fold.details.open, expected);
  }
  const button = node("BUTTON"); button.parentNode = fold.summary;
  const events = doubleClick(h, button);
  assert.equal(fold.details.open, false);
  assert.equal(events.second.defaultPrevented, false, "nested controls retain their own click behavior");
});

viewportTest("ordinary fold summaries retain both native clicks", (width) => {
  const h = harness({ width }), fold = makeFold(true);
  const events = doubleClick(h, fold.summary);
  assert.equal(fold.details.open, true);
  assert.equal(events.first.defaultPrevented, false);
  assert.equal(events.second.defaultPrevented, false);
});

viewportTest("a double tap on an open fold collapses it", (width) => {
  const h = harness({ width });
  const fold = makeFold(true);
  doubleTap(h, fold.body);
  assert.equal(fold.details.open, false);
});

viewportTest("single, slow and far-apart taps leave the fold open", (width) => {
  const h = harness({ width });
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

viewportTest("a closed fold stays closed and keyboard activation is ignored", (width) => {
  const h = harness({ width });
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

viewportTest("taps inside a fold's controls keep their own meaning", (width) => {
  const h = harness({ width });
  const fold = makeFold(true);
  const button = node("BUTTON");
  button.parentNode = fold.body;
  const summary = fold.summary;
  doubleTap(h, button);
  doubleTap(h, summary);
  assert.equal(fold.details.open, true);
});

viewportTest("the innermost open fold collapses first", (width) => {
  const h = harness({ width });
  const outer = makeFold(true);
  const inner = makeFold(true);
  inner.card.parentNode = outer.body;
  doubleTap(h, inner.body);
  assert.equal(inner.details.open, false);
  assert.equal(outer.details.open, true);
  doubleTap(h, inner.body);
  assert.equal(outer.details.open, false);
});

viewportTest("subagent sibling rows keep the old ancestor collapse owner on mobile", (width) => {
  const h = harness({ width }), outer = makeFold(true), inner = makeFold(true, true), row = node("DIV");
  inner.card.parentNode = outer.body;
  row.parentNode = inner.card;
  doubleTap(h, row);
  assert.equal(inner.details.open, width <= 700);
  assert.equal(outer.details.open, width >= 701, "mobile must retain the pre-existing ancestor search");
});

test("subagent summary gating follows viewport changes across 700px and 701px", () => {
  const h = harness({ width: 701 }), fold = makeFold(true, true);
  doubleClick(h, fold.summary);
  assert.equal(fold.details.open, false);
  h.viewport.width = 700;
  fold.details.open = true;
  assert.equal(doubleClick(h, fold.summary).second.defaultPrevented, false);
  assert.equal(fold.details.open, true);
  h.viewport.width = 701;
  assert.equal(doubleClick(h, fold.summary).second.defaultPrevented, true);
  assert.equal(fold.details.open, false);
});

viewportTest("wire() registers the gesture and folds do not double-tap zoom", (width) => {
  const h = harness({ width });
  const names = h.handlers("click").map((fn) => fn.name);
  assert.ok(names.includes("handlePreviewClick"), names.join(","));
  assert.ok(names.includes("collapseFoldForTap"), names.join(","));
  assert.match(html, /\.msg details \{[^}]*touch-action:\s*manipulation/);
});
