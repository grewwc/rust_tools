// Run with: node --test src/bin/ai/serve/app_math_fit.test.cjs
// Synthetic layout metrics guard renderer mutations; the browser evidence is
// recorded separately with the actual KaTeX DOM, app CSS and installed fonts.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createHash } = require("node:crypto");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const renderer = script.slice(script.indexOf("const esc ="), script.indexOf("function sessionTitle("));
const tex = String.raw`\text{购进设备抵扣 }13\%\ -\ \text{出租销项 }9\%\ =\ \text{倒挂 }4\%`;

function harness({ viewport = 375, intrinsicWidth = 420, display = true, observer = true } = {}) {
  const frames = [], listeners = {}, fontListeners = {}, observers = [], calls = [];
  let resolveFonts;
  const fontsReady = new Promise((resolve) => { resolveFonts = resolve; });
  function element(tag = "span") {
    return {
      tag, style: {}, attrs: {}, children: [], parentNode: null, textContent: "", className: "",
      setAttribute(name, value) { this.attrs[name] = String(value); },
      getAttribute(name) { return this.attrs[name] ?? null; },
      removeAttribute(name) { delete this.attrs[name]; },
      insertBefore(child, next) {
        child.parentNode = this;
        const index = this.children.indexOf(next);
        this.children.splice(index < 0 ? this.children.length : index, 0, child);
        return child;
      },
      remove() {
        if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1);
        this.parentNode = null;
      },
      querySelector() { return null; },
    };
  }
  const parent = element("div"), node = element();
  parent.children.push(node);
  node.parentNode = parent;
  node.dataset = { tex };
  node.textContent = (display ? "$$" : "$") + tex + (display ? "$$" : "$");
  node.classList = { contains: (name) => name === (display ? "display" : "inline") };
  node.querySelector = (selector) => selector === ".katex-display" && display && node.typeset ? node.typeset : null;
  node.scrollLeft = 0;
  let width = viewport - 48, naturalWidth = intrinsicWidth;
  Object.defineProperties(node, {
    clientWidth: { get: () => width },
    scrollWidth: { get: () => Math.max(width, Math.ceil(naturalWidth * (parseFloat(node.style.fontSize || "16") / 16))) },
    nextSibling: { get: () => parent.children[parent.children.indexOf(node) + 1] || null },
  });
  const root = {
    clientWidth: viewport, scrollTop: 0, scrollHeight: 200, clientHeight: 100,
    querySelector: () => "tex" in node.dataset ? node : null,
    querySelectorAll(selector) {
      if (selector === ".math[data-tex]") return "tex" in node.dataset ? [node] : [];
      if (selector === ".math.display") return display ? [node] : [];
      return [];
    },
  };
  const context = vm.createContext({
    document: { createElement: element, fonts: { ready: fontsReady, addEventListener(name, fn) { fontListeners[name] = fn; } } },
    window: {
      matchMedia: () => ({ matches: viewport >= 701 }),
      getComputedStyle: () => ({ fontSize: "16px" }),
      requestAnimationFrame: (fn) => { frames.push(fn); return frames.length; },
      addEventListener(name, fn) { listeners[name] = fn; },
      katex: { render(source, target, options) {
        calls.push({ tex: source, options });
        target.textContent = "typeset:" + source;
        target.typeset = { textContent: target.textContent };
      } },
    },
    $: () => root,
    ...(observer ? { ResizeObserver: class {
      constructor(callback) { this.callback = callback; this.targets = []; observers.push(this); }
      observe(target) { this.targets.push(target); }
    } } : {}),
  });
  vm.runInContext(renderer, context);
  const hints = () => parent.children.filter((child) => child.className === "math-scroll-hint");
  const flush = () => { while (frames.length) frames.shift()(); };
  const render = () => { context.typesetMath(root); flush(); };
  const resize = (viewportWidth) => {
    width = viewportWidth - 48;
    root.clientWidth = viewportWidth;
    if (observers.length) observers[0].callback();
    else listeners.resize?.();
    flush();
  };
  return { context, node, parent, root, calls, hints, render, resize, flush, frames, listeners, fontListeners, observers,
    setWidth(value) { width = value; },
    setIntrinsicWidth(value) { naturalWidth = value; },
    async finishFonts() { resolveFonts(); await fontsReady; await Promise.resolve(); flush(); },
  };
}

test("reported short formula at 440 CSS pixels retains paragraph sizing without a hint", () => {
  const h = harness({ viewport: 440, intrinsicWidth: 382 });
  // Chromium with the app CSS and KaTeX 0.19: default 1.21em yields 377/382.
  h.setWidth(377);
  h.render();
  assert.equal(h.node.style.fontSize || "", "", "a short equation must not acquire a fitted font size");
  assert.equal(h.hints().length, 0, "a short equation must not acquire an extra hint row");
  assert.match(html, /\.msg\.md \.math \.katex \{ font-size:1em; line-height:inherit; \}/,
    "KaTeX must use the surrounding paragraph size rather than its default 1.21em");
});

test("375px display formula preserves its size and exposes the complete tail by scrolling", () => {
  const h = harness();
  const markup = h.context.renderMarkdown("$$" + tex + "$$");
  assert.match(markup, /倒挂/);
  assert.match(markup, /4\\%/);
  h.render();
  assert.equal(h.node.scrollWidth, 420, "rendering must preserve the formula's natural width");
  assert.ok(h.node.scrollWidth > h.node.clientWidth, "real overflow remains in the scrollport");
  assert.equal(h.node.style.fontSize, undefined);
  assert.equal(h.node.style.transform, undefined);
  assert.equal(h.hints().length, 0);
  assert.equal(h.node.textContent, "typeset:" + tex);
  assert.equal(h.node.typeset.textContent, "typeset:" + tex, "KaTeX output remains intact");
  assert.equal(h.calls[0].tex, tex);
  h.node.scrollLeft = h.node.scrollWidth - h.node.clientWidth;
  assert.equal(h.node.scrollLeft + h.node.clientWidth, h.node.scrollWidth, "scrolling exposes the entire tail");
});

test("375px very long formula retains its full size and scroll range without a hint row", () => {
  const h = harness({ intrinsicWidth: 640 });
  h.render();
  assert.equal(h.node.style.fontSize, undefined);
  assert.equal(h.node.scrollWidth, 640);
  assert.ok(h.node.scrollWidth > h.node.clientWidth);
  assert.equal(h.hints().length, 0);
  assert.equal(h.parent.children.length, 1, "no extra row, margin or note is inserted");
  assert.deepEqual(h.node.style, {}, "the renderer must not override size, line height or theme color");
  assert.equal(h.node.getAttribute("tabindex"), null);
  assert.equal(h.node.getAttribute("aria-describedby"), null);
  assert.doesNotMatch(script, /math-scroll-hint|公式较长/);
  h.node.scrollLeft = h.node.scrollWidth - h.node.clientWidth;
  assert.equal(h.node.scrollLeft + h.node.clientWidth, h.node.scrollWidth, "scrolling exposes the entire tail");
  h.render();
  assert.equal(h.hints().length, 0, "repeated painting must not insert a hint");
  assert.equal(h.calls.length, 1, "repeated painting must not typeset again");
});

for (const [viewport, intrinsicWidth, display] of [[1280, 420, true], [375, 280, true], [375, 640, false]]) {
  test(`${viewport}px ${display ? "display" : "inline"} formula without fitting stays identical`, () => {
    const h = harness({ viewport, intrinsicWidth, display });
    h.render();
    assert.deepEqual(h.node.style, {});
    assert.deepEqual(h.node.attrs, {});
    assert.equal(h.parent.children.length, 1);
    assert.equal(h.node.textContent, "typeset:" + tex);
    assert.equal(h.node.typeset.textContent, "typeset:" + tex);
    assert.equal(h.calls[0].options.displayMode, display);
    assert.deepEqual(JSON.parse(JSON.stringify(h.calls[0].options)), {
      displayMode: display, throwOnError: true, trust: false, strict: "ignore", maxExpand: 1000, maxSize: 20,
    });
  });
}

test("rotation changes only available width without changing size or rendered content", () => {
  const h = harness({ intrinsicWidth: 640 });
  h.render();
  assert.equal(h.observers.length, 0, "CSS overflow does not need a layout observer");
  assert.deepEqual(h.frames, []);
  h.resize(1280);
  assert.equal(h.node.style.fontSize || "", "");
  assert.deepEqual(h.node.attrs, {});
  assert.equal(h.hints().length, 0);
  h.resize(375);
  assert.equal(h.node.style.fontSize || "", "");
  assert.equal(h.node.scrollWidth, 640);
  h.resize(375);
  assert.equal(h.node.style.fontSize || "", "");
  assert.equal(h.hints().length, 0);
  assert.equal(h.node.typeset.textContent, "typeset:" + tex);
  assert.equal(h.calls.length, 1);
});

test("font readiness and later font loads leave sizing and content to CSS", async () => {
  const h = harness({ intrinsicWidth: 280 });
  h.render();
  h.setIntrinsicWidth(640);
  await h.finishFonts();
  assert.equal(h.hints().length, 0);
  assert.equal(h.node.style.fontSize, undefined);
  assert.equal(h.node.scrollWidth, 640);
  h.setIntrinsicWidth(420);
  h.fontListeners.loadingdone?.();
  h.flush();
  assert.equal(h.hints().length, 0);
  assert.equal(h.node.scrollWidth, 420);
  assert.deepEqual(h.node.style, {});
  assert.equal(h.node.typeset.textContent, "typeset:" + tex);
  assert.equal(h.calls.length, 1);
});

test("a zero-width formula renders intact and gains no layout work when revealed", () => {
  const h = harness({ observer: false });
  h.setWidth(0);
  h.render();
  assert.deepEqual(h.node.style, {});
  assert.equal(h.hints().length, 0);
  h.resize(375);
  assert.equal(h.node.scrollWidth, 420);
  assert.ok(h.node.clientWidth > 0);
  assert.equal(h.listeners.resize, undefined);
  assert.deepEqual(h.node.style, {});
  assert.equal(h.frames.length, 0);
  h.flush();
  assert.equal(h.node.typeset.textContent, "typeset:" + tex);
  assert.equal(h.calls.length, 1);
});

for (const viewport of [375, 414, 440, 880, 1280]) {
  for (const [name, source, display, intrinsicWidth] of [
    ["short", tex, true, 280],
    ["Chinese inline with minus and equals", String.raw`\text{税率}13\%−9\%=4\%`, false, 180],
    ["multiline display", String.raw`\begin{aligned}x&=13\%\\y&=9\%\\x-y&=4\%\end{aligned}`, true, 180],
  ]) {
    test(`${viewport}px ${name} keeps its TeX, font size, line height and paragraph spacing`, () => {
      const h = harness({ viewport, intrinsicWidth, display });
      h.node.dataset.tex = source;
      h.node.textContent = (display ? "$$" : "$") + source + (display ? "$$" : "$");
      const markup = h.context.renderMarkdown(h.node.textContent);
      assert.equal((markup.match(/class="math /g) || []).length, 1);
      h.render();
      assert.deepEqual(h.node.style, {});
      assert.deepEqual(h.node.attrs, {});
      assert.equal(h.hints().length, 0);
      assert.equal(h.node.scrollWidth, h.node.clientWidth);
      assert.equal(h.node.textContent, "typeset:" + source);
      assert.equal(h.node.typeset.textContent, "typeset:" + source);
      assert.equal(h.calls[0].tex, source);
      assert.equal(h.calls[0].options.displayMode, display);
      assert.equal(h.parent.children.length, 1);
    });
  }
}

test("formula overflow and paragraph typography apply across every theme", () => {
  assert.match(html, /\.msg\.md \.math \{ max-width:100%; overflow-x:auto; word-break:normal; \}/);
  assert.match(html, /\.msg\.md \.math \.katex \{ font-size:1em; line-height:inherit; \}/);
  assert.match(html, /\.msg\.md \.math\.display:has\(\.vlist\) \{ padding-block:\.5em; \}/,
    "multiline fractions need space for glyphs beyond their strut without changing plain formulas");
  assert.doesNotMatch(script, /fitDisplayMath|queueMathLayout|watchMathLayout|math-scroll-hint/);
});

test("CSS snapshot pins paragraph-sized math and the existing mobile and desktop themes", () => {
  const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
  assert.equal(Buffer.byteLength(css), 89411);
  assert.equal(createHash("sha256").update(css).digest("hex"),
    "4513025a70f727f0550515052eb627608073ffb3a4f79c38c26b152758b3b2ad");
});
