// Run with: node --test src/bin/ai/serve/app_shell.test.cjs
// The mobile shell must fit the viewport the WebView actually shows. The
// document never scrolls, the message list is the only scroller, and the
// measured height follows the keyboard and the system bars.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const boot = html.match(/<script id="viewport-boot">([\s\S]*?)<\/script>/)[1];

function loadBoot(innerHeight, viewport) {
  const props = {};
  const listeners = {};
  const root = { style: { setProperty: (name, value) => { props[name] = value; } } };
  const window = {
    innerHeight,
    visualViewport: viewport,
    addEventListener: (type, fn) => { listeners[type] = fn; },
  };
  vm.runInContext(boot, vm.createContext({ window, document: { documentElement: root } }));
  return { props, listeners, window };
}

test("the measured height starts from window.innerHeight", () => {
  const { props } = loadBoot(812, null);
  assert.equal(props["--app-height"], "812px");
});

test("an open keyboard shrinks the shell to the visible viewport", () => {
  const { props } = loadBoot(812, { scale: 1, height: 512, addEventListener() {} });
  assert.equal(props["--app-height"], "512px");
});

test("pinch-zoom does not resize the layout", () => {
  const { props } = loadBoot(812, { scale: 1.7, height: 400, addEventListener() {} });
  assert.equal(props["--app-height"], "812px");
});

test("a window resize re-measures the available height", () => {
  const { props, listeners, window } = loadBoot(812, null);
  window.innerHeight = 600;
  listeners.resize();
  assert.equal(props["--app-height"], "600px");
});

test("the document cannot scroll and the message list fills its wrapper", () => {
  assert.match(css, /html \{ height:100%; overflow:hidden; \}/);
  assert.match(css, /body \{[^}]*overflow:hidden;[^}]*height:100vh; height:100dvh; height:var\(--app-height, 100dvh\);/);
  assert.match(css, /#msgwrap \{ flex:1; min-height:0; position:relative; \}/);
  assert.match(css, /#msgs \{ position:absolute; top:0; right:0; bottom:0; left:0;[^}]*overflow-y:auto;/);
});

test("the header keeps clear of a status bar or notch", () => {
  assert.match(css, /header \{[^}]*padding:10px 12px; padding-top:calc\(10px \+ env\(safe-area-inset-top\)\);/);
});

test("the model sheet is capped by the measured viewport, not dvh", () => {
  const panel = css.match(/#modelpickpanel \{[^}]*\}/)[0];
  assert.match(panel, /max-height:calc\(var\(--app-height, 100vh\) \* \.82\);/);
  assert.match(css, /#modelpicklist \{ flex:1; min-height:0; overflow-y:auto;/);
  assert.match(css, /#modelpickpanel \{ max-height:calc\(var\(--app-height, 100vh\) \* \.88\); \}/);
});

test("the composer and dialogs cap by the same measured height", () => {
  assert.match(css, /max-height:calc\(var\(--app-height, 100vh\) \* \.30\)/);
  assert.match(css, /max-height:calc\(var\(--app-height, 100vh\) \* \.38\)/);
  assert.match(css, /max-height:calc\(var\(--app-height, 100vh\) \* \.80\)/);
});
