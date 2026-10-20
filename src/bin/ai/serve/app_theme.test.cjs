// Run with: node --test src/bin/ai/serve/app_theme.test.cjs
// Theme contract of the embedded client: one light palette beside the dark
// one, a toggle in the header, and a stored choice that survives a reload.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const style = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// The two token blocks are the only place a literal palette value may appear.
function styleWithoutTokens() {
  return style.replace(/:root\s*\{[^}]*\}/g, "").replace(/:root\[data-theme="light"\]\s*\{[^}]*\}/g, "");
}

function tokenBlock(selector) {
  const at = style.indexOf(selector);
  assert.ok(at >= 0, "missing token block: " + selector);
  const open = style.indexOf("{", at);
  const close = style.indexOf("}", open);
  return style.slice(open + 1, close);
}

test("the light palette sits beside the dark one and overrides the surfaces", () => {
  const dark = tokenBlock(":root {");
  const light = tokenBlock(':root[data-theme="light"] {');
  for (const name of ["--bg", "--panel", "--line", "--fg", "--dim", "--acc", "--surface", "--code-bg"]) {
    assert.ok(dark.includes(name), "dark palette is missing " + name);
    assert.ok(light.includes(name), "light palette is missing " + name);
    const value = (block) => (block.match(new RegExp(name + ":([^;]+);")) || [])[1];
    assert.notEqual(value(light), value(dark), name + " is identical in both palettes");
  }
  // The light palette must actually be a light surface, not a renamed dark one.
  assert.match(light, /--bg:#f[0-9a-f]{5}/i);
});

test("no hardcoded dark palette value survives outside the token blocks", () => {
  const rest = styleWithoutTokens();
  for (const literal of ["#0d1117", "#161b22", "#21262d", "#1f6feb", "#79c0ff", "#426b94", "#262d36", "#3a424d", "#b6d8ff"]) {
    assert.ok(!rest.includes(literal), "unthemed dark literal left in the stylesheet: " + literal);
  }
});

test("the header carries a theme toggle with an accessible label", () => {
  const button = html.match(/<button id="themebtn"[^>]*>/);
  assert.ok(button, "the header has no #themebtn");
  assert.match(button[0], /aria-label="switch to (light|dark) theme"/);
  // The sun and the moon must both be present: the toggle cross-fades them.
  const svg = html.slice(html.indexOf('<button id="themebtn"'), html.indexOf('</button>', html.indexOf('<button id="themebtn"')));
  assert.match(svg, /class="sun"/);
  assert.match(svg, /class="moon"/);
});

test("the stored theme is applied before the first paint", () => {
  const boot = html.match(/<script id="theme-boot">([\s\S]*?)<\/script>/);
  assert.ok(boot, "the theme boot script is missing");
  assert.match(boot[1], /localStorage\.getItem\("serve\.theme"\)/);
  assert.match(boot[1], /documentElement\.setAttribute\("data-theme"/);
  // It has to sit in <head>, before the stylesheet and the body.
  assert.ok(html.indexOf('id="theme-boot"') < html.indexOf("<style>"), "the theme boot script must precede the stylesheet");
});

test("the theme helpers flip the palette, the label and the status bar", () => {
  const themeSrc = script.slice(script.indexOf('const THEME_KEY'), script.indexOf("function normalizeServeBase("));
  assert.ok(themeSrc.length > 0, "the theme helpers are missing from the client script");
  const attrs = new Map([["data-theme", "dark"]]);
  const store = new Map();
  const meta = { attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } };
  const btn = { attrs: {}, title: "", setAttribute(k, v) { this.attrs[k] = v; } };
  const document = {
    documentElement: {
      getAttribute: (k) => (attrs.has(k) ? attrs.get(k) : null),
      setAttribute: (k, v) => attrs.set(k, v),
    },
    querySelector: (sel) => (sel === 'meta[name="theme-color"]' ? meta : null),
  };
  const localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, v),
  };
  const context = vm.createContext({ document, localStorage, $: (id) => (id === "themebtn" ? btn : null) });
  vm.runInContext(themeSrc, context);

  assert.equal(context.activeTheme(), "dark");
  context.applyTheme("light", true);
  assert.equal(attrs.get("data-theme"), "light");
  assert.equal(store.get("serve.theme"), "light");
  assert.equal(meta.attrs.content, "#ffffff");
  assert.equal(btn.attrs["aria-label"], "switch to dark theme");
  assert.equal(context.activeTheme(), "light");

  // A non-persisting call repaints but keeps the stored choice.
  context.applyTheme("dark", false);
  assert.equal(attrs.get("data-theme"), "dark");
  assert.equal(store.get("serve.theme"), "light");
  assert.equal(meta.attrs.content, "#0d1117");
  assert.equal(btn.attrs["aria-label"], "switch to light theme");
});
