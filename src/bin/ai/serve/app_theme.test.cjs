// Run with: node --test src/bin/ai/serve/app_theme.test.cjs
// Theme contract of the embedded client: three palettes (dark, light, warm)
// driven by a stored preference that may also be "auto", a header button that
// cycles the palettes, a picker in the settings panel, and a choice that
// survives a reload. The light palettes are held below the luminance of white
// paper on purpose: a near-white panel is what makes a phone screen glare.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const style = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// The token blocks are the only place a literal palette value may appear.
function styleWithoutTokens() {
  return style.replace(/:root(?:\[[^\]]*\])?\s*\{[^}]*\}/g, "");
}

function tokenBlock(selector) {
  const at = style.indexOf(selector);
  assert.ok(at >= 0, "missing token block: " + selector);
  const open = style.indexOf("{", at);
  const close = style.indexOf("}", open);
  return style.slice(open + 1, close);
}

const PALETTES = {
  dark: ":root {",
  light: ':root[data-theme="light"] {',
  warm: ':root[data-theme="warm"] {',
};
const TOKENS = ["--bg", "--panel", "--line", "--fg", "--dim", "--acc", "--surface", "--code-bg"];

function tokenValue(block, name) {
  const m = block.match(new RegExp(name + ":([^;]+);"));
  return m ? m[1].trim() : null;
}

// WCAG relative luminance, for the contrast and the glare checks.
function relativeLuminance(hex) {
  const m = String(hex).match(/^#([0-9a-f]{6})$/i);
  if (!m) return null;
  const n = parseInt(m[1], 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  if (la === null || lb === null) return null;
  const hi = Math.max(la, lb);
  const lo = Math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}

test("three palettes sit side by side and each overrides the surfaces", () => {
  for (const [name, selector] of Object.entries(PALETTES)) {
    const block = tokenBlock(selector);
    for (const t of TOKENS) {
      assert.ok(block.includes(t), name + " palette is missing " + t);
      assert.ok(tokenValue(block, t), name + " palette has no value for " + t);
    }
  }
  const dark = tokenBlock(PALETTES.dark);
  for (const [name, selector] of Object.entries(PALETTES)) {
    if (name === "dark") continue;
    const block = tokenBlock(selector);
    for (const t of TOKENS) {
      assert.notEqual(tokenValue(block, t), tokenValue(dark, t), t + " is identical in dark and " + name);
    }
  }
});

test("the light palettes stay below paper white so the screen does not glare", () => {
  for (const name of ["light", "warm"]) {
    const block = tokenBlock(PALETTES[name]);
    const bg = relativeLuminance(tokenValue(block, "--bg"));
    const panel = relativeLuminance(tokenValue(block, "--panel"));
    assert.ok(bg !== null && panel !== null, name + " palette has an unparsable surface colour");
    // Light enough to read as a light theme, dark enough not to be a lamp.
    assert.ok(bg >= 0.6 && bg <= 0.9, name + " background luminance " + bg.toFixed(3) + " is out of range");
    assert.ok(panel <= 0.9, name + " panel luminance " + panel.toFixed(3) + " is too bright");
    // The card still reads as a raised surface.
    assert.ok(panel > bg, name + " panel is not lighter than its background");
  }
  // The old near-white surfaces must be gone from the whole stylesheet.
  assert.ok(!style.includes("#fbfcfd"), "the near-white panel colour is still in the stylesheet");
});

test("text stays readable in every palette", () => {
  for (const [name, selector] of Object.entries(PALETTES)) {
    const block = tokenBlock(selector);
    const bg = tokenValue(block, "--bg");
    const body = contrast(tokenValue(block, "--fg"), bg);
    const dim = contrast(tokenValue(block, "--dim"), bg);
    const acc = contrast(tokenValue(block, "--acc"), bg);
    assert.ok(body >= 7, name + " body text contrast " + body.toFixed(2) + " is too low");
    assert.ok(dim >= 4.5, name + " secondary text contrast " + dim.toFixed(2) + " is too low");
    assert.ok(acc >= 3, name + " accent contrast " + acc.toFixed(2) + " is too low");
  }
});

test("no palette literal survives outside the token blocks", () => {
  const rest = styleWithoutTokens();
  for (const selector of Object.values(PALETTES)) {
    const block = tokenBlock(selector);
    for (const m of block.matchAll(/(--[a-z0-9-]+):(#[0-9a-f]{3,8});/gi)) {
      assert.ok(!rest.includes(m[2]), "palette literal " + m[2] + " left in the stylesheet");
    }
  }
});

test("the header carries a theme toggle with one icon per palette", () => {
  const button = html.match(/<button id="themebtn"[^>]*>/);
  assert.ok(button, "the header has no #themebtn");
  assert.match(button[0], /aria-label="[^"]+"/);
  const svg = html.slice(html.indexOf('<button id="themebtn"'), html.indexOf('</button>', html.indexOf('<button id="themebtn"')));
  assert.match(svg, /class="sun"/);
  assert.match(svg, /class="moon"/);
  assert.match(svg, /class="leaf"/);
  // Each palette shows exactly one icon.
  for (const theme of ["light", "warm"]) {
    const shown = style.match(new RegExp(':root\\[data-theme="' + theme + '"\\] #themebtn \\.(\\w+) \\{ opacity:1', "g")) || [];
    assert.equal(shown.length, 1, theme + " must show exactly one icon");
  }
});

test("the settings panel can pick a theme directly", () => {
  const pick = html.match(/<div id="themepick"[\s\S]*?<\/div>/);
  assert.ok(pick, "the settings panel has no #themepick");
  for (const choice of ["auto", "dark", "light", "warm"]) {
    assert.ok(pick[0].includes('data-theme-choice="' + choice + '"'), "the picker is missing " + choice);
  }
});

test("the stored theme is applied before the first paint", () => {
  const boot = html.match(/<script id="theme-boot">([\s\S]*?)<\/script>/);
  assert.ok(boot, "the theme boot script is missing");
  assert.match(boot[1], /localStorage\.getItem\("serve\.theme"\)/);
  assert.match(boot[1], /documentElement\.setAttribute\("data-theme"/);
  // It has to sit in <head>, before the stylesheet and the body.
  assert.ok(html.indexOf('id="theme-boot"') < html.indexOf("<style>"), "the theme boot script must precede the stylesheet");
  // "auto" and "warm" both have to survive the boot script.
  assert.match(boot[1], /"auto"/);
  assert.match(boot[1], /"warm"/);
});

function themeContext(systemLight) {
  const attrs = new Map([["data-theme", "dark"]]);
  const store = new Map();
  const meta = { attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } };
  const btn = { attrs: {}, title: "", setAttribute(k, v) { this.attrs[k] = v; } };
  const pickButtons = ["auto", "dark", "light", "warm"].map((choice) => ({
    choice,
    attrs: {},
    getAttribute: (k) => (k === "data-theme-choice" ? choice : null),
    setAttribute(k, v) { this.attrs[k] = v; },
  }));
  const document = {
    documentElement: {
      getAttribute: (k) => (attrs.has(k) ? attrs.get(k) : null),
      setAttribute: (k, v) => attrs.set(k, v),
    },
    querySelector: (sel) => (sel === 'meta[name="theme-color"]' ? meta : null),
    querySelectorAll: (sel) => (sel === "#themepick button" ? pickButtons : []),
  };
  const localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, v),
  };
  const mq = { matches: !!systemLight, addEventListener() {} };
  const context = vm.createContext({
    document,
    localStorage,
    window: { matchMedia: () => mq },
    $: (id) => (id === "themebtn" ? btn : null),
  });
  const themeSrc = script.slice(script.indexOf("const THEME_KEY"), script.indexOf("function normalizeServeBase("));
  assert.ok(themeSrc.length > 0, "the theme helpers are missing from the client script");
  vm.runInContext(themeSrc, context);
  return { context, attrs, store, meta, btn, pickButtons };
}

test("the theme helpers paint the palette, the label and the status bar", () => {
  const { context, attrs, store, meta, btn } = themeContext(false);
  assert.equal(context.activeTheme(), "dark");
  context.applyTheme("light", true);
  assert.equal(attrs.get("data-theme"), "light");
  assert.equal(attrs.get("data-theme-pref"), "light");
  assert.equal(store.get("serve.theme"), "light");
  assert.equal(meta.attrs.content, "#e4e8ed");
  assert.match(btn.attrs["aria-label"], /浅色/);
  assert.equal(context.activeTheme(), "light");

  context.applyTheme("warm", true);
  assert.equal(attrs.get("data-theme"), "warm");
  assert.equal(store.get("serve.theme"), "warm");
  assert.equal(meta.attrs.content, "#e6dfd0");

  // A non-persisting call repaints but keeps the stored choice.
  context.applyTheme("dark", false);
  assert.equal(attrs.get("data-theme"), "dark");
  assert.equal(store.get("serve.theme"), "warm");
  assert.equal(meta.attrs.content, "#0d1117");
});

test("the header button cycles through the three palettes", () => {
  const { context } = themeContext(false);
  assert.equal(context.nextTheme(), "light");
  context.applyTheme("light", true);
  assert.equal(context.nextTheme(), "warm");
  context.applyTheme("warm", true);
  assert.equal(context.nextTheme(), "dark");
});

test("auto follows the system and keeps its own preference", () => {
  const light = themeContext(true);
  light.context.applyTheme("auto", true);
  assert.equal(light.attrs.get("data-theme"), "light");
  assert.equal(light.attrs.get("data-theme-pref"), "auto");
  assert.equal(light.store.get("serve.theme"), "auto");

  const dark = themeContext(false);
  dark.context.applyTheme("auto", true);
  assert.equal(dark.attrs.get("data-theme"), "dark");
  assert.equal(dark.attrs.get("data-theme-pref"), "auto");
});

test("the picker marks the stored preference as pressed", () => {
  const { context, pickButtons } = themeContext(false);
  context.applyTheme("warm", true);
  const pressed = pickButtons.filter((b) => b.attrs["aria-pressed"] === "true").map((b) => b.choice);
  assert.deepEqual(pressed, ["warm"]);
});
