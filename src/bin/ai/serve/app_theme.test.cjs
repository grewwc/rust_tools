// Run with: node --test src/bin/ai/serve/app_theme.test.cjs
// Theme contract of the embedded client: three palettes (dark, light, warm)
// driven by a stored preference that may also be "auto", a header button that
// cycles the palettes, a picker in the settings panel, and a choice that
// survives a reload. The light palettes are held below the luminance of white
// paper on purpose: a near-white panel is what makes a phone screen glare.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const style = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// The token blocks are the only place a literal palette value may appear.
function styleWithoutTokens() {
  return style.replace(/:root(?:\[[^\]]*\])?\s*\{[^}]*\}/g, "");
}

function tokenBlock(selector, source = style) {
  const at = source.indexOf(selector);
  assert.ok(at >= 0, "missing token block: " + selector);
  const open = source.indexOf("{", at);
  const close = source.indexOf("}", open);
  return source.slice(open + 1, close);
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
    // The light panels were deliberately brightened so the grey no longer
    // reads yellow; they still stop short of paper white (1.0).
    assert.ok(panel <= 0.95, name + " panel luminance " + panel.toFixed(3) + " is too bright");
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

function desktopMediaQueries() {
  const blocks = [];
  for (const match of style.matchAll(/@media \(min-width:701px\)[^{]*\{/g)) {
    const open = match.index + match[0].length;
    let depth = 1, end = open;
    while (end < style.length && depth) {
      if (style[end] === "{") depth++;
      if (style[end] === "}") depth--;
      end++;
    }
    assert.equal(depth, 0, "unclosed desktop media query");
    blocks.push({ css: style.slice(match.index, end), body: style.slice(open, end - 1) });
  }
  return blocks;
}

function tokenValues(block) {
  return Object.fromEntries([...block.matchAll(/(--[a-z0-9-]+):([^;]+);/g)]
    .map((match) => [match[1], match[2].trim()]));
}

function ruleValues(source, selector) {
  const values = {};
  for (const match of source.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const selectors = match[1].replace(/\/\*[\s\S]*?\*\//g, "").trim().split(",").map((value) => value.trim());
    if (!selectors.includes(selector)) continue;
    for (const entry of match[2].split(";")) {
      const colon = entry.indexOf(":");
      if (colon >= 0) values[entry.slice(0, colon).trim()] = entry.slice(colon + 1).trim();
    }
  }
  return values;
}

function brightnessColor(hex, factor) {
  return "#" + hex.slice(1).match(/../g).map((channel) => Math.round(parseInt(channel, 16) * factor)
    .toString(16).padStart(2, "0")).join("");
}

test("desktop palette overrides use neutral surfaces and keep the shared accent", () => {
  const expected = {
    "dark": {
      "--bg": "#1b1b1b",
      "--card-edge": "rgba(231,231,231,.12)",
      "--card-shadow": "0 2px 8px rgba(0,0,0,.12)",
      "--code-bg": "#1f1f1f",
      "--code-toolbar": "#292929",
      "--control-border": "#858585",
      "--dim": "#c4c4c4",
      "--edge": "rgba(231,231,231,.20)",
      "--fg": "#e9e9e9",
      "--fg-bold": "#ffffff",
      "--field": "#292929",
      "--fw-bold": "800",
      "--glass": "rgba(36,36,36,.95)",
      "--glass-strong": "rgba(36,36,36,.98)",
      "--line": "#3e3e3e",
      "--line-2": "#5c5c5c",
      "--media-bg": "#161616",
      "--panel": "#242424",
      "--surface": "#2e2e2e",
      "--surface-2": "#292929",
      "--surface-hover": "#383838",
      "--tint": "rgba(231,231,231,.08)",
      "--tint-2": "rgba(231,231,231,.05)",
      "--tint-3": "rgba(231,231,231,.10)",
      "--tint-4": "rgba(231,231,231,.11)"
    },
    "light": {
      "--bg": "#f1f1f1",
      "--code-bg": "#e8e8e8",
      "--code-toolbar": "#efefef",
      "--control-border": "#737373",
      "--card-edge": "rgba(46,46,46,.12)",
      "--card-shadow": "0 2px 4px rgba(46,46,46,.03), 0 10px 28px rgba(46,46,46,.05)",
      "--danger-text": "#9f3029",
      "--dim": "#3e3e3e",
      "--edge": "rgba(46,46,46,.18)",
      "--fg": "#242424",
      "--field": "#f3f3f3",
      "--fw-meta": "450",
      "--glass": "rgba(243,243,243,.95)",
      "--glass-strong": "rgba(243,243,243,.98)",
      "--line": "#9a9a9a",
      "--line-2": "#858585",
      "--media-bg": "#e0e0e0",
      "--panel": "#f3f3f3",
      "--shadow-panel": "0 8px 28px rgba(46,46,46,.10)",
      "--shadow-sm": "0 2px 8px rgba(46,46,46,.05)",
      "--surface": "#e5e5e5",
      "--surface-2": "#efefef",
      "--surface-hover": "#d9d9d9",
      "--tint": "rgba(46,46,46,.06)",
      "--tint-2": "rgba(46,46,46,.04)",
      "--tint-3": "rgba(46,46,46,.14)",
      "--tint-4": "rgba(46,46,46,.13)"
    },
    "warm": {
      "--bg": "#edeae4",
      "--card-edge": "rgba(72,66,52,.12)",
      "--card-shadow": "0 2px 8px rgba(64,60,46,.06)",
      "--code-bg": "#e9e6df",
      "--code-toolbar": "#f0ede7",
      "--control-border": "#6f6a5b",
      "--dim": "#5b5d51",
      "--edge": "rgba(52,54,45,.18)",
      "--fg": "#34362b",
      "--field": "#f0ede7",
      "--glass": "rgba(245,242,236,.95)",
      "--glass-strong": "rgba(245,242,236,.98)",
      "--line": "#d9d2c2",
      "--line-2": "#c8c0ae",
      "--media-bg": "#e6e3db",
      "--panel": "#f5f2ec",
      "--shadow-panel": "0 8px 28px rgba(64,60,46,.10)",
      "--shadow-sm": "0 2px 8px rgba(64,60,46,.05)",
      "--surface": "#e8e5de",
      "--surface-2": "#f0ede7",
      "--surface-hover": "#e2dfd7",
      "--tint": "rgba(53,105,93,.10)",
      "--tint-2": "rgba(53,105,93,.06)",
      "--tint-3": "rgba(53,105,93,.13)",
      "--tint-4": "rgba(53,105,93,.18)"
    }
  };
  const desktop = desktopMediaQueries()[0].body;
  for (const [name, selector] of Object.entries(PALETTES)) {
    const shared = tokenValues(tokenBlock(selector));
    const overrides = tokenValues(tokenBlock(selector, desktop));
    assert.deepEqual(overrides, expected[name], name + " desktop palette changed");
    const effective = { ...shared, ...overrides };
    for (const token of ["--acc", "--acc-soft", "--acc-soft-2", "--acc-line", "--acc-line-soft",
      "--acc-solid", "--acc-focus", "--on-acc", "--ok", "--err", "--mark", "--focus-ring", "--glow"]) {
      assert.equal(effective[token], shared[token], name + " changed the accent token " + token);
    }
  }
});

test("desktop light text and control boundaries retain contrast on a brighter neutral canvas", () => {
  const desktop = desktopMediaQueries()[0].body;
  const palette = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktop)) };
  for (const token of ["--bg", "--panel", "--surface", "--surface-hover", "--surface-2", "--field", "--code-bg", "--code-toolbar", "--media-bg"]) {
    assert.ok(contrast(palette["--dim"], palette[token]) >= 7, "muted text on " + token);
    assert.ok(contrast(palette["--fg"], palette[token]) >= 7, "body text on " + token);
  }
  assert.equal(palette["--bg"], "#f1f1f1");
  assert.equal(palette["--panel"], "#f3f3f3");
  assert.ok(relativeLuminance(palette["--bg"]) > relativeLuminance("#e9e9e9"));
  assert.ok(relativeLuminance(palette["--panel"]) < 0.9, "reading surface stays below paper white");
  assert.ok(contrast(palette["--line-2"], palette["--bg"]) >= 3, "folded tool boundary");
  for (const token of ["--bg", "--panel", "--field", "--surface-hover"]) {
    assert.ok(contrast(palette["--control-border"], palette[token]) >= 3, "control boundary on " + token);
  }
  const rgba = palette["--tint-3"].match(/^rgba\((\d+),(\d+),(\d+),([.\d]+)\)$/);
  assert.ok(rgba);
  const selected = "#" + palette["--panel"].slice(1).match(/../g).map((channel, index) =>
    Math.round(parseInt(channel, 16) * (1 - Number(rgba[4])) + Number(rgba[index + 1]) * Number(rgba[4]))
      .toString(16).padStart(2, "0")).join("");
  assert.ok(contrast(selected, palette["--panel"]) >= 1.2, "selected row differs from its panel");
  assert.ok(contrast(palette["--dim"], selected) >= 7, "selected row metadata remains readable");
  assert.ok(contrast(palette["--acc"], palette["--field"]) >= 4.5, "side-note text and boundary");
});

test("desktop light chrome keeps metadata weight and retains hover and focus cues", () => {
  const desktop = desktopMediaQueries()[0].body;
  const light = ':root[data-theme="light"] ';
  assert.deepEqual(ruleValues(desktop, light + "body"), {
    "--radius-card": "16px", "--radius-control": "10px"
  });
  assert.equal(tokenValues(tokenBlock(PALETTES.light, desktop))["--fw-meta"], "450");
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  for (const selector of ["#info", "#sessmeta", "#sessions li small", ".msg details", ".msg .prog"]) {
    assert.equal(ruleValues(shared, selector)["font-weight"], "var(--fw-meta)", selector + " uses the light metadata weight override");
  }
  for (const selector of ["input::placeholder", "textarea::placeholder"]) {
    assert.deepEqual(ruleValues(desktop, light + selector), { color: "var(--dim)", opacity: "1" });
  }
  for (const selector of ["#ovr select", "#omodelbtn", "#oagentbtn", "#oeffortbtn", "#oskill"]) {
    assert.deepEqual(ruleValues(desktop, light + selector), {
      "background-color": "var(--field)", "border-color": "var(--control-border)", "font-weight": "450"
    });
    const hover = selector + (selector === "#ovr select" ? ":hover" : ":hover:not(:disabled)");
    assert.equal(ruleValues(desktop, light + hover)["background-color"], "var(--surface-hover)");
    assert.equal(ruleValues(desktop, light + selector + ":focus-visible")["border-color"], "var(--acc-focus)");
  }
  assert.deepEqual(ruleValues(desktop, light + "#sidenote.show"), {
    background: "var(--field)", "border-color": "var(--acc)", color: "var(--acc)"
  });
  for (const selector of [
    '.msg.asst.tool-activity:not(.git-output):not(:has(> .task-output)):not([data-running="true"])',
    '.msg.asst:has(> details[data-fold-key^="thinking:"])',
    '.msg.asst:has(> details[data-fold-key$=":thinking"])'
  ]) {
    assert.deepEqual(ruleValues(desktop, light + selector), { "border-color": "var(--card-edge)", "box-shadow": "none" });
  }
});

test("desktop dark inline emphasis stands out while prose and headings keep their shared weights", () => {
  const desktop = desktopMediaQueries()[0].body;
  const dark = ':root:not([data-theme="light"]):not([data-theme="warm"]) ';
  const palette = { ...tokenValues(tokenBlock(PALETTES.dark)), ...tokenValues(tokenBlock(PALETTES.dark, desktop)) };
  for (const tag of ["strong", "b"]) {
    assert.deepEqual(ruleValues(desktop, dark + ".msg.asst.md " + tag + ":not(.h *)"), {
      "font-weight": "var(--fw-bold)", color: "var(--fg-bold)"
    });
  }
  assert.ok(Number(palette["--fw-bold"]) > 700, "dark emphasis is heavier than the browser's usual bold weight");
  assert.ok(
    contrast(palette["--fg-bold"], palette["--panel"]) > contrast(palette["--fg"], palette["--panel"]),
    "dark emphasis has more contrast than surrounding prose"
  );
  for (const selector of ["body", ".msg.md .h"]) {
    assert.deepEqual(
      ruleValues(style, dark + selector),
      {},
      "dark body copy and headings keep their shared weights"
    );
  }
  assert.equal(ruleValues(desktop, "#composer")["padding"], "8px 24px 20px");
});

test("eye-care desktop copy keeps the shared prose weight", () => {
  const desktop = desktopMediaQueries()[0].body, warm = ':root[data-theme="warm"] ';
  assert.deepEqual(ruleValues(desktop, warm + ".msg.md"), {}, "warm body copy keeps the shared weight");
  assert.deepEqual(
    ruleValues(desktop, ':root[data-theme="light"] body'),
    { "--radius-card": "16px", "--radius-control": "10px" },
    "light body only carries the card shape"
  );
});

test("desktop sidebar and cards share soft shapes across all three palettes", () => {
  const desktop = desktopMediaQueries()[0].body;
  for (const selector of Object.values(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    assert.match(palette["--card-edge"], /^rgba\([\d,]+,\.12\)$/);
    assert.ok(palette["--card-shadow"] && palette["--card-shadow"] !== "none");
  }
  assert.equal(ruleValues(desktop, '.msg.asst:not(.git-output)')["border-color"], "var(--card-edge)");
  assert.equal(ruleValues(desktop, '.msg.asst:not(.git-output)')["border-radius"], "var(--radius-card)");
  assert.equal(ruleValues(desktop, '.msg.asst.tool-activity[data-running="true"]')["border-color"], "var(--acc-line)");
  assert.equal(ruleValues(desktop, '#sessfilter')["border-radius"], "12px");
  assert.equal(ruleValues(desktop, '#sessions li')["border-radius"], "12px");
  assert.equal(ruleValues(desktop, '#sessions li.cur')["box-shadow"], "inset 3px 0 0 var(--acc)");
  const actions = ruleValues(desktop, '#sessions .rowbtns');
  assert.equal(actions.display, "flex");
  assert.equal(actions.gap, "6px");
  const button = ruleValues(desktop, '#sessions .rowbtns button');
  assert.equal(button.flex, "1");
  assert.equal(button["min-width"], "0");
  assert.equal(button["min-height"], "34px");
  assert.equal(button.background, "var(--field)");
  assert.equal(button["border-radius"], "10px");
  assert.equal(ruleValues(desktop, '#sessions .rowbtns button:hover:not(:disabled)').background, "var(--surface-hover)");
  assert.equal(ruleValues(desktop, '#sessions .rowbtns button.danger').color, "var(--danger-text, var(--err))");
  assert.equal(ruleValues(desktop, '#sessions button:focus-visible').outline, "2px solid var(--acc-focus)");
  assert.equal(ruleValues(desktop, '#sessions span[role="button"]:focus-visible').outline, "2px solid var(--acc-focus)");
  assert.equal(ruleValues(desktop, '#promptnav')["border-radius"], "12px");
  assert.equal(ruleValues(desktop, '#skillhint').display, "none");
});

test("desktop light surfaces soften without losing activity and keyboard focus cues", () => {
  const desktop = desktopMediaQueries()[0].body, light = ':root[data-theme="light"] ';
  assert.deepEqual(ruleValues(desktop, light + '.msg.asst:not(.git-output)'), {
    "border-color": "var(--card-edge)", "border-radius": "var(--radius-card)", "box-shadow": "var(--card-shadow)"
  });
  assert.deepEqual(ruleValues(desktop, light + '.msg.asst.tool-activity[data-running="true"]'), {
    "border-color": "var(--acc-line)"
  });
  assert.deepEqual(ruleValues(desktop, light + '#sessions'), { "border-right-color": "var(--card-edge)" });
  for (const selector of ["#promptnav", "#ovrwrap"]) {
    assert.deepEqual(ruleValues(desktop, light + selector), { "border-top-color": "var(--card-edge)" });
  }
  assert.deepEqual(ruleValues(desktop, light + '#prompt'), {
    background: "var(--field)", "border-radius": "12px", "box-shadow": "var(--shadow-sm)"
  });
  assert.deepEqual(ruleValues(desktop, light + '.execution-group > details > summary:focus-visible'), {
    outline: "2px solid var(--acc-focus)", "outline-offset": "2px"
  });
  const palette = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktop)) };
  const rgba = palette["--card-edge"].match(/^rgba\((\d+),(\d+),(\d+),([.\d]+)\)$/);
  const edge = "#" + palette["--panel"].slice(1).match(/../g).map((channel, index) =>
    Math.round(parseInt(channel, 16) * (1 - Number(rgba[4])) + Number(rgba[index + 1]) * Number(rgba[4]))
      .toString(16).padStart(2, "0")).join("");
  assert.ok(contrast(edge, palette["--panel"]) < contrast(palette["--line"], palette["--panel"]), "passive card edges recede");
});

test("desktop themes share a lightly inset content width and light top separators stay subtle", () => {
  const blocks = desktopMediaQueries();
  const desktop = blocks[0].body, narrow = blocks[1].body;
  const light = ':root[data-theme="light"] ';
  const palette = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktop)) };
  for (const selector of ["header", "#sessbar"]) {
    const rule = ruleValues(desktop, light + selector);
    assert.deepEqual(rule, { "border-bottom-color": "#d8d8d8" });
    assert.ok(contrast(rule["border-bottom-color"], palette["--panel"]) < contrast(palette["--line"], palette["--panel"]), "quiet top separator");
  }
  for (const selector of ["#msgwrap", "#ovrwrap", "#sidequeue", "#composer"]) {
    for (const [theme, root] of Object.entries(PALETTES)) {
      const expected = theme === "light" && selector === "#ovrwrap" ? { "border-top-color": "var(--card-edge)" } : {};
      const scope = root.replace(/\s*\{$/, "");
      assert.deepEqual(ruleValues(desktop, scope + " " + selector), expected, theme + " inherits the shared content width");
    }
    const outer = ruleValues(desktop, selector);
    assert.equal(outer.width, "96%", selector + " retains a small percentage gutter inside the chat column");
    assert.equal(outer["align-self"], "center", selector + " retains shared alignment");
    assert.equal(outer["max-width"], "1280px", "all desktop themes share the wider capped content area");
    assert.equal(ruleValues(narrow, selector).width, undefined, "narrow desktop must not replace the percentage width");
    assert.equal(ruleValues(narrow, selector)["min-width"], undefined, "no fixed minimum can force horizontal overflow");
  }
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  assert.equal(ruleValues(shared, "*")["box-sizing"], "border-box");
  assert.equal(ruleValues(shared, "#chat")["min-width"], "0");
  for (const edge of ["top", "right", "bottom", "left"]) {
    assert.equal(ruleValues(shared, "#msgs")[edge], "0", "transcript fills its wrapper on " + edge);
  }
  assert.equal(
    ruleValues(desktop, "#msgs").padding,
    "18px 40px 56px",
    "the left reading gutter stays put; only the right one widens to clear the scrollbar"
  );
  assert.equal(ruleValues(narrow, "#ovrwrap")["grid-template-columns"], "repeat(3,minmax(0,1fr))", "narrow desktop keeps shrinkable picker columns");
});

test("desktop dark and light stay neutral while eye-care keeps a bounded warm band", () => {
  const desktop = desktopMediaQueries()[0].body;
  const tokens = ["--bg", "--panel", "--surface", "--surface-hover", "--surface-2", "--field",
    "--code-toolbar", "--code-bg", "--media-bg", "--glass", "--glass-strong", "--tint", "--tint-2",
    "--tint-3", "--tint-4", "--edge", "--line", "--line-2", "--control-border", "--fg", "--dim", "--shadow-sm", "--shadow-panel"];
  for (const [name, selector] of Object.entries(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    for (const token of tokens) {
      const color = palette[token].match(/#([a-f0-9]{6})(?![a-f0-9])|rgba\((\d+),(\d+),(\d+),[.\d]+\)/i);
      assert.ok(color, name + " " + token + " has no parsable color");
      const [r, g, b] = color[1] ? color[1].match(/../g).map((channel) => parseInt(channel, 16)) : color.slice(2).map(Number);
      const high = Math.max(r, g, b), low = Math.min(r, g, b), chroma = high - low;
      if (chroma === 0) continue;
      const sector = high === r ? ((g - b) / chroma + 6) % 6 : high === g ? (b - r) / chroma + 2 : (r - g) / chroma + 4;
      const hue = sector * 60;
      if (name === "warm" && token.startsWith("--tint")) {
        // Selection and hover washes stay on the teal accent, not on a second hue.
        assert.ok(hue > 140 && hue < 210, name + " " + token + " left the accent hue at " + hue);
        continue;
      }
      if (name === "warm") {
        // Eye-care is warm on purpose, but only inside a paper band: no orange
        // and no saturated cream that would compete with the accent.
        assert.ok(chroma <= 26, name + " " + token + " channel spread " + chroma + " is too saturated for paper");
        assert.ok(hue > 25 && hue < 75, name + " " + token + " has a hue " + hue + " outside the warm paper band");
        continue;
      }
      assert.ok(chroma <= 6, name + " " + token + " channel spread " + chroma + " is not neutral");
      assert.ok(hue < 20 || hue > 80, name + " " + token + " has a yellow/orange hue " + hue);
      assert.ok(hue < 140 || hue > 210, name + " " + token + " has a teal hue " + hue);
    }
  }
});

test("desktop eye-care surfaces keep only a subtle warm paper tint", () => {
  const desktop = desktopMediaQueries()[0].body;
  const palette = tokenValues(tokenBlock(PALETTES.warm, desktop));
  const surfaces = ["--bg", "--panel", "--surface", "--surface-hover", "--surface-2", "--field",
    "--code-toolbar", "--code-bg", "--media-bg", "--glass", "--glass-strong"];
  for (const token of surfaces) {
    const color = palette[token].match(/#([a-f0-9]{6})(?![a-f0-9])|rgba\((\d+),(\d+),(\d+),[.\d]+\)/i);
    assert.ok(color, token + " has no parsable paper color");
    const [r, g, b] = color[1] ? color[1].match(/../g).map((channel) => parseInt(channel, 16)) : color.slice(2).map(Number);
    assert.ok(r > g && g > b, token + " lost its gentle warm tint");
    assert.ok(r - b <= 11, token + " has too much yellow: channel spread " + (r - b));
  }
});

test("desktop text and selected rows stay readable on neutral surfaces", () => {
  const desktop = desktopMediaQueries()[0].body;
  for (const [name, selector] of Object.entries(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    const surfaces = ["--bg", "--panel", "--surface", "--surface-hover", "--surface-2", "--code-bg", "--media-bg"];
    for (const token of surfaces) {
      const background = palette[token];
      assert.ok(contrast(palette["--fg"], background) >= 7, name + " body text contrast on " + token);
      assert.ok(contrast(palette["--dim"], background) >= 4.5, name + " muted text contrast on " + token);
    }
    for (const token of ["--tint", "--tint-2", "--tint-3", "--tint-4"]) {
      const rgba = palette[token].match(/^rgba\((\d+),(\d+),(\d+),([.\d]+)\)$/);
      assert.ok(rgba, name + " " + token + " is not an rgba fill");
      const base = palette["--panel"].slice(1).match(/../g).map((channel) => parseInt(channel, 16));
      const alpha = Number(rgba[4]);
      const background = "#" + base.map((channel, index) => Math.round(channel * (1 - alpha) + Number(rgba[index + 1]) * alpha)
        .toString(16).padStart(2, "0")).join("");
      assert.ok(contrast(palette["--fg"], background) >= 7, name + " selected text contrast on " + token);
      assert.ok(contrast(palette["--dim"], background) >= 4.5, name + " selected muted text contrast on " + token);
    }
    if (name !== "dark") {
      assert.ok(relativeLuminance(palette["--panel"]) <= 0.9, name + " desktop panel is too bright");
    }
    assert.ok(relativeLuminance(palette["--panel"]) > relativeLuminance(palette["--bg"]), name + " panel hierarchy");
    assert.ok(contrast(palette["--acc"], palette["--panel"]) >= 4.5, name + " link contrast");
    assert.ok(contrast(palette["--on-acc"], palette["--acc"]) >= 4.5, name + " primary button contrast");
  }
});

test("desktop rewind confirmation text stays readable through hover and pressed filters", () => {
  const desktop = desktopMediaQueries().map((block) => block.body).join("\n");
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  const ok = ruleValues(desktop, ".rewind-confirm-ok");
  assert.equal(ok.color, "var(--on-acc)");
  assert.equal(ok.background, "var(--acc)");
  assert.equal(ok["border-color"], "var(--acc)");
  assert.deepEqual(ruleValues(desktop, ".rewind-confirm-ok:hover"), { filter: "brightness(.94)", color: "var(--fg)" });
  assert.equal(ruleValues(shared, "button:hover:not(:disabled):not(.primary):not(.stop)")["background-color"], "var(--surface-hover)");
  assert.equal(ruleValues(shared, "button:active:not(:disabled)").filter, "brightness(.96)");
  for (const [name, selector] of Object.entries(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    for (const factor of [1, .96]) {
      assert.ok(contrast(brightnessColor(palette["--on-acc"], factor), brightnessColor(palette["--acc"], factor)) >= 4.5,
        name + " rewind confirmation contrast at brightness " + factor);
    }
    for (const factor of [.94, .96]) {
      assert.ok(contrast(brightnessColor(palette["--fg"], factor), brightnessColor(palette["--surface-hover"], factor)) >= 4.5,
        name + " rewind confirmation hover contrast at brightness " + factor);
    }
  }
});

test("desktop rewind confirmation retains accent text without hover capability", () => {
  function withoutHoverCapability(source) {
    const ranges = [];
    for (const match of source.matchAll(/@media \(hover:hover\)\s*\{/g)) {
      let end = match.index + match[0].length, depth = 1;
      while (end < source.length && depth) {
        if (source[end] === "{") depth++;
        if (source[end] === "}") depth--;
        end++;
      }
      assert.equal(depth, 0, "unclosed hover capability media query");
      ranges.push([match.index, end]);
    }
    for (const [start, end] of ranges.reverse()) source = source.slice(0, start) + source.slice(end);
    return source;
  }
  const desktop = withoutHoverCapability(desktopMediaQueries().map((block) => block.body).join("\n"));
  const shared = withoutHoverCapability(style.slice(0, style.indexOf("@media (min-width:701px)")));
  const idle = ruleValues(desktop, ".rewind-confirm-ok"), hover = ruleValues(desktop, ".rewind-confirm-ok:hover");
  assert.deepEqual(hover, { filter: "brightness(.94)" });
  const effective = { ...idle, ...hover };
  assert.equal(effective.color, "var(--on-acc)");
  assert.equal(effective.background, "var(--acc)");
  assert.equal(ruleValues(shared, "button:hover:not(:disabled):not(.primary):not(.stop)")["background-color"], undefined);
  for (const [name, selector] of Object.entries(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    for (const factor of [.94, .96]) {
      assert.ok(contrast(brightnessColor(palette["--on-acc"], factor), brightnessColor(palette["--acc"], factor)) >= 4.5,
        name + " rewind confirmation without hover capability at brightness " + factor);
    }
  }
});

test("desktop danger text is light-only and readable on idle, hover and pressed fills", () => {
  const desktop = desktopMediaQueries()[0].body;
  const palette = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktop)) };
  for (const selector of ["#sessionactiondelete", "#sessdelconfirm", "#del", '#teamtrigger[data-state="error"] #teamstatus']) {
    assert.deepEqual(ruleValues(desktop, ':root[data-theme="light"] ' + selector), { color: "var(--danger-text)" });
  }
  assert.equal(palette["--err"], "#b53e36");
  assert.equal(ruleValues(style, "#sessdelconfirm")["border-color"], "var(--err)");
  for (const name of ["dark", "warm"]) assert.equal(tokenValue(tokenBlock(PALETTES[name], desktop), "--danger-text"), null);
  for (const token of ["--panel", "--surface", "--surface-hover", "--surface-2"]) {
    for (const factor of [1, .96]) {
      assert.ok(contrast(brightnessColor(palette["--danger-text"], factor), brightnessColor(palette[token], factor)) >= 4.5,
        "light danger text contrast on " + token + " at brightness " + factor);
    }
  }
});

test("desktop default control borders meet 3:1 and preserve selected accents and focus outlines", () => {
  const desktop = desktopMediaQueries()[0].body;
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  for (const selector of ["#prompt", "#sessfilter", ".modelrowcheck"]) {
    assert.equal(ruleValues(desktop, selector)["border-color"], "var(--control-border)");
    assert.equal(ruleValues(desktop, selector).outline, undefined);
  }
  assert.equal(ruleValues(shared, "input").border, "1px solid var(--line)");
  assert.equal(ruleValues(shared, ".modelrowcheck").border, "1.5px solid var(--line-2)");
  for (const selector of ['.modelrow[aria-selected="true"] .modelrowcheck', '.teamchoice[aria-checked="true"] .modelrowcheck']) {
    assert.deepEqual(ruleValues(shared, selector), { background: "var(--acc)", "border-color": "var(--acc)" });
    assert.deepEqual(ruleValues(desktop, selector), {});
  }
  for (const selector of ["input:focus-visible", "textarea:focus-visible", "#modelpicklist .modelrow:focus-visible"]) {
    assert.equal(ruleValues(shared, selector).outline, "2px solid var(--acc-focus)");
  }
  for (const [name, selector] of Object.entries(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    for (const token of ["--bg", "--panel", "--field", "--surface", "--surface-hover", "--surface-2"]) {
      for (const factor of [1, .96]) {
        assert.ok(contrast(brightnessColor(palette["--control-border"], factor), brightnessColor(palette[token], factor)) >= 3,
          name + " control border contrast on " + token + " at brightness " + factor);
      }
    }
  }
});

test("desktop rewind opacity and light metadata retain readable foregrounds", () => {
  const desktop = desktopMediaQueries()[0].body;
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  assert.deepEqual(ruleValues(desktop, ".rewind-action"), { opacity: "1" });
  assert.equal(ruleValues(shared, ".rewind-action").opacity, ".62");
  for (const [name, selector] of Object.entries(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    const rgba = palette["--glass"].match(/^rgba\((\d+),(\d+),(\d+),([.\d]+)\)$/);
    const base = palette["--bg"].slice(1).match(/../g).map((channel) => parseInt(channel, 16));
    const alpha = Number(rgba[4]);
    const background = "#" + base.map((channel, index) => Math.round(channel * (1 - alpha) + Number(rgba[index + 1]) * alpha)
      .toString(16).padStart(2, "0")).join("");
    assert.ok(contrast(palette["--dim"], background) >= 4.5, name + " opaque rewind action contrast");
    if (name === "light") {
      assert.equal(palette["--dim"], "#3e3e3e");
      for (const token of ["--panel", "--surface-hover"]) {
        assert.ok(contrast(palette["--dim"], palette[token]) > contrast("#575757", palette[token]), "light metadata contrast on " + token);
      }
    }
  }
});

// Pin shared and phone CSS including the command menu styles from 4722cb07.
test("desktop contrast rules preserve the current shared and mobile CSS bytes", () => {
  const phoneApplicable = desktopMediaQueries().reduce((source, block) => source.replace(block.css, ""), style);
  assert.doesNotMatch(phoneApplicable, /--(?:control-border|danger-text)/);
  assert.equal(createHash("sha256").update(phoneApplicable).digest("hex"),
    "e9ddcc9675b0887b8946cfd7976bd1204e7cdce9ab8026d115e733e34864aca8",
    "desktop contrast changed shared or mobile CSS");
});

test("desktop overrides leave all phone-applicable CSS byte-identical", () => {
  const desktop = desktopMediaQueries();
  assert.equal(desktop.length, 3, "expected the desktop layout, narrow desktop layout, and width tuning media queries");
  let phoneApplicable = style;
  for (const block of desktop) phoneApplicable = phoneApplicable.replace(block.css, "");
  assert.equal(createHash("sha256").update(phoneApplicable).digest("hex"),
    "e9ddcc9675b0887b8946cfd7976bd1204e7cdce9ab8026d115e733e34864aca8",
    "shared or phone CSS changed outside the desktop queries");
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
