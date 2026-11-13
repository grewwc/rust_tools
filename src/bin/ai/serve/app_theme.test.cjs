// Run with: node --test src/bin/ai/serve/app_theme.test.cjs
// Theme contract of the embedded client: three palettes (dark, light, warm)
// driven by a stored preference that may also be "auto", a header button that
// cycles the palettes, a picker in the settings panel, and a choice that
// survives a reload. Desktop light uses white surfaces, one green accent and
// purple side-note borders; warm keeps its eye-care surface scale.
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
    const header = match[1].replace(/\/\*[\s\S]*?\*\//g, "").trim();
    const selectors = [];
    let start = 0, depth = 0, quote = "";
    for (let i = 0; i < header.length; i++) {
      const char = header[i];
      if (quote) { if (char === quote && header[i - 1] !== "\\") quote = ""; }
      else if (char === "'" || char === '"') quote = char;
      else if (char === "(" || char === "[") depth++;
      else if (char === ")" || char === "]") depth--;
      else if (char === "," && depth === 0) { selectors.push(header.slice(start, i).trim()); start = i + 1; }
    }
    selectors.push(header.slice(start).trim());
    if (!selectors.includes(selector)) continue;
    for (const entry of match[2].split(";")) {
      const colon = entry.indexOf(":");
      if (colon >= 0) values[entry.slice(0, colon).trim()] = entry.slice(colon + 1).trim();
    }
  }
  return values;
}

test("mobile warm user timestamps meet 4.5:1 and stay softer than body text without changing bubble or footer styles", () => {
  const media = "@media (max-width:700px) {";
  const start = style.indexOf(media);
  assert.ok(start >= 0, "missing phone media query");
  const open = start + media.length;
  let depth = 1, end = open;
  while (end < style.length && depth) {
    if (style[end] === "{") depth++;
    if (style[end] === "}") depth--;
    end++;
  }
  assert.equal(depth, 0, "unclosed phone media query");
  const phone = style.slice(open, end - 1);
  const outsidePhone = style.slice(0, start) + style.slice(end);
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  const selector = ':root[data-theme="warm"] .msg.user > .message-actions > .message-time';
  const palette = { ...tokenValues(tokenBlock(PALETTES.warm)),
    ...ruleValues(phone, ':root[data-theme="warm"]') };
  const bubble = ruleValues(shared, ".msg.user");
  const timestamp = { ...ruleValues(shared, ".message-time"),
    ...ruleValues(phone, ".message-time"), ...ruleValues(phone, selector) };
  const resolve = (value) => palette[/^var\((--[\w-]+)\)$/.exec(value)?.[1]];
  const ratio = contrast(resolve(timestamp.color), resolve(bubble.background));
  assert.ok(ratio >= 4.5, "mobile warm user timestamp contrast " + ratio.toFixed(2) + " is too low");
  const bodyRatio = contrast(resolve(bubble.color), resolve(bubble.background));
  assert.ok(ratio <= bodyRatio * 0.9, "timestamp contrast must be at least 10% softer than body text");
  assert.equal(palette["--on-user-dim"], "#e5e1d7", "phone warm timestamp keeps its dedicated secondary color");
  assert.equal(resolve(bubble.color), "#fffaf1", "user body foreground stays unchanged");
  assert.equal(resolve(bubble.background), "#35695d", "user bubble background stays unchanged");
  assert.equal(bubble.background, "var(--acc-solid)");
  assert.equal(bubble.color, "var(--on-user)", "user body text keeps its original color");
  assert.deepEqual(ruleValues(phone, selector), { color: "var(--on-user-dim)" },
    "only the user's footer timestamp color is overridden");
  assert.deepEqual(ruleValues(outsidePhone, selector), {}, "desktop keeps its existing timestamp color");
  assert.doesNotMatch(outsidePhone, /--on-user-dim/, "the secondary user foreground is scoped to phones");
  assert.equal(ruleValues(shared, ".message-time").color, "var(--dim)",
    "assistant and queued-note timestamps keep the shared secondary color");
  assert.deepEqual(ruleValues(phone, ".message-time"), { "font-size": "12px", "scroll-margin-block": "8px" });
  for (const theme of ["dark", "light"]) {
    assert.deepEqual(ruleValues(phone, selector.replace('"warm"', '"' + theme + '"')), {},
      theme + " keeps its existing timestamp color");
  }
});

test("desktop Markdown answers inherit panel backgrounds and retain stronger borders in all three themes", () => {
  const desktop = desktopMediaQueries().map((block) => block.body).join("\n");
  const selector = ".msg.asst.md:not(.pending):not(.err):not(.git-output)";
  const body = ruleValues(desktop, selector);
  assert.equal(body.background, undefined);
  assert.equal(body.border, "2px solid var(--control-border)");
  const expected = {
    dark: { page: "#1b1b1b", surface: "#2e2e2e", answer: "#242424", border: "#858585" },
    light: { page: "#f1f1f1", surface: "#e5e5e5", answer: "#f3f3f3", border: "#737373" },
    warm: { page: "#e9e8e7", surface: "#e0dfde", answer: "#efeeed", border: "#6f6a5b" },
  };
  for (const [name, root] of Object.entries(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(PALETTES.dark)),
      ...tokenValues(tokenBlock(root)), ...tokenValues(tokenBlock(PALETTES.dark, desktop)),
      ...tokenValues(tokenBlock(root, desktop)) };
    // The helper reads exact selectors; resolve the unchanged assistant base
    // background and verify that no answer-specific theme overrides remain.
    const themed = name === "dark" ? {} : ruleValues(desktop, root.replace(/\s*\{$/, "") + " " + selector);
    assert.deepEqual(themed, name === "light" ? { border: "1px solid var(--card-edge)" } : {});
    const background = ruleValues(style, ".msg.asst").background;
    assert.equal(background, "var(--panel)");
    const token = /^var\((--[\w-]+)\)$/.exec(background)?.[1];
    assert.ok(token, name + " answer uses a theme token");
    assert.deepEqual({ page: palette["--bg"], surface: palette["--surface"],
      answer: palette[token], border: palette["--control-border"] }, expected[name]);
    assert.equal(palette[token], palette["--panel"], name + " answer keeps its original background");
    assert.ok(contrast(palette["--control-border"], palette[token]) >= 3,
      name + " answer boundary contrast is at least 3:1");
  }
});

function brightnessColor(hex, factor) {
  return "#" + hex.slice(1).match(/../g).map((channel) => Math.round(parseInt(channel, 16) * factor)
    .toString(16).padStart(2, "0")).join("");
}

test("desktop palette overrides use low-saturation surfaces and preserve action accents", () => {
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
      "--panel": "#f3f3f3",
      "--line": "#9a9a9a",
      "--fg": "#242424",
      "--dim": "#3e3e3e",
      "--surface": "#e5e5e5",
      "--surface-hover": "#d9d9d9",
      "--surface-2": "#efefef",
      "--line-2": "#858585",
      "--control-border": "#737373",
      "--danger-text": "#9f3029",
      "--fw-meta": "450",
      "--field": "#f3f3f3",
      "--code-bg": "#e8e8e8",
      "--media-bg": "#e0e0e0",
      "--code-toolbar": "#efefef",
      "--glass": "rgba(243,243,243,.95)",
      "--glass-strong": "rgba(243,243,243,.98)",
      "--tint": "rgba(46,46,46,.06)",
      "--tint-2": "rgba(46,46,46,.04)",
      "--tint-3": "rgba(46,46,46,.14)",
      "--tint-4": "rgba(46,46,46,.13)",
      "--edge": "rgba(46,46,46,.18)",
      "--shadow-sm": "0 2px 8px rgba(46,46,46,.05)",
      "--shadow-panel": "0 8px 28px rgba(46,46,46,.10)",
      "--card-edge": "rgba(46,46,46,.12)",
      "--card-shadow": "0 2px 4px rgba(46,46,46,.03), 0 10px 28px rgba(46,46,46,.05)"
},
    "warm": {
      "--bg": "#e9e8e7",
      "--card-edge": "rgba(72,66,52,.12)",
      "--card-shadow": "0 2px 8px rgba(64,60,46,.06)",
      "--code-bg": "#e2e1e0",
      "--code-toolbar": "#e7e6e5",
      "--control-border": "#6f6a5b",
      "--dim": "#5b5d51",
      "--edge": "rgba(52,54,45,.18)",
      "--fg": "#34362b",
      "--field": "#e7e6e5",
      "--glass": "rgba(239,238,237,.95)",
      "--glass-strong": "rgba(239,238,237,.98)",
      "--line": "#d9d2c2",
      "--line-2": "#c8c0ae",
      "--media-bg": "#dbdad9",
      "--panel": "#efeeed",
      "--shadow-panel": "0 8px 28px rgba(64,60,46,.10)",
      "--shadow-sm": "0 2px 8px rgba(64,60,46,.05)",
      "--surface": "#e0dfde",
      "--surface-2": "#e7e6e5",
      "--surface-hover": "#d7d6d5",
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

test("light surfaces keep their page, panel and readable field hierarchy at every width", () => {
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  const expected = {
    "--bg": "#f2f0ee", "--panel": "#faf9f7", "--line": "#e3dfdb",
    "--surface": "#f1eeeb", "--surface-2": "#eae6e2",
    "--field": "#fdfdfc", "--code-bg": "#f0ece8",
    "--code-toolbar": "#f5f2ef", "--glass": "rgba(250,249,247,.95)",
    "--glass-strong": "rgba(250,249,247,.98)",
    "--composer-a": "#faf9f7", "--composer-b": "#f2f0ee"
  };
  const base = tokenValues(tokenBlock(PALETTES.light));
  const desktop = { ...base, ...tokenValues(tokenBlock(PALETTES.light, desktopMediaQueries()[0].body)) };
  for (const [name, palette] of [["shared", base], ["desktop", desktop]]) {
    const scale = name === "shared" ? { ...expected, "--surface-hover": "#eae6e2", "--media-bg": "#eae6e2" }
      : { ...expected, "--bg": "#f1f1f1", "--panel": "#f3f3f3", "--line": "#9a9a9a",
        "--surface": "#e5e5e5", "--surface-2": "#efefef", "--field": "#f3f3f3",
        "--code-bg": "#e8e8e8", "--code-toolbar": "#efefef", "--glass": "rgba(243,243,243,.95)",
        "--glass-strong": "rgba(243,243,243,.98)", "--surface-hover": "#d9d9d9", "--media-bg": "#e0e0e0" };
    for (const [token, value] of Object.entries(scale)) {
      assert.equal(palette[token], value, name + " light " + token);
    }
    if (name === "desktop") {
      assert.equal(palette["--field"], palette["--panel"], "white fields and panels share the reading surface");
      assert.equal(palette["--surface-2"], "#efefef", "user cards restore the baseline neutral fill");
      const channels = palette["--surface-2"].slice(1).match(/../g).map((channel) => parseInt(channel, 16));
      const spread = Math.max(...channels) - Math.min(...channels);
      assert.equal(spread, 0, "user fill has no independent tint");
      assert.ok(contrast(palette["--fg"], palette["--surface-2"]) >= 4.5, "user text remains readable");
    } else {
      assert.ok(relativeLuminance(palette["--field"]) > relativeLuminance(palette["--panel"]));
      assert.ok(relativeLuminance(palette["--bg"]) > relativeLuminance(palette["--surface-2"]));
    }
    assert.ok(relativeLuminance(palette["--panel"]) > relativeLuminance(palette["--bg"]), name + " page/panel hierarchy");
    for (const surface of ["--bg", "--panel", "--surface", "--surface-hover", "--surface-2", "--field", "--code-bg", "--code-toolbar", "--media-bg"]) {
      for (const text of ["--fg", "--dim"]) {
        assert.ok(contrast(palette[text], palette[surface]) >= 4.5, name + " " + text + " on " + surface);
      }
    }
  }
  assert.deepEqual(ruleValues(shared, ':root[data-theme="light"] .msg.user'), {
    background: "var(--surface-2)", color: "var(--fg)"
  });
  assert.deepEqual(ruleValues(shared, ':root[data-theme="light"] #prompt'), {
    background: "var(--field)"
  });
});

test("desktop light surfaces have low saturation and a distinct canvas from eye-care", () => {
  const desktop = desktopMediaQueries()[0].body;
  const light = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktop)) };
  const warm = { ...tokenValues(tokenBlock(PALETTES.warm)), ...tokenValues(tokenBlock(PALETTES.warm, desktop)) };
  for (const token of ["--bg", "--panel", "--surface", "--surface-hover", "--surface-2", "--field",
    "--code-bg", "--code-toolbar", "--media-bg", "--glass", "--glass-strong", "--line", "--line-2",
    "--tint", "--tint-2", "--tint-3", "--tint-4"]) {
    const color = light[token].match(/#([a-f0-9]{6})(?![a-f0-9])|rgba\((\d+),(\d+),(\d+),[.\d]+\)/i);
    assert.ok(color, token + " has no parsable low-saturation color");
    const [r, g, b] = color[1] ? color[1].match(/../g).map((channel) => parseInt(channel, 16)) : color.slice(2).map(Number);
    const neutral = ["--panel", "--surface", "--surface-2", "--field", "--code-bg", "--code-toolbar",
      "--media-bg", "--glass", "--glass-strong", "--success-bg", "--pending-bg", "--error-bg"];
    const spread = Math.max(r, g, b) - Math.min(r, g, b);
    assert.ok(spread <= (neutral.includes(token) ? 0 : token === "--bg" ? 6 : 24), token + " exceeds its neutral or accent channel spread");
  }
  assert.equal(relativeLuminance(light["--bg"]), relativeLuminance("#f1f1f1"),
    "the desktop canvas stays near white without reaching paper white");
  assert.equal(relativeLuminance(light["--bg"]) - relativeLuminance(warm["--bg"]), relativeLuminance("#f1f1f1") - relativeLuminance("#e9e8e7"),
    "the light canvas must remain visibly brighter than eye-care");
  assert.equal(relativeLuminance(light["--field"]) - relativeLuminance(warm["--field"]), relativeLuminance("#f3f3f3") - relativeLuminance("#e7e6e5"),
    "light fields must remain distinct from eye-care fields");
  for (const selector of ["header", "#sessbar"]) {
    assert.equal(ruleValues(desktop, ':root[data-theme="light"] ' + selector)["border-bottom-color"], "#d8d8d8",
      selector + " separator uses a precise low-saturation mint tint");
  }
});

test("desktop light text and control boundaries retain contrast on a brighter pastel canvas", () => {
  const desktop = desktopMediaQueries()[0].body;
  const palette = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktop)) };
  for (const token of ["--bg", "--panel", "--surface", "--surface-hover", "--surface-2", "--field", "--code-bg", "--code-toolbar", "--media-bg"]) {
    assert.ok(contrast(palette["--dim"], palette[token]) >= 7, "muted text on " + token);
    assert.ok(contrast(palette["--fg"], palette[token]) >= 7, "body text on " + token);
  }
  assert.equal(palette["--bg"], "#f1f1f1");
  assert.equal(palette["--panel"], "#f3f3f3");
  assert.ok(relativeLuminance(palette["--bg"]) > relativeLuminance("#e9e9e9"));
  assert.equal(palette["--panel"], palette["--field"], "reading panel uses the existing white body surface");
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
    "--radius-card": "16px", "--radius-control": "10px", "font-weight": "400"
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
    assert.deepEqual(ruleValues(desktop, light + selector), {
      "border-color": "var(--card-edge)", "box-shadow": "none"
    });
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
  assert.equal(ruleValues(desktop, "#composer")["padding"], "10px 24px 8px");
});

test("eye-care desktop copy keeps the shared prose weight", () => {
  const desktop = desktopMediaQueries()[0].body, warm = ':root[data-theme="warm"] ';
  assert.deepEqual(ruleValues(desktop, warm + ".msg.md"), {}, "warm body copy keeps the shared weight");
  assert.deepEqual(
    ruleValues(desktop, ':root[data-theme="light"] body'),
    { "--radius-card": "16px", "--radius-control": "10px", "font-weight": "400" },
    "light body only carries the card shape"
  );
});

test("desktop sidebar and cards share soft shapes across all three palettes", () => {
  const desktop = desktopMediaQueries()[0].body;
  for (const selector of Object.values(PALETTES)) {
    const palette = { ...tokenValues(tokenBlock(selector)), ...tokenValues(tokenBlock(selector, desktop)) };
    const alpha = "12";
    assert.match(palette["--card-edge"], new RegExp("^rgba\\([\\d,]+,\\." + alpha + "\\)$"));
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
    "border-color": "var(--card-edge)", "border-radius": "var(--radius-card)",
    "box-shadow": "var(--card-shadow)"
  });
  assert.deepEqual(ruleValues(desktop, light + '.msg.asst.tool-activity[data-running="true"]'), {
    "border-color": "var(--acc-line)"
  });
  assert.deepEqual(ruleValues(desktop, light + '#sessions'), { "border-right-color": "var(--card-edge)" });
  for (const selector of ["#promptnav", "#ovrwrap"]) {
    assert.deepEqual(ruleValues(desktop, light + selector), selector === "#ovrwrap"
      ? { "border-top-color": "var(--card-edge)", order: "0", padding: "10px 24px 4px" }
      : { "border-top-color": "var(--card-edge)" });
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
      const expected = theme === "light" ? (selector === "#ovrwrap"
        ? { "border-top-color": "var(--card-edge)", order: "0", padding: "10px 24px 4px" }
        : selector === "#composer" ? { padding: "8px 24px 20px" } : {}) : {};
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

test("desktop dark stays neutral, light keeps low-saturation pastel fills and eye-care keeps its warm band", () => {
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
      if (name === "light") {
        const pastel = ["--bg", "--surface", "--surface-hover", "--surface-2", "--code-bg", "--line", "--line-2"];
        if (pastel.includes(token)) {
          assert.ok(chroma <= 24, token + " exceeds the pastel channel spread");
        } else {
          assert.ok(chroma <= 8, token + " changed the existing neutral token");
          assert.ok(r >= g && g >= b, token + " changed the existing neutral direction");
        }
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

test("desktop text and selected rows stay readable on their theme surfaces", () => {
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
    if (name === "light") {
      assert.equal(palette["--bg"], "#f1f1f1", "the canvas stays below paper white");
      assert.equal(palette["--panel"], "#f3f3f3", "the panel uses the existing white body surface");
    } else if (name === "warm") {
      assert.ok(relativeLuminance(palette["--panel"]) <= 0.9, name + " desktop panel is too bright");
    }
    assert.ok(relativeLuminance(palette["--panel"]) > relativeLuminance(palette["--bg"]), name + " page/panel hierarchy");
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

// Pin shared and phone CSS, including the compact mobile spacing and the
// user/queued-note footer matching the existing assistant footer declarations.
// The snapshot includes the shared attachment summary and the intentional
// paragraph-sized KaTeX rule with automatic overflow on both axes, plus the
// mobile warm user timestamp color protected by its contrast test above.
test("desktop contrast rules preserve the current shared and mobile CSS bytes", () => {
  const phoneApplicable = desktopMediaQueries().reduce((source, block) => source.replace(block.css, ""), style);
  assert.doesNotMatch(phoneApplicable, /--(?:control-border|danger-text)/);
  assert.equal(createHash("sha256").update(phoneApplicable).digest("hex"),
    "dc3656b0519d34d224c5c3119805973e96ccd47a7c043d68414f9f9ec69eb461",
    "desktop contrast changed shared or mobile CSS");
});

test("desktop overrides leave all phone-applicable CSS byte-identical", () => {
  const desktop = desktopMediaQueries();
  assert.equal(desktop.length, 3, "expected the desktop layout, narrow desktop layout, and width tuning media queries");
  let phoneApplicable = style;
  for (const block of desktop) phoneApplicable = phoneApplicable.replace(block.css, "");
  assert.equal(createHash("sha256").update(phoneApplicable).digest("hex"),
    "dc3656b0519d34d224c5c3119805973e96ccd47a7c043d68414f9f9ec69eb461",
    "shared or phone CSS changed outside the desktop queries");
});

// Frozen exact-selector declarations from 37000ed6, with retained command-row
// fills for the newer completion menu. Shared rules are also protected by the
// phone/non-light hashes below; no live Git lookup is needed.
const restoredLightDeclarations = {
  ".msg.asst": {
    "align-self": "stretch",
    "background": "var(--panel)",
    "border": "1px solid var(--line)",
    "box-shadow": "none",
    "border-top-left-radius": "6px"
  },
  ".msg.user": {
    "align-self": "flex-end",
    "background": "var(--surface-2)",
    "color": "var(--fg)",
    "border-bottom-right-radius": "6px",
    "max-width": "85%",
    "border": "1px solid var(--line)",
    "box-shadow": "0 1px 2px rgba(15,23,42,.05)"
  },
  ".user-msg-row > .msg.user": {
    "border-left": "3px solid var(--acc)",
    "min-width": "0",
    "align-self": "auto"
  },
  ".msg.md blockquote": {
    "font-weight": "var(--fw-meta)",
    "margin": "4px 0",
    "padding": "6px 12px",
    "border-left": "3px solid var(--line)",
    "color": "var(--dim)",
    "border-color": "var(--acc-line)",
    "background": "var(--tint-2)",
    "border-radius": "0 10px 10px 0"
  },
  ".msg.md code": {
    "letter-spacing": "normal",
    "font-family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    "font-size": "14px",
    "background": "var(--code-bg)",
    "border": "1px solid var(--line)",
    "border-radius": "5px",
    "padding": "0 4px",
    "overflow-wrap": "anywhere"
  },
  ".msg.md pre": {
    "letter-spacing": "normal",
    "margin": "6px 0",
    "padding": "12px",
    "background": "var(--code-bg)",
    "border": "1px solid var(--line)",
    "border-radius": "var(--radius-control)",
    "overflow-x": "auto"
  },
  ".msg.md pre code": {
    "background": "none",
    "border": "0",
    "padding": "0",
    "white-space": "pre-wrap",
    "overflow-wrap": "anywhere"
  },
  ".msg.md .code-block": {
    "margin": "6px 0"
  },
  ".msg.md .code-block-toolbar": {
    "display": "flex",
    "justify-content": "space-between",
    "align-items": "center",
    "gap": "8px",
    "margin-bottom": "8px"
  },
  ".msg.md .tablewrap": {
    "overflow-x": "auto",
    "margin": "12px 0"
  },
  ".msg.md table": {
    "border-collapse": "collapse",
    "font-size": "15px"
  },
  ".msg.md th": {
    "border": "0",
    "padding": "8px 12px",
    "text-align": "left",
    "border-bottom": "1px solid var(--card-edge)",
    "vertical-align": "top",
    "background": "var(--surface-2)",
    "color": "var(--fg)",
    "font-weight": "600",
    "border-bottom-color": "var(--line)"
  },
  ".msg.md td": {
    "border": "0",
    "padding": "8px 12px",
    "text-align": "left",
    "border-bottom": "1px solid var(--card-edge)",
    "vertical-align": "top"
  },
  ".msg.md table code": {
    "font-size": "14px",
    "background": "var(--tint-2)",
    "border": "0",
    "border-radius": "3px",
    "padding": "1px 3px"
  },
  ".msg.md .code-preview iframe": {
    "display": "block",
    "width": "100%",
    "max-height": "min(70dvh,720px)",
    "border": "1px solid var(--line)",
    "border-radius": "var(--radius-control)",
    "background": "var(--code-bg)"
  },
  ".msg.md .media-image img": {
    "display": "block",
    "width": "auto",
    "max-width": "100%",
    "max-height": "min(70dvh,720px)",
    "border": "1px solid var(--line)",
    "border-radius": "var(--radius-control)",
    "background": "var(--media-bg)"
  },
  ".msg.md .media-video video": {
    "display": "block",
    "width": "100%",
    "max-width": "100%",
    "max-height": "min(70dvh,720px)",
    "border": "1px solid var(--line)",
    "border-radius": "var(--radius-control)",
    "background": "var(--media-bg)",
    "aspect-ratio": "16/9"
  },
  ".msg.phone-frame img": {
    "display": "block",
    "width": "auto",
    "max-width": "100%",
    "max-height": "min(30dvh,260px)",
    "object-fit": "contain",
    "margin": "0 auto",
    "border": "1px solid var(--line)",
    "border-radius": "var(--radius-control)",
    "background": "var(--media-bg)"
  },
  "#sidequeuelist li": {
    "display": "flex",
    "align-items": "flex-start",
    "gap": "6px",
    "padding": "6px 8px",
    "border": "1px solid var(--line)",
    "border-radius": "8px",
    "background": "var(--tint-2)",
    "font-size": "12px",
    "line-height": "1.4"
  },
  "#sessions li.cur": {
    "background": "var(--tint-3)",
    "box-shadow": "inset 3px 0 0 var(--acc)"
  },
  "#sessions li.cur:hover": {
    "background": "var(--tint-3)",
    "box-shadow": "inset 3px 0 0 var(--acc)"
  },
  "#sessions li:hover": {
    "background-color": "var(--surface-hover)",
    "background": "var(--surface-2)"
  },
  ".modelrow:hover": {
    "background": "var(--surface-2)"
  },
  ".cmdrow:hover": { "background": "var(--surface-2)" },
  ".cmdrow[data-active=\"true\"]": {
    "background": "var(--surface-2)", "box-shadow": "inset 3px 0 0 var(--acc)"
  },
  "#promptlist button:hover": {
    "background": "var(--surface-2)",
    "border-left-color": "var(--acc)"
  },
  "#jumpbottom:hover": {
    "background": "var(--surface-2)"
  },
  "#img:hover:not(:disabled)": {
    "background": "var(--surface-2)",
    "color": "var(--fg)"
  },
  "#img:focus-visible": {
    "background": "var(--surface-2)",
    "color": "var(--fg)"
  },
  ".msg.asst > details > summary:hover": {},
  ".msg.pending": {
    "font-weight": "var(--fw-meta)",
    "color": "var(--dim)",
    "font-style": "italic"
  },
  ".msg.err": {
    "border-color": "var(--err)",
    "color": "var(--err)"
  },
  ".turn .tool-activity[data-running=\"true\"]": {
    "border-color": "var(--acc-line)"
  }
};
function assertRestoredLightDeclarations(selectors) {
  const shared = style.slice(0, style.indexOf("@media (min-width:701px)"));
  const desktop = desktopMediaQueries()[0].body;
  for (const selector of selectors) {
    assert.deepEqual({ ...ruleValues(shared, selector), ...ruleValues(desktop, selector) },
      restoredLightDeclarations[selector], selector + " restores the baseline declarations");
    assert.deepEqual(ruleValues(desktop, ':root[data-theme="light"] ' + selector),
      [".msg.md pre", ".msg.md pre code"].includes(selector)
        ? { "white-space":"pre", "word-break":"normal", "overflow-wrap":"normal" } : {},
      selector + " retains only its desktop code whitespace protection");
  }
}

test("desktop light restores the neutral reading palette and original accent", () => {
  const palette = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktopMediaQueries()[0].body)) };
  const expected = { "--bg":"#f1f1f1", "--panel":"#f3f3f3", "--field":"#f3f3f3", "--surface":"#e5e5e5", "--surface-2":"#efefef", "--surface-hover":"#d9d9d9", "--code-bg":"#e8e8e8", "--code-toolbar":"#efefef", "--media-bg":"#e0e0e0", "--acc":"#246b60" };
  for (const [token, value] of Object.entries(expected)) assert.equal(palette[token], value, token);
  for (const token of ["--bg", "--panel", "--field", "--surface-2", "--code-bg", "--media-bg"]) {
    assert.ok(contrast(palette["--fg"], palette[token]) >= 7, "body contrast on " + token);
    assert.ok(contrast(palette["--acc"], palette[token]) >= 4.5, "link contrast on " + token);
    assert.ok(contrast(palette["--control-border"], palette[token]) >= 3, "control boundary on " + token);
  }
  assertRestoredLightDeclarations([".msg.asst", ".msg.user"]);
});

test("desktop light notes restore the original panel and queued-note tint", () => {
  const desktop = desktopMediaQueries()[0].body, light = ':root[data-theme="light"] ';
  assertRestoredLightDeclarations([".msg.asst", "#sidequeuelist li"]);
  for (const suffix of ["", " > details", " > details > summary"]) {
    assert.deepEqual(ruleValues(desktop, light + `.msg.asst[data-canonical-identity^='["internal_note",']` + suffix), {});
  }
  assert.deepEqual(ruleValues(desktop, light + "#sidenote.show"), { background:"var(--field)", "border-color":"var(--acc)", color:"var(--acc)" });
  assert.equal(tokenValues(tokenBlock(PALETTES.light, desktop))["--panel"], "#f3f3f3");
});

test("desktop light Markdown tables restore intrinsic layout and baseline horizontal rules", () => {
  assertRestoredLightDeclarations([".msg.md .tablewrap", ".msg.md table", ".msg.md th", ".msg.md td", ".msg.md table code"]);
  const desktop = desktopMediaQueries()[0].body;
  assert.equal(ruleValues(desktop, ".msg.md th").display, undefined, "native table columns remain intact");
  assert.doesNotMatch(desktop, /:root\[data-theme="light"\][^{}]*(?:tbody|tr):nth-child\([^)]*\)\s*\{/);
});

test("desktop light cards and user accents restore their baseline outlines", () => {
  const desktop = desktopMediaQueries()[0].body, light = ':root[data-theme="light"] ';
  assert.deepEqual(ruleValues(desktop, light + ".msg.asst:not(.git-output)"), { "border-color":"var(--card-edge)", "border-radius":"var(--radius-card)", "box-shadow":"var(--card-shadow)" });
  assertRestoredLightDeclarations([".user-msg-row > .msg.user", ".msg.asst > details > summary:hover"]);
  for (const selector of ['.msg.asst:has(> details[data-fold-key^="thinking:"])', '.msg.asst:has(> details[data-fold-key$=":thinking"])']) {
    assert.deepEqual(ruleValues(desktop, light + selector), { "border-color":"var(--card-edge)", "box-shadow":"none" });
  }
});

test("desktop light quotations, code and media restore their original surfaces", () => {
  assertRestoredLightDeclarations([".msg.md blockquote", ".msg.md code", ".msg.md pre", ".msg.md pre code", ".msg.md .code-block", ".msg.md .code-block-toolbar", ".msg.md table code", ".msg.md .code-preview iframe", ".msg.md .media-image img", ".msg.md .media-video video", ".msg.phone-frame img"]);
});

test("desktop light status and hover visuals restore the existing baseline rules", () => {
  const desktop = desktopMediaQueries()[0].body, light = ':root[data-theme="light"] ';
  assert.deepEqual(ruleValues(desktop, light + '.msg.asst.tool-activity[data-running="true"]'), { "border-color":"var(--acc-line)" });
  for (const selector of ['.msg.asst[data-execution-state="completed"]', '.msg.asst.pending', '.msg.asst[data-execution-state="running"]', '.msg.asst.err', '.msg.asst[data-execution-exception="true"]', '.msg.asst:has(> .task-output > .task-error)']) {
    assert.deepEqual(ruleValues(desktop, light + selector), {}, selector + " uses its original status surface");
  }
  assertRestoredLightDeclarations([".msg.pending", ".msg.err", '.turn .tool-activity[data-running="true"]', ".modelrow:hover", ".cmdrow:hover", '.cmdrow[data-active="true"]', "#promptlist button:hover", "#jumpbottom:hover", "#img:hover:not(:disabled)", "#img:focus-visible"]);
});

test("desktop light session selection and hover restore neutral fills and readable metadata", () => {
  assertRestoredLightDeclarations(["#sessions li.cur", "#sessions li.cur:hover", "#sessions li:hover"]);
  const palette = { ...tokenValues(tokenBlock(PALETTES.light)), ...tokenValues(tokenBlock(PALETTES.light, desktopMediaQueries()[0].body)) };
  assert.equal(palette["--tint-3"], "rgba(46,46,46,.14)");
  assert.equal(palette["--surface-2"], "#efefef");
  for (const color of ["#f3f3f3", "#efefef", "#d7d7d7"]) {
    assert.ok(contrast(palette["--fg"], color) >= 7, "session title contrast on " + color);
    assert.ok(contrast(palette["--dim"], color) >= 7, "metadata contrast on " + color);
  }
  assert.equal(ruleValues(style, '#sessions li.cur span[role="button"]')["font-weight"], "600");
});

test("desktop near-white changes preserve dark and warm token blocks byte for byte", () => {
  const sources = { shared: style.slice(0, style.indexOf("@media (min-width:701px)")), desktop: desktopMediaQueries()[0].body };
  const hashes = {
    shared_dark: "12c8183613f4dce5fb703d9de5f6a6274c0ee0049d5db5f323ae239f06cc08ff",
    shared_warm: "98d6aa190c66db657bf77a71021b2aaeec46da2b2a857fb9fe342f0b8b923a54",
    desktop_dark: "8c298e38bc187003be269b1394f42f24d1c608a579c3b005c16be2d88c25f592",
    desktop_warm: "2d255eafaeedb07824714821d8454aa1fc03dce573baccd04ba7f305c765ed84",
  };
  for (const [scope, source] of Object.entries(sources)) {
    for (const name of ["dark", "warm"]) {
      const start = source.indexOf(PALETTES[name]);
      assert.ok(start >= 0);
      const block = source.slice(start, source.indexOf("}", start) + 1);
      assert.equal(createHash("sha256").update(block).digest("hex"), hashes[scope + "_" + name]);
    }
  }
});

test("desktop pastel, paragraph-sized math and mobile warm timestamps pin every non-light rule and declaration", () => {
  const protectedCss = style.replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/:root\[data-theme="light"\][^{]*\{[^{}]*\}/g, "")
    .replace(/\s+/g, "");
  assert.equal(createHash("sha256").update(protectedCss).digest("hex"),
    "9059b25e1238a7915ad19e6449b2ffa5014e91a7e2f6f966ce477a36b0a28ba1",
    "dark or warm CSS changed beyond the pinned baseline");
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
