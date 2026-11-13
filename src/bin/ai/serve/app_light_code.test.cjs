// Run with: env -u NODE_OPTIONS node --test src/bin/ai/serve/app_light_code.test.cjs
// Desktop light prose and code contracts. The DOM double rejects HTML sinks.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const style = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const light = ':root[data-theme="light"] ';

function rule(selector) {
  const source = style.replace(/\/\*[\s\S]*?\*\//g, "");
  const values = {};
  for (const match of source.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    // Selectors used here contain no commas in attribute values or functions.
    if (!match[1].trim().split(/,\s*(?=:root)/).includes(selector)) continue;
    for (const entry of match[2].split(";")) {
      const at = entry.indexOf(":");
      if (at >= 0) values[entry.slice(0, at).trim()] = entry.slice(at + 1).trim();
    }
  }
  return values;
}

class Node {
  constructor(tag = "#text", text = "") {
    this.tagName = tag;
    this.nodeType = tag === "#text" ? 3 : tag === "#fragment" ? 11 : 1;
    this.children = [];
    this.dataset = {};
    this.value = text;
    this.className = "";
    this.classList = {
      add: (name) => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), name])].join(" "); },
      remove: (name) => { this.className = this.className.split(/\s+/).filter((v) => v !== name).join(" "); },
      contains: (name) => this.className.split(/\s+/).includes(name),
    };
  }
  set innerHTML(_) { throw new Error("code must never enter an HTML sink"); }
  get textContent() { return this.nodeType === 3 ? this.value : this.children.map((n) => n.textContent).join(""); }
  set textContent(text) {
    this.children = [new Node("#text", String(text))];
  }
  appendChild(node) {
    if (node.nodeType === 11) { for (const child of node.children) this.appendChild(child); }
    else { this.children.push(node); node.parentElement = this; }
    return node;
  }
  replaceChildren(...nodes) { this.children = []; for (const node of nodes) this.appendChild(node); }
}

function harness({ theme = "light", width = 1280 } = {}) {
  const state = { theme, width }, copied = [], callbacks = [], observers = [];
  const code = new Node("code"), pre = new Node("pre");
  pre.appendChild(code);
  const root = { querySelectorAll: () => [code] };
  const context = vm.createContext({
    document: {
      documentElement: { getAttribute: () => state.theme },
      createElement: (tag) => new Node(tag),
      createTextNode: (text) => new Node("#text", text),
      createDocumentFragment: () => new Node("#fragment"),
    },
    window: {
      matchMedia: () => ({ matches: state.width >= 701, addEventListener: (_, cb) => callbacks.push(cb) }),
      MutationObserver: class { constructor(cb) { observers.push(cb); } observe() {} },
    },
    $: () => root,
    copyCodeText: async (text) => { copied.push(text); return true; },
    setTimeout() {},
  });
  const from = html.indexOf("// BEGIN desktop light code"), to = html.indexOf("// END desktop light code");
  if (from >= 0 && to > from) vm.runInContext(html.slice(from, to), context);
  const copyStart = html.indexOf("function copyNearbyCode("), copyEnd = html.indexOf("// SVG is active", copyStart);
  vm.runInContext(html.slice(copyStart, copyEnd), context);
  return {
    state, code, pre, root, context, copied, callbacks, observers,
    render(source, language) {
      code.textContent = source;
      pre.dataset.lang = language;
      assert.equal(typeof context.decorateDesktopLightCode, "function", "desktop light code decorator is missing");
      context.decorateDesktopLightCode(root);
      return code;
    },
  };
}

// The baseline used the shared code and quotation rules. Resolve those exact
// selectors across shared and desktop CSS, then apply any light-only declaration.
function restoredRule(selector) {
  const desktopStart = style.indexOf("@media (min-width:701px)");
  const source = style.slice(0, style.indexOf("/* Keep narrow desktop"));
  const sharedAndDesktop = (target) => {
    const result = {};
    for (const match of source.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      if (!match[1].trim().split(/,\s*/).includes(target)) continue;
      for (const entry of match[2].split(";")) {
        const at = entry.indexOf(":");
        if (at >= 0) result[entry.slice(0, at).trim()] = entry.slice(at + 1).trim();
      }
    }
    return result;
  };
  assert.ok(desktopStart > 0);
  return { ...sharedAndDesktop(selector), ...rule(light + selector) };
}

test("desktop light main answers restore baseline card borders and execution folds", () => {
  const card = rule(light + ".msg.asst:not(.git-output)");
  assert.deepEqual(card, { "border-color": "var(--card-edge)", "border-radius": "var(--radius-card)", "box-shadow": "var(--card-shadow)" });
  const answer = rule(light + ".msg.asst.md:not(.pending):not(.err):not(.git-output)");
  assert.equal(answer.border, "1px solid var(--card-edge)");
  assert.equal(restoredRule(".msg.asst").background, "var(--panel)");
  assert.equal(rule(light + '.msg.asst.tool-activity[data-running="true"]')["border-color"], "var(--acc-line)");
  for (const selector of ['.msg.asst:has(> details[data-fold-key^="thinking:"])', '.msg.asst:has(> details[data-fold-key$=":thinking"])']) {
    assert.deepEqual(rule(light + selector), { "border-color": "var(--card-edge)", "box-shadow": "none" });
  }
});

test("desktop light user bubbles and quotations restore the baseline accent and tinted quote", () => {
  assert.equal(restoredRule(".user-msg-row > .msg.user")["border-left"], "3px solid var(--acc)");
  const quote = restoredRule(".msg.md blockquote");
  assert.equal(quote.background, "var(--tint-2)");
  assert.equal(quote["border-left"], "3px solid var(--line)");
  assert.equal(quote["border-color"], "var(--acc-line)");
  assert.equal(quote["border-radius"], "0 10px 10px 0");
  assert.equal(restoredRule(".msg.md code").border, "1px solid var(--line)");
  for (const cell of ["th", "td"]) {
    assert.equal(restoredRule(".msg.md " + cell).border, "0");
    assert.equal(restoredRule(".msg.md " + cell)["border-bottom"], "1px solid var(--card-edge)");
  }
});

test("desktop light code retains non-wrapping whitespace, long-line scrolling and baseline toolbar metrics", () => {
  assert.deepEqual(restoredRule(".msg.md .code-block"), { margin: "6px 0" });
  const toolbar = restoredRule(".msg.md .code-block-toolbar");
  assert.equal(toolbar["margin-bottom"], "8px");
  assert.equal(toolbar.gap, "8px");
  assert.equal(restoredRule(".msg.md .code-block-toolbar button")["border-radius"], "999px");
  const code = restoredRule(".msg.md pre code");
  assert.equal(code["white-space"], "pre");
  assert.equal(code["word-break"], "normal");
  assert.equal(code["overflow-wrap"], "normal");
  assert.equal(code.background, "none");
  assert.equal(code.border, "0");
  const pre = restoredRule(".msg.md pre");
  assert.equal(pre.background, "var(--code-bg)");
  assert.equal(pre.border, "1px solid var(--line)");
  assert.equal(pre["border-radius"], "var(--radius-control)");
  assert.equal(pre["overflow-x"], "auto");
  assert.equal(pre["white-space"], "pre");
  assert.equal(pre["word-break"], "normal");
  assert.equal(pre["overflow-wrap"], "normal");
  assert.equal(restoredRule(".msg.md code")["font-size"], "14px");
  for (const kind of ["keyword", "string", "comment", "number", "function", "punctuation"]) {
    assert.deepEqual(rule(light + ".msg.md .code-token-" + kind), {}, kind + " inherits the baseline text appearance");
  }
});

const samples = {
  javascript: 'const greet = (name) => console.log("hi", 42); // note',
  typescript: 'interface User { name: string }\nfunction greet(user: User) { return "hi" + 42; } // note',
  python: 'def greet(name):\n    # note\n    return print("hi", 42)',
  rust: 'fn main() { /* note */ println!("hi {}", 42); }',
  json: '{"name": "hi", "count": 42, "ready": true}',
  bash: '# note\nif true; then echo "hi"; exit 42; fi',
  yaml: '# note\nname: "hi"\ncount: 42\nready: true',
  html: '<!-- note --><div class="hi">42</div>',
  css: '/* note */ .hi { color: "red"; width: 42px; display: block; }',
};
for (const [language, source] of Object.entries(samples)) {
  test("safe tokenizer covers " + language + " and retains exact source bytes", () => {
    const h = harness(), code = h.render(source, language);
    assert.equal(code.textContent, source);
    assert.ok(code.classList.contains("code-highlighted"));
    const kinds = new Set(code.children.filter((node) => node.nodeType === 1).map((node) => node.className));
    assert.ok(kinds.has("code-token-string"), "quoted strings are highlighted");
    assert.ok(kinds.has("code-token-number"), "numbers are highlighted");
    if (language !== "json") assert.ok(kinds.has("code-token-comment"), "comments are highlighted");
    if (!["yaml"].includes(language)) assert.ok(kinds.has("code-token-keyword"), "keywords are highlighted");
    if (["javascript", "typescript", "python", "rust", "bash", "yaml", "css"].includes(language)) {
      assert.ok(kinds.has("code-token-function"), "calls or property names are highlighted");
    }
  });
}

test("code payloads containing HTML, quotes, backticks and interpolation stay text", async () => {
  const source = '</p><script>alert("x")</script>\nconst text = `"\' ${value}`; 🦀';
  const h = harness(), code = h.render(source, "js");
  assert.equal(code.textContent, source);
  assert.equal(h.pre.children.length, 1, "the listing cannot create sibling elements");
  for (const node of code.children) {
    assert.ok(node.nodeType === 3 || node.tagName === "span");
    assert.doesNotMatch(node.textContent, /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/,
      "a token must not split a Unicode surrogate pair");
    if (node.tagName === "span") {
      assert.equal(node.children.length, 1);
      assert.equal(node.children[0].nodeType, 3, "each token contains only a text node");
      assert.match(node.className, /^code-token-(keyword|string|comment|number|function|punctuation)$/);
    }
  }
  const host = { querySelector: () => code }, button = { closest: () => host, isConnected: true };
  h.context.copyNearbyCode(button);
  await Promise.resolve();
  assert.deepEqual(h.copied, [source], "copy retains unescaped original code");
  assert.equal(button.textContent, "copied");
});

test("unknown language and tokenizer errors keep the existing monochrome text", () => {
  const h = harness(), source = 'x <script> " \' ` ${value}';
  h.render(source, "made-up-language");
  assert.equal(h.code.children.length, 1);
  assert.equal(h.code.children[0].nodeType, 3);
  vm.runInContext('tokenizeLightCode = () => { throw new Error("test failure"); };', h.context);
  h.render(source, "javascript");
  assert.equal(h.code.textContent, source);
  assert.equal(h.code.children[0].nodeType, 3);
  assert.equal(h.code.classList.contains("code-highlighted"), false);
});

test("phone layouts and dark or warm palettes never tokenize new code", () => {
  for (const theme of ["dark", "warm", "light"]) for (const width of [700, 701, 1280]) {
    if (theme === "light" && width > 700) continue;
    const h = harness({ theme, width });
    h.render(samples.javascript, "javascript");
    assert.equal(h.code.children.length, 1, theme + " at " + width);
    assert.equal(h.code.children[0].nodeType, 3);
    assert.equal(h.callbacks.length, 1, "a passive viewport listener can detect the first desktop light switch");
    assert.equal(h.observers.length, 1, "a passive theme listener does not alter monochrome rendering");
    vm.runInContext('tokenizeLightCode = () => { throw new Error("must not tokenize outside desktop light"); };', h.context);
    h.callbacks[0]();
    h.observers[0]();
    assert.equal(h.code.children.length, 1);
    assert.equal(h.code.children[0].nodeType, 3);
  }
});

test("theme and viewport changes restore monochrome and reapply highlighting", () => {
  const h = harness();
  h.render(samples.rust, "rs");
  assert.equal(h.callbacks.length, 1);
  assert.equal(h.observers.length, 1);
  h.state.theme = "warm";
  h.observers[0]();
  assert.equal(h.code.children.length, 1);
  assert.equal(h.code.children[0].nodeType, 3);
  h.state.theme = "light";
  h.observers[0]();
  assert.ok(h.code.classList.contains("code-highlighted"));
  h.state.width = 700;
  h.callbacks[0]();
  assert.equal(h.code.children[0].nodeType, 3);
  h.state.width = 701;
  h.callbacks[0]();
  assert.ok(h.code.classList.contains("code-highlighted"));
  assert.equal(h.code.textContent, samples.rust);
  assert.equal(h.callbacks.length, 1, "only one viewport listener is installed");
  for (const start of [{ theme: "dark", width: 1280 }, { theme: "light", width: 700 }]) {
    const fresh = harness(start);
    fresh.render(samples.python, "py");
    fresh.state.theme = "light";
    fresh.state.width = 1280;
    fresh.observers[0]();
    assert.ok(fresh.code.classList.contains("code-highlighted"), "the first switch highlights already rendered code");
  }
});

test("stream decoration is idempotent and uses the current listing", () => {
  const h = harness();
  h.render(samples.python, "py");
  const nodes = [...h.code.children];
  h.context.decorateDesktopLightCode(h.root);
  assert.deepEqual(h.code.children, nodes, "unchanged code keeps its DOM nodes");
  h.render(samples.python + '\nprint("tail")', "py");
  assert.equal(h.code.textContent, samples.python + '\nprint("tail")');
  assert.ok(h.code.classList.contains("code-highlighted"));
});

test("existing code markup keeps fence labels, default code label and copy buttons", () => {
  const source = html.slice(html.indexOf("function codeBlock("), html.indexOf("// Copying uses"));
  const context = vm.createContext({ esc: (text) => String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]) });
  vm.runInContext(source, context);
  assert.match(context.codeBlock("x", "typescript"), /class="code-lang">typescript<\/span>/);
  assert.match(context.codeBlock("x", ""), /class="code-lang">code<\/span>/);
  assert.match(context.codeBlock("</p><script>", "js"), /&lt;\/p&gt;&lt;script&gt;/);
  assert.match(context.codeBlock("x", "js"), /data-copy-code/);
});

test("the live and history decoration path applies the light-only code decorator", () => {
  const body = html.slice(html.indexOf("function decorateAndLoad("), html.indexOf("function showViewer("));
  assert.match(body, /decorateDesktopLightCode\(root\)/);
  const calls = [], context = vm.createContext(Object.fromEntries([
    "decorateDesktopLightCode", "decorateMath", "decorateMedia", "loadPreviews",
  ].map((name) => [name, () => calls.push(name)])));
  vm.runInContext(body, context);
  context.decorateAndLoad({});
  assert.deepEqual(calls, ["decorateDesktopLightCode", "decorateMath", "decorateMedia", "loadPreviews"]);
});
