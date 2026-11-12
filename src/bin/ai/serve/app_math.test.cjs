// Run with: node --test src/bin/ai/serve/app_math.test.cjs
// Exercise the embedded client itself; no npm dependencies or network required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
// Assistant and tool rows render inside a per-turn wrapper, so ordered
// assertions read row keys through that wrapper instead of assuming the
// transcript's direct children are the rows themselves.
function renderedNodes(root) {
  const nodes = [];
  const walk = (node) => {
    if (typeof node.key === "string") {
      nodes.push(node);
      return;
    }
    for (const child of node.children || []) walk(child);
  };
  for (const child of root.children || []) walk(child);
  return nodes;
}
function renderedKeys(root) {
  return renderedNodes(root).map((node) => node.key);
}
// The slices below exercise individual functions; grouping and image drafts
// pull in whole-page state those slices do not carry. Supply inert versions so
// a call succeeds; app_desktop/app_task_results cover the real behaviour.
const createContext = vm.createContext.bind(vm);
vm.createContext = (sandbox = {}) => {
  const context = createContext({
  executionGroupsFor: () => ({ sync() {}, restore() {} }),
  pendingImages: [],
  window: { matchMedia: () => ({ matches: false }), addEventListener() {}, removeEventListener() {} },
  createTurnRateLine: () => () => {},
  // Rate decoration is display-only; app_desktop covers its real rendering.
  showTurnRate: () => {},
  historyExecutionOf: () => [],
  ...sandbox,
  });
  vm.runInContext(script.slice(script.indexOf("// BEGIN execution model"), script.indexOf("// END execution model")), context);
  return context;
};
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const historyRetrySrc = script.slice(script.indexOf("// BEGIN history retry"), script.indexOf("// END history retry"));
const userOriginHelpers = script.slice(script.indexOf("function isRuntimeSyntheticUser("), script.indexOf("const PHONE_LAYOUT_QUERY"));
// The layout-hold helpers are shared by the composer and the turn state; the
// slices below run on their own, so they carry the helpers with them.
const layoutHoldSrc = script.slice(
  script.indexOf("let promptComposing = false;"),
  script.indexOf("function composerBox()"),
);
const renderer = script.slice(script.indexOf("const esc ="), script.indexOf("function sessionTitle("));
const decode = (s) => s.replace(/&(amp|lt|gt|quot);/g, (_, key) => ({ amp: "&", lt: "<", gt: ">", quot: '"' })[key]);
function nodeList(nodes = []) {
  const list = {
    ...nodes, length: nodes.length,
    item(index) { return this[index] ?? null; },
    forEach(callback, thisArg) { nodes.forEach((node, index) => callback.call(thisArg, node, index, list)); },
    [Symbol.iterator]() { return nodes[Symbol.iterator](); },
  };
  return list;
}
function nodesOf(markup) {
  return [...markup.matchAll(/<span class="math (display|inline)" data-tex="([^"]*)">([\s\S]*?)<\/span>/g)].map((m) => ({
    dataset: { tex: decode(m[2]) }, textContent: decode(m[3]),
    classList: { contains: (name) => name === m[1] },
  }));
}
function harness() {
  const assets = [], calls = [];
  const msgs = {
    nodes: [], scrollTop: 0, scrollHeight: 200, clientHeight: 100,
    querySelectorAll() { return nodeList(this.nodes.filter((node) => "tex" in node.dataset)); },
    querySelector() { return this.querySelectorAll()[0] || null; },
  };
  const element = (tag) => ({
    tag, children: [], dataset: {}, hidden: false, className: "",
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    appendChild(node) { this.children.push(node); return node; },
    append(node) { this.children.push(node); },
    insertBefore(node, anchor) {
      const index = anchor ? this.children.indexOf(anchor) : -1;
      if (index >= 0) this.children.splice(index, 0, node);
      else this.children.push(node);
      return node;
    },
    remove() {}, querySelector: () => null, querySelectorAll: () => [],
  });
  const document = { createElement: element, createTextNode: (text) => ({ text }), head: { appendChild: (el) => assets.push(el) } };
  const context = vm.createContext({ document, window: {}, $: () => msgs });
  vm.runInContext(renderer, context);
  const render = (input) => context.renderMarkdown(input);
  const setReply = (input) => { msgs.nodes = nodesOf(render(input)); };
  const installKatex = () => {
    context.window.katex = { render(tex, node, options) {
      calls.push({ tex, options });
      if (tex.includes("badcommand")) throw new Error("Invalid TeX");
      node.textContent = "typeset:" + tex;
    } };
  };
  const finishLoading = async () => {
    installKatex();
    assets.forEach((el) => el.onload());
    await vm.runInContext("mathLoading", context);
  };
  return { context, render, setReply, msgs, assets, calls, installKatex, finishLoading };
}

test("embedded script parses", () => { new vm.Script(script); });

for (const width of [390, 700, 701, 1280]) {
  test("prose whitespace edges compact only on desktop at " + width + "px", () => {
    const h = harness();
    h.context.window.matchMedia = (query) => {
      assert.equal(query, "(min-width:701px)");
      return { matches: width >= 701 };
    };
    const raw = "\n \t\n \n短回复\n \t\n";
    assert.equal(h.render(raw), width >= 701
      ? '<div class="p">短回复</div>'
      : '<div class="p"> \t<br> <br>短回复<br> \t</div>');
    assert.equal(h.render("\n短回复\n"), '<div class="p">短回复</div>');
    assert.equal(h.render("first\n \t\nsecond"), '<div class="p">first<br> \t<br>second</div>');
    assert.equal(h.render(" \t\n "), width >= 701 ? "" : '<div class="p"> \t<br> </div>');
    const code = h.render("```text\n \t\n  code  \n\n```");
    assert.match(code, /<pre data-lang="text"><code> \t\n  code  \n<\/code><\/pre>/);
  });
}

function pipeTableHarness(width = 1280) {
  const h = harness();
  h.context.window.matchMedia = (query) => ({ matches: query === "(min-width:701px)" && width >= 701 });
  return h;
}

const missingDelimiterTable = [
  "| 区域 | 原色 → 新色 |",
  "| 页面背景 | `#ede7db` → `#edeae4` |",
  "| 面板 | `#f6f0e4` → `#f5f2ec` |",
  "| 卡片底色 | `#e8e2d4` → `#e8e5de` |",
].join("\n");

for (const width of [390, 700, 701, 1280]) {
  test("pipe table without delimiter renders conservatively at " + width + "px", () => {
    const { render } = pipeTableHarness(width);
    const output = render(missingDelimiterTable);
    if (width < 701) {
      assert.doesNotMatch(output, /<table>/);
      assert.match(output, /\| 区域 \| 原色 → 新色 \|<br>/);
    } else {
      assert.match(output, /<thead><tr><th>区域<\/th><th>原色 → 新色<\/th><\/tr>/);
      assert.equal((output.match(/<td>/g) || []).length, 6);
      assert.match(output, /<td><code>#ede7db<\/code> → <code>#edeae4<\/code><\/td>/);
    }
    const standard = render("| a | b |\n| --- | --- |\n| c | d |");
    assert.equal(standard, '<div class="tablewrap"><table><thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>c</td><td>d</td></tr></tbody></table></div>');
  });

  test("pipe table without delimiter escapes list continuation at " + width + "px", () => {
    const { render } = pipeTableHarness(width);
    const output = render("- 表格可见内容：\n" + missingDelimiterTable.replace(/^/gm, "  ") + "\n- 后续说明");
    assert.equal((output.match(/class="li"/g) || []).length, 2);
    assert.equal((output.match(/<table>/g) || []).length, width >= 701 ? 1 : 0);
    assert.match(output, /后续说明/);
    assert.doesNotMatch(output, /<td>后续说明/);
  });
}

test("pipe table without delimiter protects code, escaped and math pipes", () => {
  const { render } = pipeTableHarness();
  const output = render("| label | value |\n| `a|b` | left\\|right |\n| $|x|$ | **bold** |");
  assert.equal((output.match(/<td>/g) || []).length, 4);
  assert.match(output, /<td><code>a\|b<\/code><\/td><td>left\|right<\/td>/);
  assert.equal(nodesOf(output).length, 1);
  assert.match(output, /<td><b>bold<\/b><\/td>/);
  const codeEscape = render("| label | value |\n| `a\\|b` | c |\n| d | e |");
  assert.match(codeEscape, /<code>a\\\|b<\/code>/);
  const escaped = render('| label | value |\n| <script>alert(1)</script> | & |\n| "quote" | <img src=x onerror=bad> |');
  assert.match(escaped, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(escaped, /<(?:script|img)\b/);
});

test("pipe table without delimiter rejects short, ragged and ambiguous text", () => {
  const { render } = pipeTableHarness();
  for (const source of [
    "| a | b |", "| a | b |\n| c | d |",
    "a | b\nc | d\ne | f", "| a |\n| b |\n| c |",
    "| a | b |\n| c | d | e |\n| f | g |\n| h | i |",
    "| a | b |\n| c | d |\n| e | f |\n| g | h | i |",
    "| a | b |\n| c | d |\n| e | f |\n| truncated |",
    "| | b |\n| c | d |\n| e | f |",
    "| a | b |\n| c | d |\n   \n| e | f |",
    "| a | b |\n| c | d |\n| --- | ---",
    "| a | b \\|\n| c | d \\|\n| e | f \\|",
  ]) assert.doesNotMatch(render(source), /<table>/, source);
});

test("markdown shows backslash-escaped punctuation without the backslash", () => {
  const { render } = harness();
  // JSON-encoded reply text reaches the page with `\"`; Markdown treats the
  // backslash as syntax for the quote, not as a character to display.
  const prose = render('他说\\"你好\\"，路径 C:\\Users\\demo，正则 \\n，字面 \\*星号\\*');
  assert.equal(prose,
    '<div class="p">他说&quot;你好&quot;，路径 C:\\Users\\demo，正则 \\n，字面 *星号*</div>');
  assert.doesNotMatch(prose, /<i>|<b>/);
  // Code keeps its exact source, including a backslash before a quote.
  assert.match(render('代码 `a\\"b`'), /<code>a\\&quot;b<\/code>/);
  assert.match(render('```js\nconst s = \\"x\\";\n```'), /<code>const s = \\&quot;x\\&quot;;<\/code>/);
  // Unescaping happens before markup is emitted, so it cannot inject tags.
  const hostile = render('<img src=x onerror=\\"alert(1)\\">');
  assert.doesNotMatch(hostile, /<img\b/);
  assert.match(hostile, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
});

test("pipe table inference preserves the legacy delimiter grammar", () => {
  const { render } = pipeTableHarness();
  // A blank pipe row or a later delimiter already starts a standard table
  // with its preceding row as the header, rather than an inferred table.
  for (const delimiter of ["| | |", "| --- | --- |"])
    assert.equal(render("| a | b |\n| c | d |\n" + delimiter + "\n| e | f |"),
      '<div class="p">| a | b |</div><div class="tablewrap"><table><thead><tr><th>c</th><th>d</th></tr></thead><tbody><tr><td>e</td><td>f</td></tr></tbody></table></div>');
});

test("pipe table without delimiter stays literal inside fenced code", () => {
  const { render } = pipeTableHarness();
  for (const fence of ["```", "~~~"]) {
    for (const ending of ["", "\n" + fence]) {
      const output = render(fence + "text\n" + missingDelimiterTable + ending);
      assert.match(output, /<pre data-lang="text">/);
      assert.doesNotMatch(output, /<table>/);
    }
    const nested = render("- code\n  " + fence + "text\n" + missingDelimiterTable.replace(/^/gm, "  ") + "\n  " + fence);
    assert.doesNotMatch(nested, /<table>/);
  }
});

test("a fenced block that continues a bullet renders as a code block", () => {
  const { render } = pipeTableHarness(1280);
  const output = render("- 登记 Redirect URI，格式为：\n  ```text\n  https://<域名>/oauth/codebase/callback\n  ```\n例如登记：");
  assert.doesNotMatch(output, /```/);
  assert.match(output, /<pre data-lang="text"><code>https:\/\/&lt;域名&gt;\/oauth\/codebase\/callback<\/code><\/pre>/);
  assert.match(output, /<div class="li"[^>]*><span class="bullet">•<\/span><span>登记 Redirect URI，格式为：<div class="code-block">/);
  assert.match(output, /<\/div><\/span><\/div><div class="p">例如登记：<\/div>/);
  const streaming = render("- code\n  ```text\n  partial");
  assert.match(streaming, /<pre data-lang="text"><code>partial<\/code><\/pre>/);
  assert.doesNotMatch(streaming, /```/);
  const bounded = render("- code\n  ```text\n  partial\nnext paragraph");
  assert.match(bounded, /<pre data-lang="text"><code>partial<\/code><\/pre>/);
  assert.match(bounded, /<div class="p">next paragraph<\/div>/);
  const spaced = render("1. step\n   ```text\n   a\n\n   b\n   ```\n   after");
  assert.match(spaced, /<pre data-lang="text"><code>a\n\nb<\/code><\/pre>/);
  assert.match(spaced, /<\/div>after<\/span><\/div>/);
});

test("pipe table without delimiter tolerates streaming prefixes without inventing cells", () => {
  const h = pipeTableHarness();
  const source = "| a | b |\n| c | d |\n| e | f |";
  for (const preview of [false, true]) {
    const render = (input) => h.context.renderMarkdown(input, preview);
    assert.doesNotMatch(render("| a | b |\n| c | d |\n| e | f"), /<table>/);
    assert.equal((render(source).match(/<td>/g) || []).length, 4);
    assert.equal((render(source + "\n\nfollowing prose").match(/<td>/g) || []).length, 4);
    assert.match(render(source + "\n\nfollowing prose"), /<\/table><\/div><div class="p">following prose<\/div>/);
  }
});

test("history ending in an unanswered user message is treated as interrupted", () => {
  const start = script.indexOf("function isRuntimeSyntheticUser(");
  const end = script.indexOf("const PHONE_LAYOUT_QUERY", start);
  assert.ok(start >= 0 && end > start);
  const context = vm.createContext({});
  vm.runInContext(script.slice(start, end), context);
  assert.equal(context.historyEndsUnanswered([{ role: "assistant" }, { role: "user" }]), true);
  assert.equal(context.historyEndsUnanswered([{ role: "user" }, { role: "assistant" }]), false);
  assert.equal(context.historyEndsUnanswered([{ role: "user" }, { role: "assistant" }, { role: "tool" }]), false);
  assert.equal(context.historyEndsUnanswered([{ role: "user", reasoning_content: "runtime-origin:synthetic-user:v1" }]), false);
  assert.equal(context.historyEndsUnanswered([]), false);
});

test("server base drops the app entry path before API calls", () => {
  const start = script.indexOf("function normalizeServeBase(");
  const end = script.indexOf("const store =", start);
  assert.ok(start >= 0 && end > start);
  const context = vm.createContext({ URL, location: { origin: "https://10.37.110.250:8080" } });
  vm.runInContext(script.slice(start, end), context);
  assert.equal(context.normalizeServeBase("https://10.37.110.250:8080/app"), "https://10.37.110.250:8080");
  assert.equal(context.normalizeServeBase("https://10.37.110.250:8080/app/?from=home"), "https://10.37.110.250:8080");
  assert.equal(context.normalizeServeBase("https://example.test/proxy"), "https://example.test/proxy");
});

test("drafts persist independently for each session and restore on switch", () => {
  const start = script.indexOf("const composerDrafts = new Map();");
  const end = script.indexOf("function questionRowsOf(", start);
  assert.ok(start >= 0 && end > start);
  const values = new Map([["serve.draft.beta", "beta draft"]]);
  const prompt = { value: "alpha draft" };
  const panelClasses = new Set();
  const elements = {
    prompt,
    questionpanel: { classList: { remove: (name) => panelClasses.delete(name) } },
    questionbtn: {
      classList: { toggle() {} },
      setAttribute() {},
    },
    questioncount: { textContent: "" },
    msgs: { querySelectorAll: () => nodeList() },
  };
  const context = vm.createContext({
    localStorage: {
      getItem: (key) => values.get(key) || null,
      setItem: (key, value) => values.set(key, value),
      removeItem: (key) => values.delete(key),
    },
    $: (id) => elements[id],
    growPrompt() {},
    setPromptValue(value) { prompt.value = value; return true; },
    turnRunning: false,
    syncStopButton() {},
    restoreSkillPick() {},
    restoreTeamPick() {},
    abortEvents() {},
    isPhoneLayout: () => true,
    pendingImages: [],
  });
  vm.runInContext(layoutHoldSrc, context);
  vm.runInContext(historyRetrySrc, context);
  vm.runInContext('const DRAFT_PREFIX = "serve.draft."; let questionCache = null; let sessionId = "alpha";\n' + script.slice(start, end), context);
  context.switchSession("beta");
  assert.equal(values.get("serve.draft.alpha"), "alpha draft");
  assert.equal(prompt.value, "beta draft");
  assert.equal(values.get("serve.session"), "beta");
  prompt.value = "";
  context.persistDraft();
  assert.equal(values.has("serve.draft.beta"), false);
});

test("question navigator fetches one bounded page and keeps global turn numbers", async () => {
  const start = script.indexOf("function questionRowsOf(");
  const end = script.indexOf("// `parent` mounts", start);
  assert.ok(start >= 0 && end > start);
  const userOrdinals = Array.from({ length: 103 }, (_, i) => ({
    role: "user",
    content: "prompt " + (i + 1),
    user_ordinal: i + 1,
  }));
  const links = [];
  const created = [];
  const paths = [];
  let panelOpen = false;
  let loaded = null;
  let scrolledTo = null;
  let widened = 0;
  const panel = { _shown: 20, classList: {
    contains: (name) => name === "open" && panelOpen,
    add: (name) => { if (name === "open") panelOpen = true; },
  } };
  const list = {
    innerHTML: "",
    textContent: "",
    scrollTop: 0,
    scrollHeight: 40 * 20,
    childElementCount: 0,
    dataset: {},
    appendChild: (node) => links.push(node),
    addEventListener: (name, fn) => { if (name === "scroll") list.onScroll = fn; },
  };
  // The transcript on screen holds only its own bounded tail, exactly like the
  // rendered history window.
  const bubbles = userOrdinals.slice(-20).map((row) => ({
    dataset: { question: row.content, historyIndex: String(row.user_ordinal - 1) },
    offsetWidth: 1,
    focus() {},
    scrollIntoView: () => { scrolledTo = row.content; },
    classList: { remove() {}, add() {} },
  }));
  const context = vm.createContext({
    sessionId: "session-a",
    historySession: "session-a",
    outputFollowSuspended: false,
    questionCache: null,
    $: (id) => ({
      questionpanel: panel,
      questionbtn: { setAttribute() {} },
      questionlist: list,
      msgs: { querySelectorAll: (sel) => (sel === ".msg.user" ? bubbles : []) },
    })[id],
    api: async (path) => {
      if (path === "/sessions/session-a/history?meta=roles") return { json: async () => userOrdinals.map(() => "user") };
      const match = path.match(/^\/sessions\/session-a\/history\?role=user&count=(\d+)$/);
      assert.ok(match, "the navigator asks only for real user prompts, one bounded page: " + path);
      const count = Number(match[1]);
      assert.ok(count <= 320, "the fetched page stays bounded: " + count);
      paths.push(path);
      return {
        json: async () => userOrdinals.slice(userOrdinals.length - count),
        headers: { get: () => String(userOrdinals.length) },
      };
    },
    textOf: (content) => String(content || ""),
    updateQuestionButton: (count) => { context.count = count; },
    closeQuestionPanel: () => { panelOpen = false; },
    loadHistory: async (prefetched) => {
      loaded = prefetched;
      widened += 1;
      for (const row of userOrdinals) {
        if (bubbles.some((bubble) => bubble.dataset.question === row.content)) continue;
        bubbles.push({
          dataset: { question: row.content, historyIndex: String(row.user_ordinal - 1) },
          offsetWidth: 1,
          focus() {},
          scrollIntoView: () => { scrolledTo = row.content; },
          classList: { remove() {}, add() {} },
        });
      }
    },
    esc: (value) => value,
    requestAnimationFrame: (fn) => fn(),
    document: { createElement: () => {
      const node = {
        children: [],
        append(...kids) { this.children = kids; },
        appendChild(kid) { kid.parentNode = this; this.children.push(kid); return kid; },
      };
      created.push(node);
      return node;
    } },
  });
  vm.runInContext(script.slice(start, end), context);
  vm.runInContext(userOriginHelpers, context);
  await context.openQuestionPanel();
  const questionLinks = links.filter((node) => node.className === "question-link");
  assert.equal(paths.length, 1, "the first open asks for exactly one page");
  assert.match(paths[0], /role=user&count=40$/, "one bounded page of prompts, not the transcript");
  assert.equal(context.count, 40, "the badge counts the prompts that were fetched");
  assert.equal(questionLinks.length, 20, "only one batch is rendered up front");
  assert.equal(list.scrollTop, list.scrollHeight, "opening lands on the newest prompt");
  assert.equal(list.dataset.hasOlder, "1", "older prompts are reachable by scrolling up");
  // The complaint this pins: every rendered entry carries its real turn number
  // even though the fetch covered only the newest page.
  const numbers = created.filter((node) => node.className === "num").map((node) => node.textContent);
  assert.equal(numbers.length, 20, "every rendered prompt gets a number");
  assert.deepEqual(numbers, Array.from({ length: 20 }, (_, i) => String(84 + i)));
  assert.ok(numbers.every((value) => Number(value) >= 84 && Number(value) <= 103));
  assert.equal(numbers[numbers.length - 1], "103", "the newest prompt keeps the last turn number");
  // Scrolling to the top of the list asks for the previous page, with no
  // separate control anywhere in it.
  assert.equal(links.filter((node) => node.className === "question-more").length, 0, "no load-more button exists");
  const beforeGrow = links.length;
  const beforeNums = created.length;
  list.scrollTop = 0;
  list.onScroll();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const grown = links.slice(beforeGrow).filter((node) => node.className === "question-link");
  assert.equal(grown.length, 40, "asking for older prompts re-renders the wider window");
  const grownNumbers = created.slice(beforeNums)
    .filter((node) => node.className === "num").map((node) => node.textContent);
  assert.equal(grownNumbers[grownNumbers.length - 1], "103", "newer numbers stay stable while paging");
  const oldestGrown = Number(grownNumbers[0]);
  assert.ok(oldestGrown < 84, "the grown page reaches older turns: " + oldestGrown);
  // A prompt outside the rendered transcript widens it exactly once.
  await grown[0].onclick();
  assert.equal(loaded && loaded.showAll, true, "an off-screen prompt asks for the full transcript");
  assert.equal(scrolledTo, grown[0].children[1].textContent, "the widened transcript still lands on the tapped prompt");
  const widenCalls = widened;
  // A prompt already in the transcript jumps without another fetch.
  scrolledTo = null;
  await questionLinks[0].onclick();
  assert.equal(scrolledTo, "prompt 84", "the jump lands on the exact turn that was tapped");
  assert.equal(widened, widenCalls, "a target already in the transcript needs no reload");
  loaded = null;
  scrolledTo = null;
  context.sessionId = "session-b";
  await questionLinks[0].onclick();
  assert.equal(loaded, null, "an old question link cannot load its cache into another session");
  assert.equal(scrolledTo, null);
});

test("phone question navigator is a compact session-bar control", () => {
  assert.match(html, /<div id="sessbar"[^>]*>[\s\S]*?<button id="questionbtn"/);
  assert.ok(html.indexOf('id="questionbtn"') < html.indexOf('<div id="msgwrap">'));
  const buttonRule = html.match(/#questionbtn\s*\{([^}]*)\}/);
  assert.ok(buttonRule);
  assert.doesNotMatch(buttonRule[1], /position\s*:\s*absolute/);
  assert.match(html, /#questionbtn\.show\s*\{\s*display:grid;\s*\}/);
  assert.match(script, /questionbtn"\)\.onclick = \(event\) => \{ event\.stopPropagation\(\); openQuestionPanel\(\); \};/);
});

function rewindDomFixture() {
  const actions = [];
  const host = {
    child: null,
    replaceChild(next, current) {
      assert.equal(current, bubble);
      this.child = next;
      next.parentNode = this;
    },
  };
  const bubble = { dataset: {}, parentNode: host };
  const createElement = (tag) => {
    const node = {
      children: [],
      appendChild(child) { this.children.push(child); child.parentNode = this; },
      setAttribute(name, value) { this[name] = value; },
    };
    if (tag === "button") actions.push(node);
    return node;
  };
  return { actions, bubble, createElement, host };
}

test("history exposes a dedicated rewind action instead of whole-bubble taps", () => {
  const start = script.indexOf("function attachUserRewind(");
  const end = script.indexOf("async function rewindTo(", start);
  assert.ok(start >= 0 && end > start);
  const { actions, bubble, createElement, host } = rewindDomFixture();
  const rewinds = [];
  const context = vm.createContext({
    window: { matchMedia: () => ({ matches: true }) },
    document: { createElement },
    addMsg: () => bubble,
    rewindTo: (...args) => rewinds.push(args),
  });
  vm.runInContext(script.slice(start, end), context);
  context.addUserBubble("keep this preview", 17);
  assert.equal(actions.length, 1);
  assert.equal(host.child.className, "user-msg-row");
  assert.deepEqual(host.child.children, [actions[0], bubble]);
  assert.equal(actions[0].textContent, "↶");
  let stopped = false;
  actions[0].onclick({ stopPropagation: () => { stopped = true; } });
  assert.equal(stopped, true);
  assert.deepEqual(rewinds, [[17, "keep this preview"]]);
  rewinds.length = 0;
  assert.equal(typeof bubble.onclick, "undefined", "the bubble is never a rewind trigger");
  assert.equal(actions[0].title, "rewind to before this message");
  actions[0].onclick({ stopPropagation: () => {} });
  assert.deepEqual(rewinds, [[17, "keep this preview"]], "the dedicated button performs the rewind");
});

test("a live user bubble resolves the canonical index before rewinding", async () => {
  const start = script.indexOf("function attachUserRewind(");
  const end = script.indexOf("async function rewindTo(", start);
  assert.ok(start >= 0 && end > start);
  const { actions, bubble, createElement, host } = rewindDomFixture();
  const rewinds = [];
  const notes = [];
  const messages = [
    { role: "user", content: "repeat" },
    { role: "assistant", content: "old" },
    { role: "user", content: "repeat" },
    { role: "assistant", content: "new" },
  ];
  const context = vm.createContext({
    window: { matchMedia: () => ({ matches: true }) },
    document: { createElement },
    addMsg: () => bubble,
    textOf: (value) => String(value),
    api: async () => ({ headers: { get: () => "12" }, json: async () => messages }),
    note: (message) => notes.push(message),
    rewindTo: (...args) => rewinds.push(args),
  });
  vm.runInContext(script.slice(start, end), context);
  vm.runInContext(userOriginHelpers, context);
  context.addLiveUserBubble("repeat", "repeat", "session-1", 10);
  assert.equal(actions.length, 1);
  assert.deepEqual(host.child.children, [actions[0], bubble]);
  await actions[0].onclick({ stopPropagation() {} });
  assert.deepEqual(rewinds, [[10, "repeat", "session-1"]]);
  assert.deepEqual(notes, []);
  assert.match(script, /addLiveUserBubble\(text\.trim\(\) \? text : "\(images only\)", prompt, sid, rewindFloor\);/);
});

test("rewind action is visible on both layouts", () => {
  assert.match(html, /\.rewind-action \{\s*display:inline-flex;/);
  assert.match(html, /@media \(max-width:700px\) \{[\s\S]*?\.user-msg-row > \.rewind-action \{[\s\S]*?display:inline-flex;/);
  assert.match(html, /\.user-msg-row > \.rewind-action \{[\s\S]*?width:34px;[\s\S]*?height:34px;[\s\S]*?border-radius:999px;/);
});

test("a pending side note keeps re-checking the queue until it drains", () => {
  const start = script.indexOf("let sideNotesSeq = 0;");
  const end = script.indexOf("// The queue is server-side", start);
  assert.ok(start >= 0 && end > start);
  const intervals = [];
  const nodes = {
    sidequeue: { hidden: false },
    sidequeuelist: { children: [], appendChild(el) { this.children.push(el); } },
    sidequeuecount: { textContent: "" },
  };
  const context = vm.createContext({
    document: {
      visibilityState: "visible",
      createElement: () => ({ className: "", textContent: "", title: "", appendChild() {} }),
    },
    $: (id) => nodes[id],
    setInterval: (fn, ms) => { intervals.push({ fn, ms, cleared: false }); return intervals.length; },
    clearInterval: (id) => { if (intervals[id - 1]) intervals[id - 1].cleared = true; },
    refreshSideNotes: () => { context.refreshed = (context.refreshed || 0) + 1; },
  });
  vm.runInContext(script.slice(start, end), context);
  context.renderSideNotes([{ from: "user", content: "hello" }]);
  assert.equal(intervals.length, 1);
  assert.equal(intervals[0].ms, 5000);
  context.renderSideNotes([{ from: "user", content: "hello" }]);
  assert.equal(intervals.length, 1, "a second pending render reuses the running watch");
  intervals[0].fn();
  assert.equal(context.refreshed, 1);
  context.renderSideNotes([]);
  assert.equal(intervals[0].cleared, true, "an empty queue stops the watch");
});

test("a mobile side note queues text without stopping the running turn", async () => {
  const start = script.indexOf("async function sendSideNote(");
  const end = script.indexOf("async function stopTurn(", start);
  assert.ok(start >= 0 && end > start);
  const calls = [], notices = [];
  const prompt = { value: "  steer left  ", style: { height: "96px" }, focus() {} };
  const button = { disabled: false };
  const context = vm.createContext({
    sideNoteBusy: false,
    sessionId: "session-a",
    pendingImages: [],
    isTurnRunning: () => true,
    $: (id) => id === "prompt" ? prompt : button,
    setPromptValue: (value) => { prompt.value = value; return true; },
    api: async (path, options) => { calls.push([path, options]); },
    clearDraft: (id) => calls.push(["clearDraft", id]),
    refreshSideNotes() {},
    note: (message) => notices.push(message),
    JSON,
  });
  vm.runInContext(script.slice(start, end), context);
  await context.sendSideNote();
  assert.equal(calls[0][0], "/sessions/session-a/side-notes");
  assert.equal(calls[0][1].method, "POST");
  assert.deepEqual(JSON.parse(calls[0][1].body), { text: "steer left" });
  assert.deepEqual(calls[1], ["clearDraft", "session-a"]);
  assert.equal(prompt.value, "");
  assert.equal(prompt.style.height, "auto");
  assert.equal(notices.at(-1), "side note queued — applies at the next model request");
  assert.equal(button.disabled, false);
});

test("a side note posts immediately despite an expired running-state sample", async () => {
  const start = script.indexOf("async function sendSideNote(");
  const end = script.indexOf("async function stopTurn(", start);
  let polls = 0;
  const posts = [], notices = [];
  let finishPost;
  const response = new Promise((resolve) => { finishPost = resolve; });
  const prompt = { value: "continue", style: {}, focus() {} };
  const button = { disabled: false };
  const context = vm.createContext({
    sideNoteBusy: false, sessionId: "session-a", pendingImages: [],
    isTurnRunning: () => false,
    pollConfirm: () => { polls++; return new Promise(() => {}); },
    $: (id) => id === "prompt" ? prompt : button,
    setPromptValue: (value) => { prompt.value = value; return true; },
    api: (path, options) => { posts.push([path, JSON.parse(options.body)]); return response; },
    clearDraft() {}, refreshSideNotes() {}, note: (message) => notices.push(message), syncStopButton() {}, JSON,
  });
  vm.runInContext(script.slice(start, end), context);
  const sending = context.sendSideNote();
  assert.equal(polls, 0, "confirmation polling must not gate a side-note POST");
  assert.deepEqual(posts, [["/sessions/session-a/side-notes", { text: "continue" }]]);
  assert.equal(button.disabled, true);
  assert.equal(prompt.value, "continue", "keep the draft until the server acknowledges it");
  assert.equal(notices.at(-1), "sending side note…");
  await context.sendSideNote();
  assert.equal(posts.length, 1, "repeated taps must not duplicate the note");
  finishPost();
  await sending;
  assert.equal(prompt.value, "");
  assert.equal(notices.at(-1), "side note queued — applies at the next model request");
  assert.equal(button.disabled, false);
});

test("a failed side note preserves the draft and pending images", async () => {
  const start = script.indexOf("async function sendSideNote(");
  const end = script.indexOf("async function stopTurn(", start);
  let apiCalls = 0;
  const notices = [];
  const prompt = { value: "keep this", style: { height: "80px" }, focus() {} };
  const button = { disabled: false };
  const context = vm.createContext({
    sideNoteBusy: false,
    sessionId: "session-a",
    pendingImages: [],
    isTurnRunning: () => true,
    $: (id) => id === "prompt" ? prompt : button,
    api: async () => { apiCalls++; throw new Error("offline"); },
    clearDraft: () => assert.fail("failed notes must not clear drafts"), syncStopButton() {},
    note: (message) => notices.push(message),
    JSON,
  });
  vm.runInContext(script.slice(start, end), context);
  await context.sendSideNote();
  assert.equal(prompt.value, "keep this");
  assert.equal(notices.at(-1), "side note failed: offline");
  const image = { filename: "photo.png", data_base64: "cGhvdG8=" };
  context.pendingImages.push(image);
  await context.sendSideNote();
  assert.equal(apiCalls, 2);
  assert.equal(prompt.value, "keep this");
  assert.equal(prompt.style.height, "80px");
  assert.deepEqual(Array.from(context.pendingImages), [image]);
  assert.equal(notices.at(-1), "side note failed: offline");
  assert.equal(context.sideNoteBusy, false);
  assert.equal(button.disabled, false);
});

test("image side notes upload screenshots once and accept images without text", async () => {
  const start = script.indexOf("async function sendSideNote(");
  const end = script.indexOf("async function stopTurn(", start);
  for (const draft of ["look here\n[[image:photo.png]]", "look here", ""]) {
    const image = { filename: "photo.png", data_base64: "cGhvdG8=" };
    const prompt = { value: draft, style: { height: "80px" } };
    const button = { disabled: false };
    const posts = [], cleared = [];
    const context = vm.createContext({
      sideNoteBusy: false, sessionId: "session-a", pendingImages: [image],
      $: (id) => id === "prompt" ? prompt : button,
      setPromptValue: (value) => { prompt.value = value; return true; },
      api: async (path, options) => posts.push([path, JSON.parse(options.body)]),
      clearDraft: (id) => cleared.push(id), refreshSideNotes() {}, note() {}, JSON,
    });
    vm.runInContext(script.slice(start, end), context);
    await context.sendSideNote();
    const text = draft ? "look here\n[[image:photo.png]]" : "[[image:photo.png]]";
    assert.deepEqual(posts, [["/sessions/session-a/side-notes", { text, images: [image] }]]);
    assert.equal(prompt.value, "");
    assert.equal(prompt.style.height, "auto");
    assert.equal(context.pendingImages.length, 0);
    assert.deepEqual(cleared, ["session-a"]);
    assert.equal(context.sideNoteBusy, false);
    assert.equal(button.disabled, false);
  }
});

test("a slow image side-note acknowledgement preserves new attachments", async () => {
  const start = script.indexOf("async function sendSideNote(");
  const end = script.indexOf("async function stopTurn(", start);
  let finishPost;
  const response = new Promise((resolve) => { finishPost = resolve; });
  const image = { filename: "photo.png", data_base64: "cGhvdG8=" };
  const nextImage = { filename: "next.png", data_base64: "bmV4dA==" };
  const prompt = { value: "look here", style: { height: "80px" } };
  const button = { disabled: false };
  const posts = [];
  const context = vm.createContext({
    sideNoteBusy: false, sessionId: "session-a", pendingImages: [image],
    $: (id) => id === "prompt" ? prompt : button,
    api: (path, options) => { posts.push(JSON.parse(options.body)); return response; },
    clearDraft: () => assert.fail("new attachments must keep their draft"),
    refreshSideNotes() {}, note() {}, JSON,
  });
  vm.runInContext(script.slice(start, end), context);
  const sending = context.sendSideNote();
  context.pendingImages.push(nextImage);
  finishPost();
  await sending;
  assert.deepEqual(posts, [{ text: "look here\n[[image:photo.png]]", images: [image] }]);
  assert.equal(prompt.value, "look here");
  assert.equal(prompt.style.height, "80px");
  assert.deepEqual(Array.from(context.pendingImages), [image, nextImage]);
});

test("a slow side-note acknowledgement preserves edits and another session's draft", async () => {
  const start = script.indexOf("async function sendSideNote(");
  const end = script.indexOf("async function stopTurn(", start);
  for (const switchSession of [false, true]) {
    let finishPost;
    const response = new Promise((resolve) => { finishPost = resolve; });
    const posts = [];
    const prompt = { value: "original", style: { height: "90px" } };
    const button = { disabled: false };
    const context = vm.createContext({
      sideNoteBusy: false, sessionId: "session-a", pendingImages: [],
      $: (id) => id === "prompt" ? prompt : button,
      api: (path, options) => { posts.push([path, JSON.parse(options.body)]); return response; },
      clearDraft: () => assert.fail("must not clear a changed or different session's draft"),
      note() {}, JSON,
    });
    vm.runInContext(script.slice(start, end), context);
    const sending = context.sendSideNote();
    if (switchSession) context.sessionId = "session-b";
    else prompt.value = "edited while sending";
    const expected = prompt.value;
    finishPost();
    await sending;
    assert.deepEqual(posts, [["/sessions/session-a/side-notes", { text: "original" }]]);
    assert.equal(prompt.value, expected);
    assert.equal(prompt.style.height, "90px");
    assert.equal(button.disabled, false);
    assert.equal(context.sideNoteBusy, false);
  }
});

test("empty side notes are rejected locally without polling or disabling the control", async () => {
  const start = script.indexOf("async function sendSideNote(");
  const end = script.indexOf("async function stopTurn(", start);
  let focused = false;
  const notices = [];
  const prompt = { value: "   ", focus() { focused = true; } };
  const button = { disabled: false };
  const context = vm.createContext({
    sideNoteBusy: false, sessionId: "session-a", pendingImages: [],
    $: (id) => id === "prompt" ? prompt : button,
    api: () => assert.fail("empty notes must not issue requests"),
    note: (message) => notices.push(message),
  });
  vm.runInContext(script.slice(start, end), context);
  await context.sendSideNote();
  assert.equal(notices.at(-1), "type a side note first");
  assert.equal(focused, true);
  assert.equal(button.disabled, false);
  assert.equal(context.sideNoteBusy, false);
});

test("phone session hold targets its own row and scrolling cancels it", () => {
  const start = script.indexOf("let sessionActionTarget = null;");
  const end = script.indexOf("async function deleteSession(", start);
  assert.ok(start >= 0 && end > start);
  let now = 1000, nextTimer = 1;
  const timers = new Map();
  const cleared = [];
  const deleted = [];
  const document = { activeElement: null };
  const elements = Object.fromEntries([
    "sessionactions", "sessionactionstitle", "sessionactionmark", "sessionactionfork", "sessionactiondelete",
    "sessionactioncancel", "sessionactionsscrim", "sessionactionspanel",
  ].map((id) => [id, { hidden: true, textContent: "", focus() { document.activeElement = this; } }]));
  const media = { matches: true };
  const context = vm.createContext({
    document, $: (id) => elements[id],
    window: { matchMedia: () => media },
    Date: { now: () => now },
    setTimeout: (fn, ms) => { const id = nextTimer++; timers.set(id, { fn, ms, at: now + ms }); return id; },
    clearTimeout: (id) => { cleared.push(id); timers.delete(id); },
    sessionTitle: (s) => s.summary,
    toggleSessionMark: () => assert.fail("holding must not mark a session"),
    deleteSession: (id) => deleted.push(id),
  });
  vm.runInContext(script.slice(start, end), context);
  context.wireSessionActions();
  const advance = (ms) => {
    now += ms;
    for (const [id, timer] of [...timers]) {
      if (timer.at <= now) { timers.delete(id); timer.fn(); }
    }
  };
  const row = () => {
    const classes = new Set();
    const marker = { focus() { document.activeElement = this; } };
    return {
      dataset: {}, marker,
      classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name), contains: (name) => classes.has(name) },
      querySelector: (selector) => { assert.equal(selector, ".mk"); return marker; },
    };
  };
  const touch = (x, y) => ({
    pointerType: "touch", isPrimary: true, button: 0, clientX: x, clientY: y,
    target: { closest: () => null },
  });

  const held = row();
  const heldSession = { id: "hi-session", summary: "Hi session" };
  context.bindSessionLongPress(held, heldSession);
  elements.sessionactiondelete.onclick();
  assert.deepEqual(deleted, []);
  held.onpointerdown(touch(10, 20));
  assert.equal(held.classList.contains("holding"), true);
  assert.equal(timers.size, 1);
  assert.equal([...timers.values()][0].ms, 650);
  advance(649);
  assert.equal(elements.sessionactions.hidden, true);
  assert.deepEqual(deleted, []);
  advance(1);
  assert.equal(elements.sessionactions.hidden, false);
  assert.equal(elements.sessionactionstitle.textContent, "Hi session");
  assert.equal(vm.runInContext("sessionActionTarget", context), heldSession);
  assert.equal(vm.runInContext("sessionActionReturnFocus", context), held.marker);
  assert.equal(document.activeElement, elements.sessionactionmark);
  assert.deepEqual(deleted, []);
  assert.equal(context.sessionHoldConsumed(held), true);
  assert.equal(held.classList.contains("holding"), false);
  held.onpointerup();
  assert.equal(elements.sessionactions.hidden, false);
  assert.deepEqual(deleted, []);
  elements.sessionactiondelete.onclick();
  assert.deepEqual(deleted, ["hi-session"]);
  assert.equal(elements.sessionactions.hidden, true);
  assert.equal(document.activeElement, held.marker);

  const scrolled = row();
  const otherSession = { id: "keep-session", summary: "Keep session" };
  context.bindSessionLongPress(scrolled, otherSession);
  scrolled.onpointerdown(touch(10, 20));
  scrolled.onpointermove(touch(10, 40));
  assert.deepEqual(cleared, [2]);
  advance(650);
  assert.equal(timers.size, 0);
  assert.equal(elements.sessionactions.hidden, true);
  assert.equal(scrolled.classList.contains("holding"), false);
  assert.equal(context.sessionHoldConsumed(scrolled), false);
  assert.deepEqual(deleted, ["hi-session"]);

  scrolled.onpointerdown(touch(10, 20));
  scrolled.onpointercancel();
  assert.deepEqual(cleared, [2, 3]);
  advance(650);
  assert.equal(timers.size, 0);
  assert.equal(elements.sessionactions.hidden, true);
  assert.equal(scrolled.classList.contains("holding"), false);
  assert.equal(context.sessionHoldConsumed(scrolled), false);
  assert.deepEqual(deleted, ["hi-session"]);

  scrolled.onpointerdown({ ...touch(10, 20), pointerType: "mouse" });
  scrolled.onpointerdown({ ...touch(10, 20), target: { closest: () => elements.sessionactiondelete } });
  media.matches = false;
  scrolled.onpointerdown(touch(10, 20));
  advance(650);
  assert.equal(timers.size, 0);
  assert.equal(elements.sessionactions.hidden, true);
  assert.deepEqual(deleted, ["hi-session"]);

  media.matches = true;
  scrolled.onpointerdown(touch(10, 20));
  advance(650);
  assert.equal(elements.sessionactions.hidden, false);
  assert.equal(elements.sessionactionstitle.textContent, "Keep session");
  assert.equal(vm.runInContext("sessionActionTarget", context), otherSession);
  assert.equal(vm.runInContext("sessionActionReturnFocus", context), scrolled.marker);
  assert.deepEqual(deleted, ["hi-session"]);
  elements.sessionactiondelete.onclick();
  assert.deepEqual(deleted, ["hi-session", "keep-session"]);
  assert.equal(elements.sessionactions.hidden, true);
  assert.equal(document.activeElement, scrolled.marker);
});

test("targeted session delete preserves the open session", async () => {
  const start = script.indexOf("async function deleteSession(");
  const end = script.indexOf("// Render the drawer", start);
  assert.ok(start >= 0 && end > start);
  const calls = [];
  let sessionsLoaded = 0;
  let historyLoaded = 0;
  const context = vm.createContext({
    lastItems: [{ id: "hi-session", title: "hi" }, { id: "open-session", title: "open" }],
    sessionId: "open-session", pendingNew: null, lastMsgCount: 5, turnRunning: true,
    sessionTitle: (s) => s.title,
    api: async (path, options) => { calls.push([path, options]); },
    loadSessions: async () => { sessionsLoaded += 1; },
    loadHistory: async () => { historyLoaded += 1; },
    clearDraft: (id) => calls.push(["clearDraft", id]),
    stopStaleFollow: () => calls.push(["stop"]),
    hideConfirm: () => calls.push(["hide"]),
    localStorage: { removeItem: (key) => calls.push(["remove", key]) },
    addMsg: () => {},
  });
  vm.runInContext(script.slice(start, end), context);
  await vm.runInContext('deleteSession("hi-session", true)', context);
  assert.equal(calls[0][0], "/sessions/hi-session");
  assert.equal(calls[0][1].method, "DELETE");
  assert.equal(context.sessionId, "open-session");
  assert.equal(sessionsLoaded, 1);
  assert.equal(historyLoaded, 0);
  assert.equal(calls.some(([kind, id]) => kind === "clearDraft" && id === "hi-session"), true);
});

test("screenshot display and inline formulas preserve their exact TeX", () => {
  const { render } = harness();
  const formula = String.raw`0.999\ldots := \lim_{n\to\infty} S_n,\quad S_n = \underbrace{0.99\ldots9}_{n\text{个}} = 1 - 10^{-n}`;
  const result = render("## Definition\n\n$$" + formula + "$$\n\n" + String.raw`$S_n$ converges in $\mathbb{R}$.`);
  const nodes = nodesOf(result);
  assert.deepEqual(nodes.map((n) => n.dataset.tex), [formula, "S_n", String.raw`\mathbb{R}`]);
  assert.equal(nodes[0].classList.contains("display"), true);
  assert.match(result, /<div class="h lv2">Definition<\/div>/);
});

test("standalone display formulas do not inherit paragraph blank-line gaps", () => {
  const { render } = harness();
  const formula = String.raw`G_{\mu\nu}+\Lambda g_{\mu\nu}=\frac{8\pi G}{c^4}T_{\mu\nu}`;
  const result = render("Before\n\n$$\n" + formula + "\n$$\n\nAfter");
  assert.match(result, /^<div class="p">Before<\/div><span class="math display"/);
  assert.match(result, /<\/span><div class="p">After<\/div>$/);
  assert.doesNotMatch(result, /<br><br><span class="math display"|<\/span><br><br>/);
  assert.equal(nodesOf(result)[0].dataset.tex, "\n" + formula + "\n");
});

test("phone display formula spacing stays compact", () => {
  assert.match(html, /@media \(max-width:700px\) \{[\s\S]*?\.msg \{ max-width:94%; \}/);
  assert.match(html, /@media \(max-width:700px\) \{[\s\S]*?\.msg\.md \.math\.display \{ margin:4px 0; padding:0; \}/);
});

test("assistant cards span the row so a collapsed fold stays tappable", () => {
  const rule = html.match(/\.msg\.asst \{[^}]*\}/);
  assert.ok(rule, "the assistant bubble rule is missing");
  assert.match(rule[0], /align-self:stretch/, "a content-sized assistant card leaves unclickable dead space beside a collapsed fold");
});

test("all delimiters and multiline TeX are protected before Markdown", () => {
  const { render } = harness();
  const input = String.raw`$x^2$ \(y_1\) $$a*b*c$$ \[\begin{aligned}
|x| &= 1 \\
y &= \text{a ** b}
\end{aligned}\]`;
  const nodes = nodesOf(render(input));
  assert.equal(nodes.length, 4);
  assert.deepEqual(nodes.map((n) => n.classList.contains("display")), [false, false, true, true]);
  assert.equal(nodes[2].dataset.tex, "a*b*c");
  assert.match(nodes[3].dataset.tex, /\|x\| &= 1/);
  assert.match(nodes[3].dataset.tex, /a \*\* b/);
  assert.doesNotMatch(render(input), /<(?:table|b|i)>/);
});

test("code, links, escaped dollars and currency remain literal", () => {
  const { render } = harness();
  for (const input of [
    "`$x$`", "``$x$``", "```tex\n$$x$$\n```", "~~~\n$x$\n~~~", "```tex\n$x$",
    String.raw`\$x\$ and $5 and $10`, "$5-$10", "unfinished `code $x$",
    "[link](https://example.com/$x$)", "[link](https://example.com/Foo_($x$))", "<https://example.com/$x$>",
  ]) assert.equal(nodesOf(render(input)).length, 0, input);
  assert.match(render("`$x$`"), /<code>\$x\$<\/code>/);
  assert.match(render("[link](https://example.com/$x$)"), /href="https:\/\/example.com\/\$x\$"/);
});

test("ordinary Markdown and HTML safety survive", () => {
  const { render } = harness();
  assert.match(render("**bold** and *italic* and ~~old~~"), /<b>bold<\/b> and <i>italic<\/i> and <s>old<\/s>/);
  assert.match(render("# Heading\n- item"), /<div class="h lv1">Heading<\/div>/);
  assert.match(render("<img src=x onerror=alert(1)>"), /&lt;img/);
  assert.doesNotMatch(render("[bad](javascript:alert(1))"), /<a /);
  const attack = '$\\text{"/><img src=x onerror=alert(1)>}$';
  assert.doesNotMatch(render(attack), /<img/);
  assert.equal(nodesOf(render(attack))[0].textContent, attack);
});

test("emphasis encloses inline code without interpreting its contents", () => {
  const { render } = harness();
  const input = '以后可以直接说：**“用 `play_mahjong_discard` 分析这手牌。”**';
  assert.equal(render(input), '<div class="p">以后可以直接说：<b>“用 <code>play_mahjong_discard</code> 分析这手牌。”</b></div>');
  assert.match(render('**Use `a` and `b`.**'), /<b>Use <code>a<\/code> and <code>b<\/code>\.<\/b>/);
  assert.match(render('**`**$x$** <img>`**'), /<b><code>\*\*\$x\$\*\* &lt;img&gt;<\/code><\/b>/);
  assert.match(render('*`x`* and ~~`y`~~'), /<i><code>x<\/code><\/i> and <s><code>y<\/code><\/s>/);
  const math = render('**Use `a` and $x*y*z$.**');
  assert.equal(nodesOf(math)[0].dataset.tex, 'x*y*z');
  assert.match(math, /<b>Use <code>a<\/code> and <span class="math/);
  assert.match(render('```\n**`literal`**\n```'), /<code>\*\*`literal`\*\*/);
  for (let end = 0; end <= input.length; end++) {
    const partial = render(input.slice(0, end));
    assert.doesNotMatch(partial, /[\uE000\uE001]/);
  }
});

test("code slots cannot collide with text or enter link attributes", () => {
  const { render } = harness();
  const collision = '\uE000code 0\uE001';
  assert.ok(render(collision + ' **`x`**').includes(collision));
  for (const input of ['[link](https://example.com/`x y`)', '![alt](https://example.com/`x`)']) {
    assert.doesNotMatch(render(input), /<(?:a|img) /);
  }
  assert.match(render('[`label`](https://example.com)'), /href="https:\/\/example.com"[^>]*><code>label<\/code><\/a>/);
  assert.doesNotMatch(render('**`<img src=x onerror=alert(1)>`**'), /<img/);
  assert.doesNotMatch(render('[`bad`](javascript:alert(1))'), /<a /);
});

test("complete SVG blocks render in an isolated responsive frame", () => {
  const { render } = harness();
  const svg = '<svg viewBox="0 0 320 160"><defs><linearGradient id="g"><stop stop-color="navy"/></linearGradient></defs><rect width="320" height="160" fill="url( \'#g\' )"/></svg>';
  for (const input of ["```svg\n" + svg + "\n```", svg]) {
    const result = render(input);
    assert.match(result, /class="code-preview code-preview-svg"/);
    assert.match(result, /data-preview-view="render" aria-pressed="true">render<\/button>/);
    assert.match(result, /data-preview-view="source" aria-pressed="false">source<\/button>/);
    assert.match(result, /<iframe sandbox referrerpolicy="no-referrer" loading="lazy"/);
    assert.match(result, /style="aspect-ratio:2\.0000 \/ 1"/);
    assert.match(result, /script-src 'none'/);
    assert.match(result, /&lt;svg viewBox=&quot;0 0 320 160&quot;&gt;/);
    assert.match(result, /<pre class="code-preview-source" hidden><code>&lt;svg/);
    assert.doesNotMatch(result, /<svg\b/);
  }
  assert.match(render("```svg\n" + svg), /<pre data-lang="svg"><code>/);
});

test("preview controls switch between isolated rendering and escaped source", () => {
  const { context } = harness();
  const frame = { hidden: false };
  const source = { hidden: true };
  const classes = new Set();
  const buttons = ["render", "source"].map((view) => ({
    dataset: { previewView: view },
    pressed: view === "render" ? "true" : "false",
    setAttribute(name, value) { if (name === "aria-pressed") this.pressed = value; },
  }));
  const preview = {
    classList: { toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
    querySelector(selector) { return selector === "iframe" ? frame : source; },
    querySelectorAll() { return nodeList(buttons); },
  };
  for (const button of buttons) {
    button.closest = (selector) => (selector === ".code-preview" ? preview : null);
  }
  const click = (button) => context.handlePreviewClick({
    target: { closest: (selector) => (selector === "[data-copy-code]" ? null : button) },
  });

  click(buttons[1]);
  assert.equal(frame.hidden, true);
  assert.equal(source.hidden, false);
  assert.equal(buttons[0].pressed, "false");
  assert.equal(buttons[1].pressed, "true");
  assert.equal(classes.has("show-source"), true);

  click(buttons[0]);
  assert.equal(frame.hidden, false);
  assert.equal(source.hidden, true);
  assert.equal(buttons[0].pressed, "true");
  assert.equal(buttons[1].pressed, "false");
  assert.equal(classes.has("show-source"), false);
});

test("active or externally-referencing SVG falls back to escaped source", () => {
  const { render } = harness();
  for (const svg of [
    '<svg onload="alert(1)"><circle r="5"/></svg>',
    '<svg><script>alert(1)</script></svg>',
    '<svg><image href="https://example.com/track.png"/></svg>',
    '<svg><rect style="fill:url(https://example.com/paint.svg)"/></svg>',
    '<svg><style>@import "https://example.com/x.css"</style></svg>',
  ]) {
    const fenced = render("```svg\n" + svg + "\n```");
    assert.doesNotMatch(fenced, /code-preview/);
    assert.match(fenced, /<pre data-lang="svg"><code>&lt;svg/);
    const raw = render(svg);
    assert.doesNotMatch(raw, /code-preview/);
    assert.match(raw, /&lt;svg/);
  }
});

// The document the frame renders, read back out of the srcdoc attribute.
function frameDoc(markup) {
  const m = /srcdoc="([^"]*)"/.exec(markup);
  if (!m) return "";
  return m[1].replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
}

test("HTML code fences render in an isolated frame and keep the source", () => {
  const { render } = harness();
  const html = '<!DOCTYPE html><html><head><title>Pelican</title></head>' +
    '<body><svg viewBox="0 0 10 10"><circle r="4"/></svg></body></html>';
  const result = render("```html\n" + html + "\n```");
  assert.match(result, /class="code-preview code-preview-html"/);
  assert.match(result, /<iframe sandbox referrerpolicy="no-referrer" loading="lazy"/);
  assert.match(result, /script-src 'none'/);
  assert.match(result, /name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1&quot;/);
  assert.match(result, /<pre class="code-preview-source" hidden><code>&lt;!DOCTYPE html&gt;/);
  // The document itself never enters the chat DOM.
  assert.doesNotMatch(result, /<circle/);
  assert.match(frameDoc(result), /<circle r="4"\/>/);
});

test("HTML previews drop scripts, auto-refresh and network references", () => {
  const { render } = harness();
  const html = [
    "<!doctype html><html><head>",
    '<meta http-equiv="refresh" content="0;url=https://example.com/">',
    "<script>fetch('https://example.com/steal')</script>",
    '</head><body><a href="https://example.com/x">go</a>',
    '<img src="https://example.com/pic.png"></body></html>',
  ].join("");
  const result = render("```html\n" + html + "\n```");
  assert.match(result, /class="code-preview code-preview-html"/);
  assert.match(result, /scripts disabled/);
  const doc = frameDoc(result);
  assert.doesNotMatch(doc, /steal/);
  assert.doesNotMatch(doc, /example\.com/);
  assert.doesNotMatch(doc, /refresh/);
  assert.match(doc, /href="#"/);
  // The source next to the frame stays what the model wrote.
  assert.match(result, /&lt;script&gt;fetch/);
});

test("every code block carries a copy control", () => {
  const { render } = harness();
  const result = render("```rust\nfn main() {}\n```");
  assert.match(result, /class="code-block"/);
  assert.match(result, /data-copy-code/);
  assert.match(result, /<pre data-lang="rust"><code>fn main\(\) \{\}/);
});

test("the copy control copies the source next to it", async () => {
  const { context } = harness();
  const copied = [];
  context.navigator = { clipboard: { writeText: async (text) => { copied.push(text); } } };
  const code = { textContent: "line one\nline two" };
  const host = { querySelector: () => code };
  const button = {
    textContent: "copy", isConnected: true,
    closest: (selector) => selector === "[data-copy-code]" ? button
      : (selector === ".code-preview, .code-block" ? host : null),
  };
  context.handlePreviewClick({
    target: { closest: (selector) => (selector === "[data-copy-code]" ? button : null) },
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(copied, ["line one\nline two"]);
  assert.equal(button.textContent, "copied");
});

test("remote images and media render inline without autoplay", () => {
  const { render } = harness();
  const image = render("![K2](https://example.com/K2_(mountain).jpg)");
  assert.match(image, /class="media-image"/);
  assert.match(image, /<img src="https:\/\/example\.com\/K2_\(mountain\)\.jpg" alt="" loading="lazy" decoding="async" referrerpolicy="no-referrer">/);
  assert.match(image, /class="media-caption">K2<\/span>/);

  const video = render("[5 second clip](https://example.com/sample.mp4?download=1)");
  assert.match(video, /class="media-video"/);
  assert.match(video, /<video controls playsinline preload="none" src="https:\/\/example\.com\/sample\.mp4\?download=1"><\/video>/);
  assert.doesNotMatch(video, /autoplay/);
  assert.match(video, /class="media-caption"[^>]*>5 second clip<\/a>/);

  const audio = render("[generated speech](https://example.com/speech.mp3)");
  assert.match(audio, /class="media-audio"/);
  assert.match(audio, /<audio controls playsinline preload="none" src="https:\/\/example\.com\/speech\.mp3"><\/audio>/);
  assert.doesNotMatch(audio, /autoplay/);

  assert.doesNotMatch(render("![bad](javascript:alert(1))"), /<img/);
  assert.match(render("[page](https://example.com/page)"), /<a href="https:\/\/example\.com\/page"/);
});

test("math inside tables, lists and quotes does not consume Markdown syntax", () => {
  const { render } = harness();
  const table = render("| math | value |\n| --- | --- |\n| $|x|$ | $1$ |");
  assert.match(table, /<table>/);
  assert.equal(nodesOf(table).length, 2);
  assert.equal((table.match(/<td>/g) || []).length, 2);
  assert.equal(nodesOf(render("- $x$\n> $y$\n## $z$")).length, 3);
});

test("every stream prefix is safe and final paint matches a history render", () => {
  const { render } = harness();
  const reply = String.raw`Intro $S_n$.
\[\lim_{n\to\infty} S_n = 1\]
` + "\n```tex\n$x$\n```";
  for (let i = 0; i <= reply.length; i++) {
    assert.doesNotThrow(() => render(reply.slice(0, i)));
    assert.doesNotMatch(render(reply.slice(0, i)), /\uE000math/);
  }
  assert.equal(nodesOf(render("$$x")).length, 0);
  assert.equal(nodesOf(render("$$x$$")).length, 1);
  assert.equal(render(reply), harness().render(reply));
});

test("literal slot-like text cannot alias a generated formula", () => {
  const { render } = harness();
  const literal = "\uE000math 0\uE001";
  const result = render(literal + " $x$");
  assert.ok(result.includes(literal));
  assert.equal(nodesOf(result).length, 1);
});

test("late resources repaint only current nodes and preserve scroll position", async () => {
  const h = harness();
  h.setReply("$old$");
  h.context.decorateMath(h.msgs);
  h.context.decorateMath(h.msgs);
  assert.equal(h.assets.length, 2);
  h.setReply("$$new$$");
  await h.finishLoading();
  assert.deepEqual(h.calls.map((c) => c.tex), ["new"]);
  assert.equal(h.msgs.scrollTop, 0);
  assert.equal(h.calls[0].options.displayMode, true);
  assert.equal(h.calls[0].options.trust, false);
  assert.equal(h.calls[0].options.maxExpand, 1000);
  assert.equal(h.calls[0].options.maxSize, 20);
  h.setReply("$next$");
  h.context.decorateMath(h.msgs);
  assert.equal(h.calls.length, 2);
  assert.equal(h.assets.length, 2);
  h.context.decorateMath(h.msgs);
  assert.equal(h.calls.length, 2);
});

test("failed resources keep escaped source without retrying every stream paint", async () => {
  const h = harness();
  h.setReply("$x$");
  h.context.decorateMath(h.msgs);
  h.assets[0].onerror(new Error("offline"));
  await vm.runInContext("mathLoading", h.context);
  h.context.decorateMath(h.msgs);
  assert.equal(h.assets.length, 2);
  assert.equal(h.msgs.nodes[0].textContent, "$x$");
  assert.equal(h.calls.length, 0);
});

test("invalid TeX falls back per formula without blocking its neighbors", () => {
  const h = harness();
  h.installKatex();
  h.setReply(String.raw`$\badcommand{<img>}$ and $x$`);
  h.context.typesetMath(h.msgs);
  assert.equal(h.msgs.nodes[0].textContent, String.raw`$\badcommand{<img>}$`);
  assert.equal(h.msgs.nodes[1].textContent, "typeset:x");
});

test("a math slot cannot manufacture a link or inject markup into href", () => {
  const { render } = harness();
  for (const input of [
    "[link](https://example.com/$x y$)", "<https://example.com/$x y$>",
    String.raw`[link](https://example.com/$\text{\" onmouseover=\"alert(1)}$)`,
  ]) {
    const result = render(input);
    assert.doesNotMatch(result, /<a /);
    assert.equal(nodesOf(result).length, 1);
  }
});

test("the real streaming paint leaves the formulas for the final render", async () => {
  const h = harness();
  const bubble = {
    set innerHTML(markup) { h.msgs.nodes = nodesOf(markup); },
    classList: { remove() {} },
    querySelector: () => h.msgs.querySelector(),
    querySelectorAll: () => h.msgs.querySelectorAll(),
  };
  Object.assign(h.context, {
    bubble, raw: "$x", paintTimer: null, sid: "s", sessionId: "s",
    activityDirty: false,
    answerDirty: true, progressDirty: false, thinkingDirty: false, thinkingDoneDirty: false,
    mainRate: () => "",
    shouldFollowOutput: () => true, followOutput() {},
  });
  const paint = script.match(/const paint = \(\) => \{([\s\S]*?)\n  \};/)[1];
  const runPaint = () => vm.runInContext("(() => {" + paint + "})()", h.context);
  runPaint();
  assert.equal(h.assets.length, 0);
  h.context.raw = "$x$";
  h.context.answerDirty = true;
  runPaint();
  // The bubble is rebuilt on every paint, so typesetting it on every paint
  // typesets every formula again and again. The finished reply is typeset
  // once, at the end.
  assert.equal(h.assets.length, 0);
  assert.equal(h.msgs.nodes[0].textContent, "$x$");
  h.context.decorateMath(h.msgs);
  assert.equal(h.assets.length, 2);
  await h.finishLoading();
  assert.equal(h.msgs.nodes[0].textContent, "typeset:x");
});

test("a long reply is painted less often than a short one", () => {
  const h = harness();
  assert.equal(h.context.replyPaintDelay(500), 50);
  assert.equal(h.context.replyPaintDelay(5000), 100);
  assert.equal(h.context.replyPaintDelay(50000), 180);
  // The delay never grows without bound: the reply still moves.
  assert.ok(h.context.replyPaintDelay(1000000) <= 200);
});

test("stream paint batches answer, thinking and progress into one follow pass", () => {
  const h = harness();
  let followChecks = 0;
  let follows = 0;
  let progressWrites = 0;
  let thinkingWrites = 0;
  let answerWrites = 0;
  const summaryWrites = [];
  const summary = { set textContent(value) { summaryWrites.push(value); } };
  const think = {
    body: { set textContent(value) { thinkingWrites++; assert.equal(value, "reasoning"); } },
    box: { querySelector: () => summary },
  };
  const bubble = {
    classList: { remove() {} },
    set innerHTML(markup) { answerWrites++; h.msgs.nodes = nodesOf(markup); },
    querySelector: () => null,
    querySelectorAll: () => nodeList(),
  };
  Object.assign(h.context, {
    answerDirty: true, progressDirty: true, thinkingDirty: true, thinkingDoneDirty: true, thinkRate: "", mainRate: () => "",
    activityDirty: false,
    expandProgress: true, prog: {}, progDetails: { open: false, dataset: {} }, renderProgress: () => { progressWrites++; },
    think, thinkRaw: "reasoning", bubble, raw: "answer", paintTimer: 1,
    sid: "s", sessionId: "s",
    shouldFollowOutput: () => { followChecks++; return true; },
    followOutput: (follow) => { follows++; assert.equal(follow, true); },
  });
  const paint = script.match(/const paint = \(\) => \{([\s\S]*?)\n  \};/)[1];
  const runPaint = () => vm.runInContext("(() => {" + paint + "})()", h.context);
  runPaint();
  assert.deepEqual(
    { followChecks, follows, progressWrites, thinkingWrites, answerWrites },
    { followChecks: 1, follows: 1, progressWrites: 1, thinkingWrites: 1, answerWrites: 1 },
  );
  assert.deepEqual(summaryWrites, ["thinking…", "thinking"]);
  assert.equal(h.context.progDetails.open, true);
  runPaint();
  assert.deepEqual(
    { followChecks, follows, progressWrites, thinkingWrites, answerWrites },
    { followChecks: 1, follows: 1, progressWrites: 1, thinkingWrites: 1, answerWrites: 1 },
  );
  assert.deepEqual(summaryWrites, ["thinking…", "thinking"]);
});

test("separate thinking phases create separate folds", () => {
  const start = script.indexOf("function createThinkingFold(");
  const end = script.indexOf("// The three override picks", start);
  assert.ok(start >= 0 && end > start);
  const summary = { textContent: "thinking" };
  const body = { textContent: "" };
  const fold = () => ({ box: { dataset: {}, querySelector: () => summary }, body });
  let creations = 0;
  let insertions = 0;
  const parent = {
    children: [],
    dataset: {},
    classList: { contains: () => false },
    insertBefore(node, before) { insertions++; assert.ok(node); assert.equal(before, reply); },
  };
  const reply = { parentNode: parent };
  const context = vm.createContext({
    addFold(title, text, host) {
      creations++;
      assert.equal(title, "thinking…");
      assert.equal(text, "");
      assert.equal(host, parent);
      summary.textContent = title;
      return fold();
    },
  });
  vm.runInContext(script.slice(start, end), context);
  const first = context.createThinkingFold(parent, reply);
  const second = context.createThinkingFold(parent, reply);
  assert.notEqual(second, first);
  assert.equal(creations, 2);
  assert.equal(insertions, 2);
  assert.equal(summary.textContent, "thinking…");
  assert.equal(body.textContent, "");
});

test("delta blocks defer layout work to the coalesced paint", () => {
  const start = script.indexOf('const { kind, payload } = parseBlock');
  const end = script.indexOf("if (finished) break;", start);
  const blockDispatch = script.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.doesNotMatch(blockDispatch, /shouldFollowOutput|followOutput/);
  assert.match(blockDispatch, /kind === "delta"[\s\S]*?answerDirty = true;[\s\S]*?schedulePaint\(\)/);
  assert.match(blockDispatch, /kind === "thinking_start"[\s\S]*?closeThinkingPhase\(\)/);
  assert.match(blockDispatch, /kind === "thinking"[\s\S]*?if \(text\)[\s\S]*?createThinkingFold\(box, phaseBubble\)[\s\S]*?thinkingDirty = true;[\s\S]*?schedulePaint\(\)/);
  assert.doesNotMatch(blockDispatch, /kind !== "thinking_start" && kind !== "thinking" && kind !== "thinking_done"/);
  assert.match(blockDispatch, /kind === "done"[\s\S]*?closeThinkingPhase\(\)/);
  assert.match(blockDispatch, /kind === "subagent_progress"[\s\S]*?activity.subagents\(payload\)/);
});

test("progress cannot split an active thinking phase", () => {
  const start = script.indexOf('const { kind, payload } = parseBlock');
  const end = script.indexOf("if (finished) break;", start);
  const blockDispatch = script.slice(start, end);
  // Payload-derived speed was removed because the structured `rate` event now
  // carries the display sample.
  const rateBranch = blockDispatch.match(/else if \(kind === "rate"\) \{([\s\S]*?)\n        \}/);
  const subagentBranch = blockDispatch.match(/else if \(kind === "subagent_progress"\) \{([\s\S]*?)\n        \}/);
  assert.ok(rateBranch);
  assert.ok(subagentBranch);
  assert.match(rateBranch[1], /rateSampleOf\(JSON\.parse\(payload\)\)/);
  assert.match(rateBranch[1], /thinkingRates\.speed\(sample\)/);
  assert.match(rateBranch[1], /setTokenRate\(sample\)/);
  assert.doesNotMatch(blockDispatch, /else if \(payload\)/);
  assert.doesNotMatch(rateBranch[1], /closeThinkingPhase/);
  assert.doesNotMatch(subagentBranch[1], /closeThinkingPhase/);
  assert.doesNotMatch(rateBranch[1], /ensureProgressPhase\(\)|progLines\.push/);
  assert.match(subagentBranch[1], /activity.subagents\(payload\)/);
  assert.doesNotMatch(blockDispatch, /closeProgressPhase/);
  assert.doesNotMatch(script, /finalizing response/);
});

test("prompt composition defers reflow, draft writes and keyboard shortcuts", () => {
  const start = script.indexOf('$("prompt").addEventListener("compositionstart"');
  const end = script.indexOf("const toggleSessions", start);
  const promptHandlers = script.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.match(promptHandlers, /compositionstart[\s\S]*?promptComposing = true/);
  assert.match(promptHandlers, /compositionend[\s\S]*?promptComposing = false[\s\S]*?growPrompt\(\)[\s\S]*?persistDraft\(\)/);
  assert.match(promptHandlers, /addEventListener\("input", \(e\)[\s\S]*?if \(promptComposing \|\| e\.isComposing\) \{[\s\S]*?return;/);
  assert.match(promptHandlers, /addEventListener\("keydown", \(e\)[\s\S]*?if \(promptComposing \|\| e\.isComposing \|\| e\.keyCode === 229\) return/);
  // A layout change during a composition cancels the word on Android WebView,
  // so the composer holds its height until the composition ends.
  assert.match(script, /function growPrompt\(\)[\s\S]*?if \(promptComposing\) return;/);
  assert.match(script, /addEventListener\("resize", \(\) => \{[\s\S]*?growPrompt\(\)/);
});

test("media decoration leaves math text and rendered math subtrees alone", () => {
  const h = harness();
  const node = { nodeValue: "plot.svg", parentElement: { closest: (selector) => selector === ".math" } };
  h.context.document.createTreeWalker = () => {
    let pending = true;
    return { currentNode: node, nextNode() { const next = pending; pending = false; return next; } };
  };
  h.context.NodeFilter = { SHOW_TEXT: 4 };
  const start = script.indexOf("function decorateMedia(root)");
  const end = script.indexOf("\n}", start) + 2;
  vm.runInContext("const MAX_PREVIEWS_PER_MSG = 4;\n" + script.slice(start, end), h.context);
  assert.doesNotThrow(() => h.context.decorateMedia(h.msgs));
});

function previewHarness(paths, api) {
  const slots = paths.map((path) => ({
    dataset: { path }, children: [], classList: { add() {} },
    append(...nodes) { this.children.push(...nodes); },
    replaceWith(node) { this.replacement = node; },
  }));
  const subtree = (nodes) => ({
    querySelectorAll(selector) { return nodeList(nodes.filter((n) => !selector.includes(":not") || !n.dataset.held)); },
    contains: (node) => nodes.includes(node),
  });
  const root = subtree(slots);
  const requested = [], revoked = [];
  let serial = 0;
  const context = vm.createContext({
    sessionId: "s", previewUrls: new Map(), phoneFrameUrls: new Map(), $: () => root,
    api: async (url) => { requested.push(url); return api ? api(url) : { ok: true, blob: async () => ({}) }; },
    URL: { createObjectURL: () => "blob:" + (++serial), revokeObjectURL: (url) => revoked.push(url) },
    document: { createElement: (tag) => ({ tag }), createTextNode: (text) => ({ text }) },
  });
  vm.runInContext(script.slice(script.indexOf("async function loadPreviews("), script.indexOf("// A history fold is titled")), context);
  return { context, root, slots, subtree, requested, revoked };
}

test("assistant previews load automatically once and cleanup is scoped to removed rows", async () => {
  const h = previewHarness(["result.png", "other.png"]);
  await h.context.loadPreviews(h.root);
  await h.context.loadPreviews(h.root);
  assert.equal(h.requested.length, 2);
  assert.equal(h.slots[0].children[0].tag, "img");
  assert.equal(h.slots[0].children[0].src, "blob:1");
  h.context.revokePreviews(h.subtree([h.slots[0]]));
  assert.deepEqual(h.revoked, ["blob:1"]);
  assert.equal(h.context.previewUrls.get(h.slots[1]), "blob:2");
  h.context.revokePreviews();
  assert.deepEqual(h.revoked, ["blob:1", "blob:2"]);
  assert.equal(h.context.previewUrls.size, 0);
});

test("a retired preview revokes a late blob without mounting it or fetching later slots", async () => {
  let finishBlob, startBlob;
  const started = new Promise((resolve) => { startBlob = resolve; });
  const h = previewHarness(["result.png", "other.png"], () => ({
    ok: true, blob: () => new Promise((resolve) => { finishBlob = resolve; startBlob(); }),
  }));
  const loading = h.context.loadPreviews(h.root);
  await started;
  h.context.revokePreviews();
  finishBlob({});
  await loading;
  assert.deepEqual(h.revoked, ["blob:1"]);
  assert.equal(h.requested.length, 1);
  assert.equal(h.slots[0].children.length, 0);
  assert.equal(h.context.previewUrls.size, 0);
});

test("a failed retired preview cannot retry its basename in a different session", async () => {
  let fail;
  const h = previewHarness(["dir/result.png"], () => new Promise((_, reject) => { fail = reject; }));
  const loading = h.context.loadPreviews(h.root);
  h.context.revokePreviews();
  h.context.sessionId = "other";
  fail(new Error("not found"));
  await loading;
  assert.deepEqual(h.requested, ["/sessions/s/file?path=dir%2Fresult.png"]);
  assert.equal(h.slots[0].replacement, undefined);
});

function historyHarness({
  delayed = false, existingKeys = [], existingNodes = [], follow = true,
  messages = [{ role: "user", content: "prompt" }, { role: "assistant", content: "reply" }], total = "2",
} = {}) {
  let resolveResponse;
  let apiCalls = 0;
  let clears = 0;
  const retired = [];
  const scrollEvents = [];
  const response = {
    headers: { get: () => total },
    json: async () => messages,
  };
  const pending = delayed ? new Promise((resolve) => { resolveResponse = () => resolve(response); }) : Promise.resolve(response);
  const root = {
    children: [], scrollTop: 0, scrollHeight: 100,
    querySelectorAll() { return []; },
    insertBefore(node, anchor) {
      if (node.remove) node.remove();
      // Real DOM inserts a fragment's children, not the fragment itself.
      if (node && node.fragment) {
        for (const child of node.children.slice()) this.insertBefore(child, anchor);
        node.children = [];
        return node;
      }
      const index = anchor === null ? this.children.length : this.children.indexOf(anchor);
      assert.ok(index >= 0, "the insertion anchor must be a child");
      this.children.splice(index, 0, node);
      node.parent = this;
      Object.defineProperty(node, "previousElementSibling", {
        configurable: true,
        get() { return root.children[root.children.indexOf(this) - 1] || null; },
      });
      return node;
    },
    appendChild(node) { return this.insertBefore(node, null); },
    get lastElementChild() { return this.children[this.children.length - 1]; },
    set textContent(value) { assert.equal(value, ""); clears += 1; this.children = []; },
  };
  for (const node of existingNodes) { node.children ||= []; node.dataset ||= {}; root.appendChild(node); }
  for (const node of root.children) node.remove = function remove() {
    const i = root.children.indexOf(this);
    if (i >= 0) root.children.splice(i, 1);
  };
  const rows = [
    { key: "old", text: "prompt", role: "user", historyIndex: 0 },
    { key: "new", text: "reply", role: "assistant", historyIndex: 1 },
  ];
  const context = vm.createContext({
    sessionId: "s", historySession: existingKeys.length ? "s" : "", historyKeys: existingKeys.slice(),
    lastMsgCount: null, staleKey: "", staleDone: false, previewUrls: new Map(), turns: new Map(),
    pendingUserInputs: new Map(), historyLoadSeq: 0, staleBox: null,
    recoveredView: null, eventSession: "", armHistoryFollow() {},
    AbortController, setTimeout, clearTimeout,
    textOf: (content) => content, isRuntimeSyntheticUser: (message) => !!message.synthetic,
    // loadHistory builds its failure row through the real DOM helpers.
    document: {
      createElement: () => ({
        children: [], dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
        appendChild(node) { this.children.push(node); return node; },
        remove() {}, querySelector: () => null, querySelectorAll: () => [],
      }),
      createDocumentFragment: () => ({
        children: [], fragment: true,
        appendChild(node) { this.children.push(node); return node; },
      }),
    },
    $: () => root,
    api: async () => { apiCalls += 1; return pending; },
    // Phone layout: no desktop owner state absorbs the rebuild's clamp.
    window: { matchMedia: () => ({ matches: false }) },
    resetTranscriptScroll() { scrollEvents.push("reset"); },
    historyRowsOf: () => ({ rows, shownCount: rows.length }),
    histKey: () => "tail", histEndsTurn: () => true, historyEndsUnanswered: () => true,
    syncStopButton() {}, renderSessionBar() {}, attachLiveBox() {}, showTurnRate() {},
    updateQuestionButton() {},
    shouldFollowOutput: () => follow,
    followOutput(wasFollowing) {
      scrollEvents.push("follow");
      if (wasFollowing) root.scrollTop = root.scrollHeight;
    },
    revokePreviews(node) { retired.push(node); },
    renderHistoryRow(row, parent) {
      const node = {
        children: [],
        key: row.key, parent, dataset: { canonicalIndex: String(row.historyIndex ?? rows.indexOf(row)), canonicalKey: row.key },
        classList: { contains: (name) => row.role === "user" && name === "user-msg-row" },
      };
      node.remove = function remove() {
        const i = root.children.indexOf(this);
        if (i >= 0) root.children.splice(i, 1);
      };
      parent.appendChild(node);
    },
    // A swallowed render error would otherwise show up as an empty transcript.
    addMsg(kind, text) { if (kind === "err") throw new Error(String(text)); },
  });
  const start = script.indexOf("async function loadHistory(");
  const end = script.indexOf("// Historical user inputs", start);
  vm.runInContext(historyRetrySrc, context);
  vm.runInContext(script.slice(start, end), context);
  vm.runInContext(script.slice(script.indexOf("function attachLiveBox("), script.indexOf("// A turn the user has left")), context);
  return {
    context, root, rows, resolveResponse,
    apiCalls: () => apiCalls, clears: () => clears, revokes: () => retired.length, retired,
    scrollEvents: () => scrollEvents.slice(),
  };
}

test("history stays mounted while a refresh is in flight", async () => {
  const old = { key: "visible-old" };
  const h = historyHarness({ delayed: true, existingNodes: [old] });
  const loading = h.context.loadHistory();
  await Promise.resolve();
  assert.deepEqual(h.root.children, [old]);
  assert.equal(h.clears(), 0);
  h.resolveResponse();
  await loading;
  assert.equal(h.apiCalls(), 1);
  assert.equal(h.clears(), 1);
  assert.deepEqual(renderedKeys(h.root), ["old", "new"]);
});

test("a phone switch resets the scroll baseline before following the tail", async () => {
  const h = historyHarness({ delayed: true, follow: true });
  h.root.scrollHeight = 900;
  const loading = h.context.loadHistory();
  h.resolveResponse();
  await loading;
  // The rebuilt transcript clamps the old offset, and that queued scroll event
  // would read as a reader scroll-up and veto the follow without the reset.
  assert.deepEqual(h.scrollEvents().slice(-2), ["reset", "follow"]);
  assert.equal(h.root.scrollTop, h.root.scrollHeight);
});

test("stale history appends canonical progress without replacing its prefix", async () => {
  const old = { key: "mounted-old" };
  const liveUser = { key: "temporary-user" };
  const staleBox = { key: "temporary-live-box" };
  const h = historyHarness({ existingKeys: ["old"], existingNodes: [old, liveUser, staleBox] });
  await h.context.loadHistory({ messages: [], histTotal: 2, appendOnly: true });
  const firstProgress = h.root.children[1];
  assert.equal(h.apiCalls(), 0);
  assert.equal(h.clears(), 0);
  assert.deepEqual(h.retired, [staleBox, liveUser]);
  assert.equal(h.root.children[0], old);
  assert.deepEqual(renderedKeys(h.root), ["mounted-old", "new"]);
  h.rows.push({ key: "newer", text: "later progress" });
  await h.context.loadHistory({ messages: [], histTotal: 3, appendOnly: true });
  assert.equal(h.clears(), 0);
  assert.equal(h.root.children[0], old);
  assert.equal(h.root.children[1], firstProgress);
  assert.deepEqual(renderedKeys(h.root), ["mounted-old", "new", "newer"]);
});

test("a settled input whose recorded floor drifted is reconciled, not duplicated", async () => {
  // The page can believe the session is longer than the store (another client
  // rewound it), so the floor captured at send time sits past the message.
  // A settled input is certainly persisted: leaving its local row in the
  // transcript showed the same question twice, with the copy pinned to the end.
  const row = { key: "pending-user", remove() {} };
  const h = historyHarness({ messages: [
    { role: "user", content: "定一个中午11点的闹钟" },
    { role: "assistant", content: "reply" },
  ] });
  h.context.pendingUserInputs.set("s", [
    { row, prompt: "定一个中午11点的闹钟", minIndex: 9, settled: true },
  ]);
  await h.context.loadHistory();
  assert.equal(h.context.pendingUserInputs.has("s"), false, "the input leaves the pending list");
  assert.equal(h.retired.includes(row), true, "the acknowledged local row is dropped");
});

test("an in-flight input never claims an older identical message", async () => {
  // Until the turn's POST returns, an equal text in history can only be an
  // earlier input: dropping the local row there is the "my message vanished"
  // regression.
  const row = { key: "pending-user", remove() {} };
  const h = historyHarness({ messages: [
    { role: "user", content: "定一个中午11点的闹钟" },
    { role: "assistant", content: "reply" },
  ] });
  h.context.pendingUserInputs.set("s", [
    { row, prompt: "定一个中午11点的闹钟", minIndex: 9, settled: false },
  ]);
  await h.context.loadHistory();
  assert.equal(h.context.pendingUserInputs.has("s"), true, "it stays pending");
  assert.equal(renderedNodes(h.root).includes(row), true, "its local row stays in the transcript");
});

test("history retains image nodes and URLs across repeated progress and normal refresh", async () => {
  const old = { key: "mounted-old" };
  const h = historyHarness({ existingKeys: ["old"], existingNodes: [old] });
  h.context.previewUrls.set(old, "blob:old-preview");
  await h.context.loadHistory({ messages: [], histTotal: 2, appendOnly: true });
  const reply = h.root.children[1];
  h.rows.push({ key: "newer", text: "more progress" });
  await h.context.loadHistory({ messages: [], histTotal: 3 });
  assert.equal(h.clears(), 0);
  assert.equal(h.revokes(), 0);
  assert.equal(h.root.children[0], old);
  assert.equal(h.root.children[1], reply);
  assert.equal(h.context.previewUrls.get(old), "blob:old-preview");
});

test("a sliding history window retires only rows that left the window", async () => {
  const first = { key: "first" }, image = { key: "image" };
  const h = historyHarness({ existingKeys: ["first", "image"], existingNodes: [first, image] });
  h.rows.splice(0, h.rows.length, { key: "image" }, { key: "new" });
  await h.context.loadHistory({ messages: [], histTotal: 201, appendOnly: true });
  assert.equal(h.clears(), 0);
  assert.equal(h.root.children[0], image);
  assert.deepEqual(h.retired, [first]);
});

test("a changed canonical row rebuilds safely without retiring a running box", async () => {
  const old = { key: "old" }, live = { key: "running" };
  const h = historyHarness({ existingKeys: ["different"], existingNodes: [old, live] });
  h.context.turns.set("s", { box: live });
  let liveWasDetached = false;
  h.context.revokePreviews = (root) => { liveWasDetached = !root.children.includes(live); };
  h.context.attachLiveBox = () => h.root.appendChild(live);
  await h.context.loadHistory({ messages: [], histTotal: 2 });
  assert.equal(h.clears(), 1);
  assert.equal(liveWasDetached, true);
  assert.equal(h.root.lastElementChild, live);
});

for (const prefix of ["old", "different"]) test(`unacknowledged user input survives history refresh (${prefix})`, async () => {
  const old = { key: "old", classList: { contains: (name) => name === "user-msg-row" } };
  const user = { key: "pending-user" }, live = { key: "running" };
  const h = historyHarness({ existingKeys: [prefix], existingNodes: [old, user, live] });
  h.context.pendingUserInputs.set("s", [{ row: user, prompt: "prompt", minIndex: 2 }]);
  h.context.turns.set("s", { box: live });
  const older = [{ role: "user", content: "prompt" }, { role: "assistant", content: "reply" }];
  await h.context.loadHistory({ messages: older, histTotal: 2 });
  assert.deepEqual(renderedKeys(h.root), ["old", "new", "pending-user", "running"]);
  assert.equal(h.root.children[2], user);
  assert.equal(h.root.lastElementChild, live);
  assert.equal(h.retired.includes(user), false, "pending input previews must stay owned by the input");
  h.rows.push({ key: "canonical-user", text: "prompt" });
  await h.context.loadHistory({ messages: [...older, { role: "user", content: "prompt" }], histTotal: 3 });
  assert.equal(renderedNodes(h.root).includes(user), false, "canonical acknowledgement replaces, not duplicates, the pending input");
  assert.equal(renderedKeys(h.root).filter((key) => key === "canonical-user").length, 1);
  assert.equal(h.context.pendingUserInputs.has("s"), false);
});

for (const caption of ["caption", ""]) for (const ocr of [false, true]) test(`persisted image input replaces its pending bubble (${caption || "image only"}, OCR=${ocr})`, async () => {
  const user = { key: "pending-user" };
  const h = historyHarness({ existingNodes: [user] });
  vm.runInContext(script.slice(script.indexOf("function textOf("), script.indexOf("// Persisted runtime origin sidecar")), h.context);
  h.context.pendingUserInputs.set("s", [{ row: user, prompt: caption + "\n[[image:p.png]]", minIndex: 0 }]);
  const content = ocr ? caption + "\n\n[Attached Image Content via ocr]\nrecognized text" : [
    { type: "reference", kind: "image", name: "p.png", asset: "attachments/p.png" },
    { type: "text", text: caption },
  ];
  h.rows.splice(0, h.rows.length, { key: "canonical-user" }, { key: "reply" });
  const history = { messages: [{ role: "user", content }, { role: "assistant", content: "reply" }], histTotal: 2 };
  await h.context.loadHistory(history);
  assert.equal(h.context.pendingUserInputs.has("s"), false);
  assert.deepEqual(renderedKeys(h.root), ["canonical-user", "reply"]);
  await h.context.loadHistory(history);
  assert.deepEqual(renderedKeys(h.root), ["canonical-user", "reply"]);
});

for (const prefix of ["old", "different"]) test(`history refresh retains the active recovered stream (${prefix})`, async () => {
  const old = { key: "old" }, live = { key: "recovered" };
  const h = historyHarness({ existingKeys: [prefix], existingNodes: [old, live] });
  h.context.eventSession = "s";
  h.context.recoveredView = { box: live };
  let retiredLive = false;
  h.context.revokePreviews = (node) => { retiredLive ||= node === live || !!node.children?.includes(live); };
  await h.context.loadHistory({ messages: [], histTotal: 2 });
  assert.equal(h.root.lastElementChild, live, "the stream must keep painting a mounted view");
  assert.equal(retiredLive, false, "stream previews must survive canonical history refresh");
  assert.equal(renderedNodes(h.root).filter((node) => node === live).length, 1);
});

test("an older in-flight history response cannot overwrite a newer snapshot", async () => {
  const h = historyHarness();
  const responses = [];
  h.context.api = () => new Promise((resolve) => responses.push(resolve));
  h.context.historyRowsOf = (messages) => ({ rows: messages.map((m) => ({ key: m.content })), shownCount: messages.length });
  const old = h.context.loadHistory(), fresh = h.context.loadHistory();
  const respond = (text) => ({ headers: { get: () => "1" }, json: async () => [{ role: "assistant", content: text }] });
  responses[1](respond("new"));
  await fresh;
  responses[0](respond("old"));
  await old;
  assert.deepEqual(renderedKeys(h.root), ["new"]);
});

for (const change of ["refresh", "turn", "session", "stream"]) test(`stale history fetch cannot outlive a newer ${change}`, async () => {
  const oldRow = { key: "mounted" };
  const h = historyHarness({ existingNodes: [oldRow] }), c = h.context;
  let respond;
  Object.assign(c, {
    staleSession: "s", staleTries: 0, STALE_FOLLOW_TRIES: 60, staleBusy: false,
    staleDone: false, eventActive: false, runEpoch: 0, document: { visibilityState: "visible" },
    pollConfirm: async () => true, stopStaleFollow() {},
    api: () => new Promise((resolve) => { respond = resolve; }),
    histKey: JSON.stringify,
    historyRowsOf: (messages) => ({ rows: messages.map((m) => ({ key: m.content })), shownCount: messages.length }),
  });
  vm.runInContext(script.slice(script.indexOf("async function reloadStale()"), script.indexOf("// A restart can end a turn")), c);
  const loading = c.reloadStale();
  await new Promise(setImmediate);
  if (change === "refresh") await c.loadHistory({ messages: [{ role: "assistant", content: "new" }], histTotal: 1 });
  if (change === "turn") c.runEpoch++;
  if (change === "session") { c.sessionId = "other"; c.staleSession = "other"; }
  if (change === "stream") c.eventActive = true;
  const expected = renderedNodes(h.root);
  respond({ headers: { get: () => "1" }, json: async () => [{ role: "assistant", content: "stale" }] });
  await loading;
  assert.deepEqual(h.root.children, expected);
});

test("user image references stay visible as placeholders; assistant media stays Markdown", () => {
  const h = harness();
  vm.runInContext(script.slice(script.indexOf("function textOf("), script.indexOf("const PHONE_LAYOUT_QUERY")), h.context);
  const rendered = [];
  Object.assign(h.context, {
    callLabel: () => "tool",
    sessionId: "session-a",
    addUserBubble: (text, index) => rendered.push({ user: text, index }),
    addMsgMd: (text) => rendered.push({ assistant: h.render(text) }),
  });
  const start = script.indexOf("function historyRowsOf(");
  const end = script.indexOf("function attachLiveBox(", start);
  vm.runInContext(script.slice(start, end), h.context);
  const messages = [
    { role: "user", content: [{ type: "reference", kind: "image", name: "photo.png", asset: "opaque-key" }] },
    { role: "user", content: [{ type: "text", text: "caption" }, { type: "image_url", image_url: { url: "data:image/png;base64,PRIVATE" } }] },
    { role: "assistant", content: "![result](https://example.com/result.png)" },
  ];
  const history = h.context.historyRowsOf(messages, 12, false);
  assert.equal(history.rows.length, 3);
  // The page wraps rows in a per-turn container, so the renderer always
  // receives a parent; a bare root keeps the placeholder assertions local.
  const root = { children: [], appendChild(child) { this.children.push(child); } };
  history.rows.forEach((row) => h.context.renderHistoryRow(row, root));
  assert.equal(rendered[0].user, "[image: photo.png]");
  assert.equal(rendered[0].index, 9);
  assert.equal(rendered[1].user, "caption\n[image]");
  assert.doesNotMatch(JSON.stringify(rendered), /PRIVATE|opaque-key/);
  assert.match(rendered[2].assistant, /<img src="https:\/\/example.com\/result.png"/);
  assert.equal(h.context.textOf(messages[0].content), "", "non-display consumers keep their text-only contract");
  assert.notEqual(history.rows[0].key, h.context.historyRowsOf(messages, 13, false).rows[0].key);
});

test("image reread followups stay internal without losing real prompts or canonical indices", () => {
  const h = harness(), rendered = [];
  const chunks = [
    ["function textOf(", "const PHONE_LAYOUT_QUERY"],
    ["function historyRowsOf(", "function attachLiveBox("],
    ["function questionRowsOf(", "async function openQuestionPanel("],
    ["function findUserHistoryIndex(", "function addLiveUserBubble("],
  ];
  for (const [start, end] of chunks) vm.runInContext(script.slice(script.indexOf(start), script.indexOf(end)), h.context);
  Object.assign(h.context, {
    sessionId: "session-a",
    taskReportOf: () => null,
    addUserBubble: (text, index) => rendered.push({ role: "user", text, index }),
    addMsgMd: (text) => rendered.push({ role: "assistant", text }),
    addFold: (label, text) => {
      rendered.push({ role: "tool", label, text });
      return { box: { querySelector: () => null, dataset: {} } };
    },
    FOLD_ROLE_LABELS: {},
  });
  const content = [{ type: "reference", kind: "image", name: "hand.jpg", asset: "attachments/hand.jpg" }, { type: "text", text: "which tile?" }];
  const origin = "runtime-origin:synthetic-user:v1";
  const messages = [
    { role: "user", content },
    { role: "tool", content: "Image file detected" },
    { role: "user", content, reasoning_content: origin },
    { role: "assistant", content: "![result](https://example.test/result.png)" },
    { role: "user", content },
  ];
  const before = JSON.stringify(messages);
  const history = h.context.historyRowsOf(messages, 20, false);
  assert.equal(history.shownCount, 5, "hidden messages still occupy canonical slots");
  assert.deepEqual(Array.from(history.rows, (r) => r.historyIndex), [15, 16, 18, 19]);
  const root = { children: [], appendChild(child) { this.children.push(child); } };
  history.rows.forEach((row) => h.context.renderHistoryRow(row, root));
  assert.deepEqual(rendered.map((r) => r.role), ["user", "tool", "assistant", "user"]);
  assert.equal(rendered[0].text, "[image: hand.jpg]\nwhich tile?");
  assert.equal(rendered[2].text, messages[3].content, "assistant images remain Markdown");
  assert.deepEqual(Array.from(h.context.questionRowsOf(messages, 20), (r) => r.historyIndex), [15, 19]);
  assert.equal(h.context.findUserHistoryIndex(messages.slice(0, 3), 18, "which tile?", 0), 15);
  assert.equal(h.context.findUserHistoryIndex(messages.slice(0, 3), 18, "which tile?", 16), null);
  assert.equal(h.context.findUserHistoryIndex(messages, 20, "which tile?", 0), 19, "genuine repeated prompts remain selectable");
  assert.equal(JSON.stringify(messages), before, "projection must not mutate canonical messages");
  const windowed = Array.from({ length: 202 }, () => ({ role: "assistant", content: "reply" }));
  windowed[2] = messages[2];
  windowed[3] = messages[0];
  const tail = h.context.historyRowsOf(windowed, 302, false);
  // Internal rows are identified by runtime origin metadata only; image-bearing
  // prompts stay visible, so nothing is filtered out of this window.
  assert.equal(tail.shownCount, 202);
  assert.equal(tail.rows[0].historyIndex, 100, "window indices stay absolute when nothing is filtered");
});

test("synthetic origin is exact metadata, never a content or image heuristic", () => {
  const context = vm.createContext({});
  vm.runInContext(userOriginHelpers, context);
  const rust = fs.readFileSync(path.join(__dirname, "../history/types.rs"), "utf8");
  const origin = rust.match(/const RUNTIME_SYNTHETIC_USER_ORIGIN: &str = "([^"]+)";/)[1];
  assert.equal(context.isRuntimeSyntheticUser({ role: "user", content: [], reasoning_content: origin }), true);
  for (const message of [
    { role: "user", content: origin },
    { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AA==" } }] },
    { role: "user", content: "repeat", reasoning_content: "other metadata" },
    { role: "assistant", content: "reply", reasoning_content: origin },
  ]) assert.equal(context.isRuntimeSyntheticUser(message), false);
});

test("a stale live box is not reattached beside canonical history", () => {
  const box = {};
  const root = { appended: [], appendChild(node) { this.appended.push(node); }, scrollTop: 0, scrollHeight: 10 };
  const context = vm.createContext({
    sessionId: "s", staleBox: box, turns: new Map([["s", { box }]]),
    $: () => root, armHistoryFollow() {}, pendingUserInputs: new Map(),
    shouldFollowOutput: () => true, followOutput() {},
  });
  const start = script.indexOf("function attachLiveBox(");
  const end = script.indexOf("// A turn the user has left", start);
  vm.runInContext(script.slice(start, end), context);
  context.attachLiveBox();
  assert.deepEqual(root.appended, []);
});

test("unmatched pending user rows precede trailing canonical answers", () => {
  const root = {
    children: [], scrollTop: 0, scrollHeight: 10,
    get lastElementChild() { return this.children[this.children.length - 1] || null; },
    insertBefore(node, anchor) {
      const oldIndex = this.children.indexOf(node);
      if (oldIndex >= 0) this.children.splice(oldIndex, 1);
      const index = anchor === null ? this.children.length : this.children.indexOf(anchor);
      assert.ok(index >= 0, "the insertion anchor must be a child");
      this.children.splice(index, 0, node);
      return node;
    },
    appendChild(node) { return this.insertBefore(node, null); },
  };
  const row = (key, user = false, index) => ({
    key,
    dataset: index === undefined ? {} : { canonicalIndex: String(index), canonicalKey: key },
    classList: { contains: (name) => user && name === "user-msg-row" },
    get previousElementSibling() { return root.children[root.children.indexOf(this) - 1] || null; },
  });
  const canonicalUser = row("canonical user", true, 0), answer = row("assistant answer", false, 2);
  const pendingUser = row("pending user", true), box = row("live box");
  root.children = [canonicalUser, answer];
  const context = vm.createContext({
    sessionId: "s", staleBox: null, turns: new Map([["s", { box }]]),
    eventSession: "", recoveredView: null,
    $: () => root, armHistoryFollow() {},
    pendingUserInputs: new Map([["s", [{ row: pendingUser, prompt: "unmatched prompt", minIndex: 1, beforeKey: answer.key }]]]),
    shouldFollowOutput: () => true, followOutput() {},
  });
  const start = script.indexOf("function attachLiveBox(");
  const end = script.indexOf("// A turn the user has left", start);
  vm.runInContext(script.slice(start, end), context);
  context.attachLiveBox();
  assert.deepEqual(root.children, [canonicalUser, pendingUser, answer, box]);

  root.children = [canonicalUser];
  context.turns.clear();
  context.attachLiveBox();
  assert.deepEqual(root.children, [canonicalUser, pendingUser], "without an answer the pending prompt appends");

  root.children = [answer];
  context.attachLiveBox();
  assert.deepEqual(root.children, [pendingUser, answer], "a window with no user rows still puts the prompt first");

  const secondPending = row("second pending user", true);
  root.children = [canonicalUser, answer];
  context.pendingUserInputs.set("s", [
    { row: pendingUser, minIndex: 1, beforeKey: answer.key },
    { row: secondPending, minIndex: 1, beforeKey: answer.key },
  ]);
  context.turns.set("s", { box });
  context.attachLiveBox();
  assert.deepEqual(root.children, [canonicalUser, pendingUser, secondPending, answer, box], "pending prompts keep their input order");
});

function indexedHistoryHarness(options = {}) {
  const h = historyHarness(options);
  const append = (role, text, parent = h.root) => {
    const node = {
      key: text, children: [], dataset: {},
      classList: { contains: (name) => role === "user" && name === "user-msg-row" },
      remove() {
        const index = h.root.children.indexOf(this);
        if (index >= 0) h.root.children.splice(index, 1);
      },
    };
    parent.appendChild(node);
    return node;
  };
  Object.assign(h.context, {
    addUserBubble: (text, index, parent) => append("user", text, parent),
    addMsgMd: (text, parent) => append("assistant", text, parent),
    // Cards carry their <details>; this stub has none, so a stored fold cannot
    // be restored through it and the row renderer must cope with that.
    addFold: (label, text, parent) => ({ ...append("tool", text, parent), box: { querySelector: () => null, dataset: {} } }),
    FOLD_ROLE_LABELS: {},
  });
  const start = script.indexOf("function historyRowsOf(");
  const end = script.indexOf("// A session the server reports", start);
  vm.runInContext(script.slice(start, end), h.context);
  // Record what the row renderer was asked to draw: the transcript's own
  // labels are internal to the page, but "this row reached the renderer" is
  // the guarantee these tests are about.
  const rendered = [];
  const renderRow = h.context.renderHistoryRow;
  h.context.renderHistoryRow = (row, parent) => {
    rendered.push(row.message?.content ?? row.text);
    return renderRow(row, parent);
  };
  h.rendered = rendered;
  return h;
}

test("session history renders stored null tool arguments without losing later rows", async () => {
  const messages = [
    { role: "assistant", content: null, tool_calls: [{
      id: "call-1", type: "function", function: { name: "read_file", arguments: "null" },
    }] },
    { role: "tool", tool_call_id: "call-1", content: "Invalid arguments: expected an object" },
    { role: "assistant", content: "Later reply" },
  ];
  const old = { key: "A transcript", dataset: {} };
  const h = indexedHistoryHarness({ messages, total: "2", existingKeys: ["A"], existingNodes: [old] });
  const labels = [], errors = [], addFold = h.context.addFold;
  Object.assign(h.context, {
    sessionId: "B", historySession: "A", abortEvents() {},
    console: { error: (...args) => errors.push(args) },
    addMsg: (role, text) => h.context.addMsgMd(text, h.root),
    addFold: (label, ...args) => { labels.push(label); return addFold(label, ...args); },
  });
  for (const [start, end] of [
    ["function textOf(", "const PHONE_LAYOUT_QUERY"],
    ["function toolArgsOf(", "function addFold("],
  ]) vm.runInContext(script.slice(script.indexOf(start), script.indexOf(end)), h.context);
  await h.context.loadHistory();
  assert.equal(h.context.historySession, "B");
  assert.deepEqual(renderedKeys(h.root), ["Invalid arguments: expected an object", "Later reply"]);
  assert.deepEqual(labels, ["read_file"]);
  assert.deepEqual(errors, []);
});

test("prefetched history from another session cannot replace or cancel the selected load", async () => {
  const old = { key: "A transcript", dataset: {} };
  const h = indexedHistoryHarness({
    delayed: true, messages: [{ role: "assistant", content: "B transcript" }], total: "1",
    existingKeys: ["A"], existingNodes: [old],
  });
  Object.assign(h.context, { sessionId: "B", historySession: "A", abortEvents() {} });
  const loading = h.context.loadHistory();
  await h.context.loadHistory({ sessionId: "A", messages: [{ role: "assistant", content: "Stale A transcript" }], histTotal: 1 });
  const sequenceAfterPrefetch = h.context.historyLoadSeq, rowsAfterPrefetch = renderedNodes(h.root);
  h.resolveResponse();
  await loading;
  assert.equal(sequenceAfterPrefetch, 1, "reject stale cached rows before invalidating B's pending response");
  assert.deepEqual(rowsAfterPrefetch, [old]);
  assert.equal(h.context.historySession, "B");
  assert.deepEqual(renderedKeys(h.root), ["B transcript"]);
  assert.equal(h.apiCalls(), 1);
});

for (const stage of ["projection", "row"]) test(`a ${stage} render failure replaces the previous session with a visible error`, async () => {
  const old = { key: "A transcript", dataset: {} };
  const h = indexedHistoryHarness({
    messages: [{ role: "assistant", content: "B transcript" }], total: "1",
    existingKeys: ["A"], existingNodes: [old],
  });
  const failure = new Error("Failed at " + stage), errors = [];
  Object.assign(h.context, {
    sessionId: "B", historySession: "A", abortEvents() {},
    console: { error: (...args) => errors.push(args) },
    addMsg: (role, text) => h.context.addMsgMd(text, h.root),
    [stage === "projection" ? "historyRowsOf" : "renderHistoryRow"]: () => { throw failure; },
  });
  await h.context.loadHistory();
  assert.deepEqual(renderedKeys(h.root), ["history render failed: Failed at " + stage]);
  assert.equal(h.context.historySession, "");
  assert.deepEqual(Array.from(h.context.historyKeys), []);
  assert.equal(h.context.lastMsgCount, null);
  assert.equal(h.context.canonicalHistoryTotal, 0);
  assert.equal(errors.length, 1);
  assert.equal(errors[0][1], failure, "retain the original exception in the Console");
});

test("identical history refresh preserves canonical nodes around a pending input", async () => {
  const user = { key: "pending user", classList: { contains: (name) => name === "user-msg-row" } };
  const live = { key: "live box" };
  const messages = [
    { role: "user", content: "old prompt" },
    { role: "user", content: "" },
    { role: "assistant", content: "owned answer" },
  ];
  const h = indexedHistoryHarness({ messages, total: "3", existingNodes: [user, live] });
  h.context.pendingUserInputs.set("s", [{ row: user, prompt: "unmatched prompt", minIndex: 1 }]);
  h.context.turns.set("s", { box: live });
  const snapshot = { messages, histTotal: 3 };
  await h.context.loadHistory(snapshot);
  assert.deepEqual(renderedKeys(h.root), ["old prompt", "pending user", "owned answer", "live box"]);
  const first = renderedNodes(h.root), rawFirst = h.root.children.slice(), answer = first[2];
  const clears = h.clears();
  await h.context.loadHistory(snapshot);
  assert.deepEqual(renderedKeys(h.root), first.map((node) => node.key));
  assert.deepEqual(h.root.children, rawFirst, "the mounted transcript is not rebuilt");
  renderedNodes(h.root).forEach((node, index) => assert.equal(node, first[index], "unchanged canonical rows keep their nodes"));
  assert.equal(renderedNodes(h.root).filter((node) => node === answer).length, 1, "the canonical answer is retained exactly once");
  assert.equal(h.retired.includes(answer), false, "a retained answer keeps its previews");
  assert.equal(h.clears(), clears, "unchanged history is not rebuilt");
});

test("a later pending input follows the previous turn's finished answer", async () => {
  const user = { key: "pending user", classList: { contains: (name) => name === "user-msg-row" } };
  const live = { key: "live box" };
  const messages = [{ role: "user", content: "old prompt" }, { role: "assistant", content: "finished answer" }];
  const h = indexedHistoryHarness({ messages, existingNodes: [user, live] });
  h.context.pendingUserInputs.set("s", [{ row: user, prompt: "next prompt", minIndex: 2 }]);
  h.context.turns.set("s", { box: live });
  await h.context.loadHistory({ messages, histTotal: 2 });
  assert.deepEqual(renderedKeys(h.root), ["old prompt", "finished answer", "pending user", "live box"]);
  assert.equal(h.root.children[2], user);
  assert.equal(h.root.lastElementChild, live);
});

for (const drift of ["grew", "shrank"]) test(`pending placement follows its history anchor when the count ${drift}`, async () => {
  const user = { key: "pending user" }, live = { key: "live box" };
  const oldPrompt = { role: "user", content: "old prompt" };
  const h = indexedHistoryHarness({ existingNodes: [user, live] });
  await h.context.loadHistory({ messages: [oldPrompt], histTotal: drift === "shrank" ? 9 : 1 });
  const afterHistoryKey = h.root.lastElementChild.dataset.canonicalIdentity;
  h.context.pendingUserInputs.set("s", [{
    row: user, prompt: "next prompt", minIndex: drift === "shrank" ? 9 : 1, afterHistoryKey,
  }]);
  h.context.turns.set("s", { box: live });
  const messages = drift === "shrank"
    ? [oldPrompt, { role: "user", content: "" }, { role: "assistant", content: "owned answer" }]
    : [oldPrompt, { role: "assistant", content: "finished answer" }];
  await h.context.loadHistory({ messages, histTotal: messages.length });
  const expected = drift === "shrank"
    ? ["old prompt", "pending user", "owned answer", "live box"]
    : ["old prompt", "finished answer", "pending user", "live box"];
  assert.deepEqual(renderedKeys(h.root), expected);
  const first = h.root.children.slice();
  await h.context.loadHistory({ messages, histTotal: messages.length });
  assert.deepEqual(h.root.children, first, "reattachment keeps the same prompt and answer nodes in order");
});

for (const acknowledged of [false, true]) test(`two pending prompts keep their history anchors (first acknowledged=${acknowledged})`, async () => {
  const oldPrompt = { role: "user", content: "old prompt" };
  const firstRow = { key: "first pending" }, secondRow = { key: "second pending" }, live = { key: "live box" };
  const h = indexedHistoryHarness({ existingNodes: [firstRow, secondRow, live] });
  await h.context.loadHistory({ messages: [oldPrompt], histTotal: 1 });
  const first = {
    row: firstRow, prompt: "first prompt", minIndex: 1,
    afterHistoryKey: h.root.lastElementChild.dataset.canonicalIdentity,
  };
  const second = { row: secondRow, prompt: "second prompt", minIndex: 1, afterInput: first };
  h.context.pendingUserInputs.set("s", [first, second]);
  h.context.turns.set("s", { box: live });
  const messages = [oldPrompt, { role: "assistant", content: "finished answer" }];
  if (acknowledged) messages.push({ role: "user", content: "first prompt" }, { role: "assistant", content: "first answer" });
  await h.context.loadHistory({ messages, histTotal: messages.length });
  assert.deepEqual(renderedKeys(h.root), acknowledged
    ? ["old prompt", "finished answer", "first prompt", "first answer", "second pending", "live box"]
    : ["old prompt", "finished answer", "first pending", "second pending", "live box"]);
  assert.equal(h.context.pendingUserInputs.get("s").length, acknowledged ? 1 : 2);
  const before = h.root.children.slice();
  await h.context.loadHistory({ messages, histTotal: messages.length });
  assert.deepEqual(h.root.children, before);
});

test("an identical earlier prompt cannot move the anchor past the first pending input", async () => {
  const firstRow = { key: "first pending" }, secondRow = { key: "second pending" }, live = { key: "live box" };
  const h = indexedHistoryHarness({ existingNodes: [firstRow, secondRow, live] });
  const oldPrompt = { role: "user", content: "repeat" };
  await h.context.loadHistory({ messages: [oldPrompt], histTotal: 1 });
  const previous = h.root.lastElementChild.dataset;
  const first = {
    row: firstRow, prompt: "repeat", minIndex: 1,
    afterHistoryKey: previous.canonicalIdentity, afterHistoryIndex: Number(previous.canonicalIndex),
  };
  const second = { row: secondRow, prompt: "next prompt", minIndex: 1, afterInput: first };
  h.context.pendingUserInputs.set("s", [first, second]);
  h.context.turns.set("s", { box: live });
  const messages = [oldPrompt, { role: "assistant", content: "old answer" }, oldPrompt, { role: "assistant", content: "new answer" }];
  await h.context.loadHistory({ messages, histTotal: 4 });
  assert.deepEqual(renderedKeys(h.root), ["repeat", "old answer", "repeat", "new answer", "second pending", "live box"]);
  assert.equal(h.context.pendingUserInputs.get("s")[0], second);
});

function scrollHarness({ phone = true, scrollTop = 0 } = {}) {
  const classes = new Set();
  const attrs = {};
  let timer = null;
  let frame = null;
  let frameRequests = 0;
  const button = {
    tabIndex: -1,
    classList: { toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
    setAttribute(name, value) { attrs[name] = value; },
  };
  const msgs = {
    scrollHeight: 1000, clientHeight: 400, scrollTop,
    scrollTo(options) { this.lastScroll = options; this.scrollTop = options.top; },
  };
  const context = vm.createContext({
    window: { matchMedia: (query) => ({ matches: query === "(min-width:701px)" ? !phone : phone }) },
    sessionId: "scroll-session",
    $: (id) => id === "msgs" ? msgs : button,
    setTimeout(fn) { timer = fn; return 1; },
    requestAnimationFrame(fn) { frame = fn; frameRequests++; return frameRequests; },
  });
  const start = script.indexOf("const PHONE_LAYOUT_QUERY");
  const end = script.indexOf("// `parent` mounts", start);
  vm.runInContext(script.slice(start, end), context);
  return {
    context, msgs, button, classes, attrs,
    flushTimer: () => { if (timer) timer(); },
    flushFrame: () => { const fn = frame; frame = null; if (fn) fn(); },
    frameRequests: () => frameRequests,
  };
}

test("phone output follows only while the reader remains near the bottom", () => {
  const away = scrollHarness({ phone: true, scrollTop: 100 });
  assert.equal(away.context.shouldFollowOutput(), false);
  away.context.followOutput(false);
  assert.equal(away.msgs.scrollTop, 100);
  assert.equal(away.classes.has("show"), true);
  assert.equal(away.attrs["aria-hidden"], "false");

  const near = scrollHarness({ phone: true, scrollTop: 596 });
  assert.equal(near.context.shouldFollowOutput(), true);
  near.context.followOutput(true);
  near.context.followOutput(true);
  assert.equal(near.frameRequests(), 1, "stream paints must coalesce into one scroll per frame");
  near.flushFrame();
  assert.equal(near.msgs.scrollTop, 1000);
  assert.equal(near.classes.has("show"), false);

  const directed = scrollHarness({ phone: true, scrollTop: 597 });
  directed.context.noteTranscriptIntent(-1);
  assert.equal(directed.context.shouldFollowOutput(), false, "an upward wheel/touch intent locks before layout changes scrollTop");
  directed.msgs.scrollTop = 600;
  directed.context.followOutput(false);
  assert.equal(directed.msgs.scrollTop, 600, "live DOM updates cannot clear an explicit upward-scroll lock");
  directed.msgs.scrollTop = 597;
  directed.context.noteTranscriptScroll(600);
  assert.equal(directed.context.shouldFollowOutput(), false, "an upward gesture locks following even inside the bottom gap");
  directed.context.followOutput(false);
  assert.equal(directed.msgs.scrollTop, 597);
  directed.msgs.scrollTop = 600;
  directed.context.noteTranscriptScroll(590);
  assert.equal(directed.context.shouldFollowOutput(), false, "content growth that moves scrollTop must not resume following");
  directed.context.noteTranscriptIntent(1);
  assert.equal(directed.context.shouldFollowOutput(), true, "a downward gesture back at the live edge restores following");
});

test("desktop output follows only while the reader is at the live edge", () => {
  const away = scrollHarness({ phone: false, scrollTop: 0 });
  assert.equal(away.context.shouldFollowOutput(), false);
  away.context.followOutput(false);
  assert.equal(away.msgs.scrollTop, 0);
  assert.equal(away.classes.has("show"), true);
  const live = scrollHarness({ phone: false, scrollTop: 600 });
  assert.equal(live.context.shouldFollowOutput(), true);
  live.context.followOutput(true);
  live.flushFrame();
  assert.equal(live.msgs.scrollTop, 1000);
  assert.equal(live.classes.has("show"), false);
});

for (const phone of [false, true]) test((phone ? "phone" : "desktop") + " jump button shows away from the bottom and hides near it", () => {
  const h = scrollHarness({ phone, scrollTop: 100 });
  h.context.updateJumpButton();
  assert.equal(h.classes.has("show"), true);
  assert.equal(h.attrs["aria-hidden"], "false");
  assert.equal(h.button.tabIndex, 0);
  h.msgs.scrollTop = 596;
  h.context.updateJumpButton();
  assert.equal(h.classes.has("show"), false);
  assert.equal(h.attrs["aria-hidden"], "true");
  assert.equal(h.button.tabIndex, -1);
});

for (const phone of [true, false]) test("the " + (phone ? "phone" : "desktop") + " jump button scrolls smoothly to the live edge", () => {
  const h = scrollHarness({ phone, scrollTop: 100 });
  h.context.scrollToLatest(true);
  assert.equal(h.msgs.lastScroll.top, 1000);
  assert.equal(h.msgs.lastScroll.behavior, "smooth");
  assert.equal(h.classes.has("show"), false);
  h.flushTimer();
  assert.equal(h.attrs["aria-hidden"], "true");
});

test("an explicit landing clears the previous transcript's scroll lock", () => {
  const h = scrollHarness({ phone: false, scrollTop: 120 });
  h.context.noteTranscriptIntent(-1);
  assert.equal(h.context.shouldFollowOutput(), false, "the reader is locked away from the live edge");
  h.context.scrollToLatest(false);
  assert.equal(h.msgs.scrollTop, 1000, "the landing jumps to the newest turn");
  assert.equal(h.context.shouldFollowOutput(), true, "and the lock no longer vetoes following");
});

test("a sidebar pick lands on the newest turn after its history loads", () => {
  // The pick starts a landing transcript scroll for the incoming session and
  // loads its history behind that intent; a failed load must not break the pick.
  assert.match(html, /if \(window\.matchMedia && window\.matchMedia\("\(min-width:701px\)"\)\.matches\n\s+&& typeof beginTranscriptScroll === "function"\) beginTranscriptScroll\(nextId\);/);
  assert.match(html, /Promise\.resolve\(loadHistory\(\)\)\.catch\(\(\) => \{\}\);/);
  assert.match(html, /outputFollowSuspended = false;\n  jumpingToBottom = false;\n  restoreDraft\(\);/);
});

test("same-session history refresh preserves a phone reader's position", async () => {
  const old = { key: "mounted-old" };
  const h = historyHarness({ existingKeys: ["old"], existingNodes: [old], follow: false });
  h.root.scrollTop = 37;
  await h.context.loadHistory({ messages: [], histTotal: 2, appendOnly: false });
  assert.equal(h.root.scrollTop, 37);
  assert.equal(h.root.children[0], old);
  assert.deepEqual(renderedKeys(h.root), ["mounted-old", "new"]);
});

test("the jump-to-latest control is an overlaid round button on phone and desktop", () => {
  assert.match(html, /<button id="jumpbottom"[^>]*aria-label="jump to latest message"/);
  assert.match(html, /#jumpbottom \{[\s\S]*?position:absolute;[\s\S]*?border-radius:50%;/);
  assert.match(html, /@media \(max-width:700px\) \{[\s\S]*?#jumpbottom\.show \{ opacity:1;/);
  assert.match(html, /@media \(min-width:701px\) \{[\s\S]*?#jumpbottom \{ display:grid;[\s\S]*?#jumpbottom\.show \{ opacity:1;/);
});

test("the phone composer keeps an idle prompt compact beside smaller action buttons", () => {
  const desktopCss = html.slice(html.indexOf("<style>"), html.indexOf("@media (max-width:700px)"));
  const phoneCss = html.slice(html.indexOf("@media (max-width:700px)"), html.indexOf("</style>"));
  assert.match(desktopCss, /#composer \{ display:flex;/);
  assert.match(phoneCss, /#composer \{[\s\S]*?display:grid;[\s\S]*?grid-template-areas:"image prompt send";[\s\S]*?align-items:end;/);
  assert.match(phoneCss, /#composer\.side-note-ready \{[\s\S]*?grid-template-areas:"image prompt sidenote send";/);
  assert.match(phoneCss, /#sidenote\.show \{[\s\S]*?grid-area:sidenote;[\s\S]*?display:inline-flex;/);
  assert.match(phoneCss, /#prompt \{[\s\S]*?grid-area:prompt;[\s\S]*?min-height:40px;[\s\S]*?max-height:calc\(var\(--app-height, 100vh\) \* \.38\);/);
  assert.match(desktopCss, /#prompt \{[\s\S]*?-webkit-user-select:text; user-select:text; -webkit-touch-callout:default;/);
  assert.doesNotMatch(phoneCss, /#prompt:focus[^\{]*\{[^}]*min-height/);
  assert.match(phoneCss, /#img, #send \{[\s\S]*?width:42px;[\s\S]*?height:42px;[\s\S]*?border-radius:50%;/);
  assert.match(script, /\$\("sidenote"\)\.onclick = sendSideNote;/);
  assert.match(html, /<button id="send" class="primary">➤<\/button>/);
});

function growPromptHeight({ phone, scrollHeight, innerHeight = 600, value = "message" }) {
  const start = script.indexOf("function growPrompt(");
  const end = script.indexOf("function textOf(", start);
  const prompt = { value, scrollHeight, style: { height: "" } };
  const context = vm.createContext({
    window: { innerHeight, matchMedia: () => ({ matches: phone }) },
    $: (id) => { assert.equal(id, "prompt"); return prompt; },
    promptComposing: false,
    traceMark: () => {},
  });
  vm.runInContext(script.slice(start, end), context);
  context.growPrompt();
  return prompt.style.height;
}

test("prompt auto-growth uses a roomier phone cap without changing desktop", () => {
  assert.equal(growPromptHeight({ phone: true, scrollHeight: 500 }), "228px");
  assert.equal(growPromptHeight({ phone: false, scrollHeight: 500 }), "180px");
  assert.equal(growPromptHeight({ phone: true, scrollHeight: 96 }), "96px");
  assert.equal(growPromptHeight({ phone: true, scrollHeight: 84, value: "" }), "auto");
});

test("composer height is reset after send and refreshed after image attachment", () => {
  assert.match(html, /setPromptValue\("", true\);[^\n]*\n\s*\$\("prompt"\)\.style\.height = "auto";/);
  assert.match(html, /added\.map\([\s\S]*?\.join\("\\n"\);[\s\S]*?setPromptValue\(prompt\.value \+ marker\)[\s\S]*?growPrompt\(\);/);
  assert.match(html, /e\.key === "Enter" && \(e\.ctrlKey \|\| e\.metaKey\)/);
  assert.match(html, /\$\("prompt"\)\.addEventListener\("paste", handlePromptPaste\);/);
});

function pasteHarness(value = "") {
  const calls = [];
  const prompt = {
    value,
    selectionStart: value.length,
    selectionEnd: value.length,
    setRangeText(text, start, end) {
      this.value = this.value.slice(0, start) + text + this.value.slice(end);
      this.selectionStart = this.selectionEnd = start + text.length;
    },
  };
  class Reader {
    readAsDataURL(file) {
      this.result = "data:" + file.type + ";base64," + (file.data || "aW1hZ2U=");
      this.onload();
    }
  }
  const context = vm.createContext({
    $: (id) => { assert.equal(id, "prompt"); return prompt; },
    growPrompt: () => calls.push("grow"),
    persistDraft: () => calls.push("draft"),
    promptComposing: false,
    promptHistory: null, traceMark() {},
    setPromptValue: (value) => { prompt.value = value; return true; },
    note: (message) => calls.push(message),
    pendingImages: [],
    FileReader: Reader,
    sessionId: "session-a",
    composerImageDrafts: new Map(),
    draftKey: (id) => "draft:" + id,
  });
  const start = script.indexOf("function insertPromptText(");
  const end = script.indexOf("function textOf(", start);
  vm.runInContext(script.slice(start, end), context);
  return { context, prompt, calls };
}

test("prompt paste inserts plain text at the selection and persists the draft", () => {
  const h = pasteHarness("before after");
  h.prompt.selectionStart = 7;
  h.prompt.selectionEnd = 12;
  let prevented = false;
  h.context.handlePromptPaste({
    clipboardData: { items: [], getData: (type) => type === "text/plain" ? "pasted" : "" },
    preventDefault: () => { prevented = true; },
  });
  assert.equal(prevented, true);
  assert.equal(h.prompt.value, "before pasted");
  assert.deepEqual(h.calls, ["grow", "draft"]);
});

test("prompt paste attaches clipboard images without inserting their fallback URL", async () => {
  const h = pasteHarness("caption");
  const file = { name: "photo one.png", type: "image/png" };
  let prevented = false;
  await h.context.handlePromptPaste({
    clipboardData: {
      items: [{ kind: "file", type: "image/png", getAsFile: () => file }],
      getData: () => "https://example.test/photo.png",
    },
    preventDefault: () => { prevented = true; },
  });
  assert.equal(prevented, true);
  assert.equal(h.prompt.value, "caption\n[[image:photo_one.png]]");
  assert.equal(JSON.stringify(h.context.pendingImages),
    '[{"filename":"photo_one.png","data_base64":"aW1hZ2U="}]');
  assert.deepEqual(h.calls, ["grow", "draft"]);
});

test("prompt paste accepts mobile clipboard image files without DataTransfer items", async () => {
  const h = pasteHarness();
  const file = { name: "mobile.png", type: "image/png" };
  await h.context.handlePromptPaste({
    clipboardData: { files: [file], getData: () => "" },
    preventDefault() {},
  });
  assert.equal(h.prompt.value, "[[image:mobile.png]]");
  assert.equal(h.context.pendingImages.length, 1);
});

test("prompt paste gives same-named clipboard images distinct upload names", async () => {
  const h = pasteHarness();
  const paste = (file) => h.context.handlePromptPaste({
    clipboardData: {
      items: [{ kind: "file", type: "image/png", getAsFile: () => file }],
      getData: () => "",
    },
    preventDefault() {},
  });
  // Browsers hand every clipboard image the same generic filename; the server
  // stages uploads by name, so equal names would collapse to one picture.
  await paste({ name: "image.png", type: "image/png", data: "b25l" });
  await paste({ name: "image.png", type: "image/png", data: "dHdv" });
  await paste({ name: "IMAGE.PNG", type: "image/png", data: "dGhyZWU=" });
  assert.equal(h.prompt.value,
    "[[image:image.png]]\n[[image:image-2.png]]\n[[image:IMAGE-3.PNG]]");
  assert.equal(JSON.stringify(h.context.pendingImages),
    '[{"filename":"image.png","data_base64":"b25l"},'
    + '{"filename":"image-2.png","data_base64":"dHdv"},'
    + '{"filename":"IMAGE-3.PNG","data_base64":"dGhyZWU="}]');
});
