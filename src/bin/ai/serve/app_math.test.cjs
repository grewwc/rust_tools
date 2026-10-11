// Run with: node --test src/bin/ai/serve/app_math.test.cjs
// Exercise the embedded client itself; no npm dependencies or network required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const renderer = script.slice(script.indexOf("const esc ="), script.indexOf("function sessionTitle("));
const decode = (s) => s.replace(/&(amp|lt|gt|quot);/g, (_, key) => ({ amp: "&", lt: "<", gt: ">", quot: '"' })[key]);
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
    querySelectorAll() { return this.nodes.filter((node) => "tex" in node.dataset); },
    querySelector() { return this.querySelectorAll()[0] || null; },
  };
  const document = { createElement: (tag) => ({ tag }), head: { appendChild: (el) => assets.push(el) } };
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

test("phone history exposes a dedicated rewind action without whole-bubble taps", () => {
  const start = script.indexOf("function addUserBubble(");
  const end = script.indexOf("async function rewindTo(", start);
  assert.ok(start >= 0 && end > start);
  const actions = [];
  const rewinds = [];
  const bubble = { appendChild: (node) => actions.push(node) };
  const context = vm.createContext({
    window: { matchMedia: () => ({ matches: true }) },
    document: { createElement: () => ({ setAttribute(name, value) { this[name] = value; } }) },
    addMsg: () => bubble,
    rewindTo: (...args) => rewinds.push(args),
  });
  vm.runInContext(script.slice(start, end), context);
  context.addUserBubble("keep this preview", 17);
  assert.equal(actions.length, 1);
  assert.equal(actions[0].textContent, "↶ rewind");
  let stopped = false;
  actions[0].onclick({ stopPropagation: () => { stopped = true; } });
  assert.equal(stopped, true);
  assert.deepEqual(rewinds, [[17, "keep this preview"]]);
  rewinds.length = 0;
  bubble.onclick({ target: { closest: () => null } });
  assert.equal(rewinds.length, 0);
  context.window.matchMedia = () => ({ matches: false });
  bubble.onclick({ target: { closest: () => null } });
  assert.deepEqual(rewinds, [[17, "keep this preview"]]);
});

test("rewind action is visible only in the phone layout", () => {
  assert.ok(html.includes(".rewind-action { display:none; }"));
  assert.match(html, /@media \(max-width:700px\) \{[\s\S]*?\.msg\.rewindable \.rewind-action \{[\s\S]*?display:block;/);
});

test("phone session hold targets its own row and scrolling cancels it", () => {
  const start = script.indexOf("const SESSION_HOLD_MS");
  const end = script.indexOf("async function deleteSession(", start);
  assert.ok(start >= 0 && end > start);
  let scheduled = null;
  const cleared = [];
  const deleted = [];
  const context = vm.createContext({
    window: { matchMedia: () => ({ matches: true }) },
    Date: { now: () => 1000 },
    setTimeout: (fn, ms) => { scheduled = { fn, ms, id: 9 }; return 9; },
    clearTimeout: (id) => cleared.push(id),
    deleteSession: (id) => deleted.push(id),
  });
  vm.runInContext(script.slice(start, end), context);
  const row = () => {
    const classes = new Set();
    return {
      dataset: {},
      classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name), contains: (name) => classes.has(name) },
    };
  };
  const touch = (x, y) => ({
    pointerType: "touch", isPrimary: true, button: 0, clientX: x, clientY: y,
    target: { closest: () => null },
  });

  const held = row();
  context.bindSessionLongPress(held, { id: "hi-session" });
  held.onpointerdown(touch(10, 20));
  assert.equal(held.classList.contains("holding"), true);
  assert.equal(scheduled.ms, 650);
  scheduled.fn();
  assert.deepEqual(deleted, ["hi-session"]);
  assert.equal(context.sessionHoldConsumed(held), true);
  assert.equal(held.classList.contains("holding"), false);

  const scrolled = row();
  context.bindSessionLongPress(scrolled, { id: "keep-session" });
  scrolled.onpointerdown(touch(10, 20));
  scrolled.onpointermove(touch(10, 40));
  assert.deepEqual(cleared, [9]);
  assert.equal(scrolled.classList.contains("holding"), false);
  assert.deepEqual(deleted, ["hi-session"]);
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
    confirm: (text) => { calls.push(["confirm", text]); return true; },
    api: async (path, options) => { calls.push([path, options]); },
    loadSessions: async () => { sessionsLoaded += 1; },
    loadHistory: async () => { historyLoaded += 1; },
    stopStaleFollow: () => calls.push(["stop"]),
    hideConfirm: () => calls.push(["hide"]),
    localStorage: { removeItem: (key) => calls.push(["remove", key]) },
    addMsg: () => {},
  });
  vm.runInContext(script.slice(start, end), context);
  await vm.runInContext('deleteSession("hi-session")', context);
  assert.deepEqual(calls[0], ["confirm", 'delete "hi"?']);
  assert.equal(calls[1][0], "/sessions/hi-session");
  assert.equal(calls[1][1].method, "DELETE");
  assert.equal(context.sessionId, "open-session");
  assert.equal(sessionsLoaded, 1);
  assert.equal(historyLoaded, 0);
  assert.equal(calls.some(([kind]) => kind === "remove"), false);
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

test("the real streaming paint typesets completed formulas before the final event", async () => {
  const h = harness();
  const bubble = {
    set innerHTML(markup) { h.msgs.nodes = nodesOf(markup); },
    querySelector: () => h.msgs.querySelector(),
    querySelectorAll: () => h.msgs.querySelectorAll(),
  };
  Object.assign(h.context, {
    bubble, raw: "$x", paintTimer: null, sid: "s", sessionId: "s",
    shouldFollowOutput: () => true, followOutput() {},
  });
  const paint = script.match(/const paint = \(\) => \{([\s\S]*?)\n  \};/)[1];
  vm.runInContext("{" + paint + "}", h.context);
  assert.equal(h.assets.length, 0);
  h.context.raw = "$x$";
  vm.runInContext("{" + paint + "}", h.context);
  assert.equal(h.assets.length, 2);
  await h.finishLoading();
  assert.equal(h.msgs.nodes[0].textContent, "typeset:x");
  h.context.raw += " and $y$";
  vm.runInContext(paint, h.context);
  assert.deepEqual(h.msgs.nodes.map((node) => node.textContent), ["typeset:x", "typeset:y"]);
});

test("image decoration leaves math text and rendered math subtrees alone", () => {
  const h = harness();
  const node = { nodeValue: "plot.svg", parentElement: { closest: (selector) => selector === ".math" } };
  h.context.document.createTreeWalker = () => {
    let pending = true;
    return { currentNode: node, nextNode() { const next = pending; pending = false; return next; } };
  };
  h.context.NodeFilter = { SHOW_TEXT: 4 };
  const start = script.indexOf("function decorateImages(root)");
  const end = script.indexOf("\n}", start) + 2;
  vm.runInContext("const MAX_PREVIEWS_PER_MSG = 4;\n" + script.slice(start, end), h.context);
  assert.doesNotThrow(() => h.context.decorateImages(h.msgs));
});

function historyHarness({ delayed = false, existingKeys = [], existingNodes = [], follow = true } = {}) {
  let resolveResponse;
  let apiCalls = 0;
  let clears = 0;
  let revokes = 0;
  const response = {
    headers: { get: () => "2" },
    json: async () => [{ role: "user", content: "prompt" }, { role: "assistant", content: "reply" }],
  };
  const pending = delayed ? new Promise((resolve) => { resolveResponse = () => resolve(response); }) : Promise.resolve(response);
  const root = {
    children: [], scrollTop: 0, scrollHeight: 100,
    appendChild(node) { this.children.push(node); node.parent = this; return node; },
    get lastElementChild() { return this.children[this.children.length - 1]; },
    set textContent(value) { assert.equal(value, ""); clears += 1; this.children = []; },
  };
  for (const node of existingNodes) root.appendChild(node);
  for (const node of root.children) node.remove = function remove() {
    const i = root.children.indexOf(this);
    if (i >= 0) root.children.splice(i, 1);
  };
  const rows = [
    { key: "old", text: "prompt" },
    { key: "new", text: "reply" },
  ];
  const context = vm.createContext({
    sessionId: "s", historySession: existingKeys.length ? "s" : "", historyKeys: existingKeys.slice(),
    lastMsgCount: null, staleKey: "", staleDone: false, previewUrls: [],
    $: () => root,
    api: async () => { apiCalls += 1; return pending; },
    historyRowsOf: () => ({ rows, shownCount: rows.length }),
    histKey: () => "tail", histEndsTurn: () => true,
    syncStopButton() {}, renderSessionBar() {}, attachLiveBox() {},
    shouldFollowOutput: () => follow,
    followOutput(wasFollowing) { if (wasFollowing) root.scrollTop = root.scrollHeight; },
    revokePreviews() { revokes += 1; },
    renderHistoryRow(row, parent) {
      const node = { key: row.key, parent };
      node.remove = function remove() {
        const i = root.children.indexOf(this);
        if (i >= 0) root.children.splice(i, 1);
      };
      parent.appendChild(node);
    },
    addMsg() {},
  });
  const start = script.indexOf("async function loadHistory(");
  const end = script.indexOf("// Historical user inputs", start);
  vm.runInContext(script.slice(start, end), context);
  return {
    context, root, rows, resolveResponse,
    apiCalls: () => apiCalls, clears: () => clears, revokes: () => revokes,
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
  assert.deepEqual(h.root.children.map((node) => node.key), ["old", "new"]);
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
  assert.equal(h.revokes(), 0);
  assert.equal(h.root.children[0], old);
  assert.deepEqual(h.root.children.map((node) => node.key), ["mounted-old", "new"]);
  h.rows.push({ key: "newer", text: "later progress" });
  await h.context.loadHistory({ messages: [], histTotal: 3, appendOnly: true });
  assert.equal(h.clears(), 0);
  assert.equal(h.root.children[0], old);
  assert.equal(h.root.children[1], firstProgress);
  assert.deepEqual(h.root.children.map((node) => node.key), ["mounted-old", "new", "newer"]);
});

test("stale history falls back to safe preview cleanup when object URLs exist", async () => {
  const old = { key: "mounted-old" };
  const h = historyHarness({ existingKeys: ["old"], existingNodes: [old] });
  h.context.previewUrls.push("blob:old-preview");
  await h.context.loadHistory({ messages: [], histTotal: 2, appendOnly: true });
  assert.equal(h.clears(), 1);
  assert.equal(h.revokes(), 1);
  assert.notEqual(h.root.children[0], old);
  assert.deepEqual(h.root.children.map((node) => node.key), ["old", "new"]);
});

test("a stale live box is not reattached beside canonical history", () => {
  const box = {};
  const root = { appended: [], appendChild(node) { this.appended.push(node); }, scrollTop: 0, scrollHeight: 10 };
  const context = vm.createContext({
    sessionId: "s", staleBox: box, turns: new Map([["s", { box }]]),
    $: () => root, armHistoryFollow() {},
  });
  const start = script.indexOf("function attachLiveBox(");
  const end = script.indexOf("// A turn the user has left", start);
  vm.runInContext(script.slice(start, end), context);
  context.attachLiveBox();
  assert.deepEqual(root.appended, []);
});

function scrollHarness({ phone = true, scrollTop = 0 } = {}) {
  const classes = new Set();
  const attrs = {};
  let timer = null;
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
    window: { matchMedia: () => ({ matches: phone }) },
    $: (id) => id === "msgs" ? msgs : button,
    setTimeout(fn) { timer = fn; return 1; },
  });
  const start = script.indexOf("const PHONE_LAYOUT_QUERY");
  const end = script.indexOf("// `parent` mounts", start);
  vm.runInContext(script.slice(start, end), context);
  return { context, msgs, button, classes, attrs, flushTimer: () => { if (timer) timer(); } };
}

test("phone output follows only while the reader remains near the bottom", () => {
  const away = scrollHarness({ phone: true, scrollTop: 100 });
  assert.equal(away.context.shouldFollowOutput(), false);
  away.context.followOutput(false);
  assert.equal(away.msgs.scrollTop, 100);
  assert.equal(away.classes.has("show"), true);
  assert.equal(away.attrs["aria-hidden"], "false");

  const near = scrollHarness({ phone: true, scrollTop: 520 });
  assert.equal(near.context.shouldFollowOutput(), true);
  near.context.followOutput(true);
  assert.equal(near.msgs.scrollTop, 1000);
  assert.equal(near.classes.has("show"), false);
});

test("desktop output preserves always-follow behavior", () => {
  const h = scrollHarness({ phone: false, scrollTop: 0 });
  assert.equal(h.context.shouldFollowOutput(), true);
  h.context.followOutput(true);
  assert.equal(h.msgs.scrollTop, 1000);
  assert.equal(h.classes.has("show"), false);
});

test("the phone jump button scrolls smoothly to the live edge", () => {
  const h = scrollHarness({ phone: true, scrollTop: 100 });
  h.context.scrollToLatest(true);
  assert.equal(h.msgs.lastScroll.top, 1000);
  assert.equal(h.msgs.lastScroll.behavior, "smooth");
  assert.equal(h.classes.has("show"), false);
  h.flushTimer();
  assert.equal(h.attrs["aria-hidden"], "true");
});

test("same-session history refresh preserves a phone reader's position", async () => {
  const old = { key: "mounted-old" };
  const h = historyHarness({ existingKeys: ["old"], existingNodes: [old], follow: false });
  h.root.scrollTop = 37;
  await h.context.loadHistory({ messages: [], histTotal: 2, appendOnly: false });
  assert.equal(h.root.scrollTop, 37);
  assert.deepEqual(h.root.children.map((node) => node.key), ["old", "new"]);
});

test("the jump-to-latest control is a phone-only overlaid round button", () => {
  assert.match(html, /<button id="jumpbottom"[^>]*aria-label="jump to latest message"/);
  assert.match(html, /#jumpbottom \{[\s\S]*?position:absolute;[\s\S]*?border-radius:50%;/);
  assert.match(html, /@media \(max-width:700px\) \{[\s\S]*?#jumpbottom\.show \{ opacity:1;/);
});

test("the phone composer stacks a roomy prompt above round action buttons", () => {
  const desktopCss = html.slice(html.indexOf("<style>"), html.indexOf("@media (max-width:700px)"));
  const phoneCss = html.slice(html.indexOf("@media (max-width:700px)"), html.indexOf("</style>"));
  assert.match(desktopCss, /#composer \{ display:flex;/);
  assert.match(phoneCss, /#composer \{[\s\S]*?display:grid;[\s\S]*?grid-template-areas:"prompt prompt prompt" "image \. send";/);
  assert.match(phoneCss, /#prompt \{[\s\S]*?grid-area:prompt;[\s\S]*?min-height:42px;[\s\S]*?max-height:38dvh;/);
  assert.match(phoneCss, /#prompt:focus, #prompt:not\(:placeholder-shown\) \{ min-height:84px; \}/);
  assert.match(phoneCss, /#img, #send \{[\s\S]*?width:48px;[\s\S]*?border-radius:50%;/);
  assert.match(html, /<button id="send" class="primary">➤<\/button>/);
});

function growPromptHeight({ phone, scrollHeight, innerHeight = 600, value = "message" }) {
  const start = script.indexOf("function growPrompt(");
  const end = script.indexOf("function textOf(", start);
  const prompt = { value, scrollHeight, style: { height: "" } };
  const context = vm.createContext({
    window: { innerHeight, matchMedia: () => ({ matches: phone }) },
    $: (id) => { assert.equal(id, "prompt"); return prompt; },
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
  assert.match(html, /\$\("prompt"\)\.value = "";\s*\$\("prompt"\)\.style\.height = "auto";/);
  assert.match(html, /pendingImages\.map\([\s\S]*?\.join\("\\n"\);\s*growPrompt\(\);/);
  assert.match(html, /e\.key === "Enter" && \(e\.ctrlKey \|\| e\.metaKey\)/);
});