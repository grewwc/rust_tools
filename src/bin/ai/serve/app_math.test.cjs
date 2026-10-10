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

test("screenshot display and inline formulas preserve their exact TeX", () => {
  const { render } = harness();
  const formula = String.raw`0.999\ldots := \lim_{n\to\infty} S_n,\quad S_n = \underbrace{0.99\ldots9}_{n\text{个}} = 1 - 10^{-n}`;
  const result = render("## Definition\n\n$$" + formula + "$$\n\n" + String.raw`$S_n$ converges in $\mathbb{R}$.`);
  const nodes = nodesOf(result);
  assert.deepEqual(nodes.map((n) => n.dataset.tex), [formula, "S_n", String.raw`\mathbb{R}`]);
  assert.equal(nodes[0].classList.contains("display"), true);
  assert.match(result, /<div class="h lv2">Definition<\/div>/);
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
  Object.assign(h.context, { bubble, raw: "$x", paintTimer: null, sid: "s", sessionId: "s" });
  const paint = script.match(/const paint = \(\) => \{([\s\S]*?)\n  \};/)[1];
  vm.runInContext(paint, h.context);
  assert.equal(h.assets.length, 0);
  h.context.raw = "$x$";
  vm.runInContext(paint, h.context);
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