// Run with: node --test src/bin/ai/serve/app_copy.test.cjs
// Exercise the page's renderer, delegated clicks and completion hooks without
// network access or a browser dependency.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function between(start, end) {
  const first = script.indexOf(start), last = script.indexOf(end, first);
  assert.ok(first >= 0 && last > first, `${start} through ${end}`);
  return script.slice(first, last);
}

function harness({ clipboard, fallback = true, phone = false } = {}) {
  const copied = [], areas = [], focusCalls = [];
  const document = { activeElement: null };
  class Element {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.children = [];
      this.parentNode = null;
      this.dataset = {};
      this.style = {};
      this.attributes = {};
      this.className = "";
      this.disabled = false;
      this.classList = {
        contains: (name) => this.className.split(/\s+/).includes(name),
        add: (...names) => { this.className = [...new Set(this.className.split(/\s+/).filter(Boolean).concat(names))].join(" "); },
        remove: (...names) => { this.className = this.className.split(/\s+/).filter((name) => !names.includes(name)).join(" "); },
      };
    }
    get firstChild() { return this.children[0] || null; }
    get lastElementChild() { return this.children.at(-1) || null; }
    get nextElementSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; }
    get isConnected() { return this === document.body || !!this.parentNode?.isConnected; }
    appendChild(child) {
      child.remove();
      child.parentNode = this;
      this.children.push(child);
      return child;
    }
    insertBefore(child, before) {
      if (before === null) return this.appendChild(child);
      child.remove();
      const index = this.children.indexOf(before);
      assert.ok(index >= 0, "the insertion anchor is a child");
      this.children.splice(index, 0, child);
      child.parentNode = this;
      return child;
    }
    removeChild(child) {
      assert.ok(this.children.includes(child));
      this.children.splice(this.children.indexOf(child), 1);
      child.parentNode = null;
      return child;
    }
    remove() { if (this.parentNode) this.parentNode.removeChild(this); }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) { return this.attributes[name] ?? null; }
    matches(selector) {
      const attr = /^\[([\w-]+)\]$/.exec(selector);
      if (attr) return this.getAttribute(attr[1]) !== null;
      const tag = /^[\w-]+/.exec(selector);
      const classes = [...selector.matchAll(/\.([\w-]+)/g)].map((match) => match[1]);
      return (!tag || this.tagName === tag[0].toUpperCase()) && classes.every((name) => this.classList.contains(name));
    }
    closest(selector) {
      for (let node = this; node; node = node.parentNode) {
        if (selector.split(",").some((part) => node.matches(part.trim()))) return node;
      }
      return null;
    }
    querySelectorAll(selector) {
      return this.children.flatMap((child) => [
        ...(selector.split(",").some((part) => child.matches(part.trim())) ? [child] : []),
        ...child.querySelectorAll(selector),
      ]);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    set innerHTML(markup) {
      while (this.firstChild) this.firstChild.remove();
      this.text = undefined;
      if (markup) {
        const fragment = new Element("#text");
        fragment.markup = String(markup);
        this.appendChild(fragment);
      }
    }
    get innerHTML() { return this.markup ?? this.children.map((child) => child.innerHTML).join(""); }
    set textContent(value) {
      while (this.firstChild) this.firstChild.remove();
      this.markup = undefined;
      this.text = String(value);
    }
    get textContent() {
      if (this.text !== undefined) return this.text;
      if (this.markup !== undefined) return this.markup.replace(/<[^>]*>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
      return this.children.map((child) => child.textContent).join("");
    }
    get innerText() { return this.renderedText ?? this.textContent; }
    focus(options) { document.activeElement = this; focusCalls.push({ element: this, options }); }
    select() { this.focus(); this.setSelectionRange(0, this.value.length); }
    setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  }
  document.createElement = (tag) => {
    const node = new Element(tag);
    if (tag === "textarea") areas.push(node);
    return node;
  };
  document.body = new Element("body");
  document.execCommand = (command) => {
    assert.equal(command, "copy");
    const area = document.activeElement;
    copied.push(area.value);
    if (fallback instanceof Error) throw fallback;
    return fallback;
  };
  const msgs = document.createElement("div");
  document.body.appendChild(msgs);
  const rates = new Map();
  const context = vm.createContext({
    document, window: {}, $: () => msgs,
    sessionId: "s", isPhoneLayout: () => phone,
    localStorage: { getItem: (key) => rates.get(key) || null, setItem: (key, value) => rates.set(key, value) },
    shouldFollowOutput: () => false, followOutput() {}, updateQuestionButton() {}, decorateAndLoad() {},
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    formatElapsed: () => "0:00",
    createLiveActivity: () => ({ label: () => "Processing…", setPhase() {}, anchor() {}, finish() {}, tool() {}, subagents() {} }),
    ...(clipboard === undefined ? {} : { navigator: { clipboard } }),
  });
  vm.runInContext(between("// BEGIN execution model", "// END execution model"), context);
  vm.runInContext(between("const esc =", "function sessionTitle("), context);
  vm.runInContext(between("function addMsg(", "// --- server-side media"), context);
  vm.runInContext(between("const RATE_STORAGE =", "function addFold("), context);
  return {
    context, document, msgs, copied, areas, focusCalls,
    answer: (text, parent = msgs) => context.addMsgMd(text, parent),
    tap: (target) => context.handlePreviewClick({ target }),
  };
}

function streamHarness(options) {
  const h = harness(options);
  const box = h.document.createElement("div"), liveEntry = { box };
  h.msgs.appendChild(box);
  Object.assign(h.context, {
    box, sid: "s", sessionId: "s", sendBase: "", liveEntry, turns: new Map([["s", liveEntry]]),
    turnRunningPid: 0, endedTurnPid: 0,
    finished: false, turnError: false, finishFollow: false, confirmPollSeq: 0, stopStaleFollow() {},
  });
  vm.runInContext(between('  let bubble = addMsgMd("", box);', "  let turnStarted = false;"), h.context);
  const dispatch = between('        if (kind === "done") {', '      if (finished) break;').replace(/\s*}\s*$/, "");
  const completion = between("    flushPaint(); // flush the tail before final decoration or stale recovery", "    if (!finished && !turnError) {");
  return {
    ...h, box,
    emit(kind, value) {
      h.context.kind = kind;
      h.context.payload = typeof value === "string" ? value : JSON.stringify(value || {});
      vm.runInContext(`for (;;) { ${dispatch}\nbreak; } flushPaint();`, h.context);
      if (kind === "done") vm.runInContext(completion, h.context);
    },
    seal: () => vm.runInContext("sealAnswerSegment();", h.context),
  };
}

test("completed assistant answers have exactly one copy control and other rows have none", () => {
  const h = harness();
  const first = h.answer("第一条结论"), second = h.answer("第二条结论");
  const user = h.context.addMsg("user", "问题", "user", h.msgs);
  const tool = h.context.addMsg("tool", "tool output", "asst", h.msgs);
  const pending = h.context.addMsg("assistant", "Responding…", "asst md pending", h.msgs);
  const empty = h.answer("\n ");
  const body = first.querySelector(".answer-body"), rendered = body.firstChild;
  h.context.addAnswerCopy(first, "第一条结论");
  for (const row of [user, tool, pending]) h.context.addAnswerCopy(row, row.textContent);
  assert.equal(h.msgs.querySelectorAll("[data-copy-answer]").length, 2);
  for (const row of [first, second]) assert.equal(row.querySelectorAll("[data-copy-answer]").length, 1);
  for (const row of [user, tool, pending, empty]) assert.equal(row.querySelectorAll("[data-copy-answer]").length, 0);
  assert.equal(first.children[0], body);
  assert.equal(body.firstChild, rendered, "the rendered nodes survive footer attachment");
  assert.equal(first.children[1].className, "answer-actions", "the footer follows the text in normal flow");
  assert.equal(first.querySelector("[data-copy-answer]").type, "button");
  assert.equal(first.querySelector("[data-copy-answer]").getAttribute("aria-label"), "复制");
  assert.equal(first.querySelector(".answer-copy-status").getAttribute("role"), "status");
  assert.match(html, /\.msg\.md \.answer-copy \{[^}]*min-width:52px;[^}]*min-height:32px;/);
  assert.match(html, /\.msg\.md \.answer-copy::after \{[^}]*inset:-6px;/);
  assert.doesNotMatch(html.match(/\.msg\.md \.answer-actions \{([^}]*)\}/)[1], /position:\s*absolute/);
});

test("tapping the answer control copies raw markdown with its line breaks through the clipboard", async () => {
  const copied = [];
  const h = harness({ clipboard: { writeText: async (text) => { copied.push(text); } } });
  const raw = "# 结论\n\n第一行。\n第二行。\n\n- 保留 Markdown\n";
  const answer = h.answer(raw), button = answer.querySelector("[data-copy-answer]");
  const icon = h.document.createElement("span");
  button.appendChild(icon);
  assert.equal(await h.tap(icon), true, "delegation also handles a nested tap target");
  assert.deepEqual(copied, [raw]);
  assert.equal(h.areas.length, 0);
  assert.equal(answer.querySelector(".answer-copy-status").textContent, "已复制");
  assert.equal(button.disabled, false);
});

test("an unavailable clipboard copies through a readonly textarea and restores focus", async () => {
  const h = harness();
  const raw = "结论一\n\n结论二\n", answer = h.answer(raw);
  const prompt = h.document.createElement("textarea");
  h.document.body.appendChild(prompt);
  prompt.focus();
  assert.equal(await h.tap(answer.querySelector("[data-copy-answer]")), true);
  assert.deepEqual(h.copied, [raw]);
  const area = h.areas.at(-1);
  assert.equal(area.getAttribute("readonly"), "");
  assert.equal(area.getAttribute("aria-hidden"), "true");
  assert.equal(area.style.opacity, "0");
  assert.equal(area.selectionStart, 0);
  assert.equal(area.selectionEnd, raw.length);
  assert.equal(area.parentNode, null, "the temporary textarea is removed");
  assert.equal(h.document.activeElement, prompt);
  assert.equal(h.focusCalls.at(-1).options.preventScroll, true);
  assert.equal(answer.querySelector(".answer-copy-status").textContent, "已复制");
});

test("a rejected clipboard promise falls back to selection copy", async () => {
  const attempts = [];
  const h = harness({ clipboard: { writeText: async (text) => { attempts.push(text); throw new Error("denied"); } } });
  const raw = "第一行\n第二行", answer = h.answer(raw);
  assert.equal(await h.tap(answer.querySelector("[data-copy-answer]")), true);
  assert.deepEqual(attempts, [raw]);
  assert.deepEqual(h.copied, [raw]);
  assert.equal(answer.querySelector(".answer-copy-status").textContent, "已复制");
});

for (const fallback of [false, new Error("selection copy denied")]) {
  test(`both copy paths failing reports an inline status (fallback ${fallback === false ? "returns false" : "throws"})`, async () => {
    const h = harness({ clipboard: { writeText: async () => { throw new Error("clipboard denied"); } }, fallback });
    const answer = h.answer("无法复制的结论"), button = answer.querySelector("[data-copy-answer]");
    assert.equal(await h.tap(button), false);
    assert.deepEqual(h.copied, ["无法复制的结论"]);
    assert.equal(answer.querySelector(".answer-copy-status").textContent, "复制失败");
    assert.equal(button.disabled, false, "the user can retry");
    assert.equal(h.areas[0].parentNode, null, "failure also removes the temporary textarea");
  });
}

test("without stored source the copy uses rendered body text and excludes the footer", async () => {
  const copied = [];
  const h = harness({ clipboard: { writeText: async (text) => { copied.push(text); } } });
  const answer = h.answer("原始文本"), body = answer.querySelector(".answer-body");
  body.renderedText = "第一行\n\n第二行\n";
  answer.querySelector(".answer-copy-status").textContent = "已复制";
  h.context.answer = answer;
  vm.runInContext("answerCopyTexts.delete(answer);", h.context);
  assert.equal(await h.tap(answer.querySelector("[data-copy-answer]")), true);
  assert.deepEqual(copied, ["第一行\n\n第二行\n"]);
});

test("history answers copy the original message including surrounding line breaks", async () => {
  const copied = [];
  const h = harness({ clipboard: { writeText: async (text) => { copied.push(text); } } });
  h.context.textOf = (content) => content;
  vm.runInContext(between("function renderHistoryRow(", "// A session the server reports"), h.context);
  const raw = "\n# 结论\n\n完整的回答\n\n";
  h.context.renderHistoryRow({ message: { role: "assistant", content: raw }, text: raw.trim(), historyIndex: 2 }, h.msgs);
  const answer = h.msgs.lastElementChild;
  assert.equal(answer.querySelector(".answer-body").innerHTML, h.context.renderMarkdown(raw.trim()));
  assert.equal(await h.tap(answer.querySelector("[data-copy-answer]")), true);
  assert.deepEqual(copied, [raw]);
  assert.equal(answer.dataset.canonicalIndex, "2", "history indexing remains on the answer wrapper");
});

test("partial and pending answers expose no copy control until the stream finishes", async () => {
  const copied = [];
  const h = streamHarness({ clipboard: { writeText: async (text) => { copied.push(text); } } });
  assert.equal(h.box.querySelectorAll("[data-copy-answer]").length, 0);
  h.emit("delta", { delta: "尚未完成" });
  h.emit("answer_pending");
  assert.equal(h.box.querySelectorAll("[data-copy-answer]").length, 0);
  h.emit("replace", { text: "最终结论\n第二行" });
  assert.equal(h.box.querySelectorAll("[data-copy-answer]").length, 0);
  h.emit("done");
  assert.equal(h.box.querySelectorAll("[data-copy-answer]").length, 1);
  assert.equal(await h.tap(h.box.querySelector("[data-copy-answer]")), true);
  assert.deepEqual(copied, ["最终结论\n第二行"]);
});

test("sealed answer rounds copy independently while the next round is still streaming", async () => {
  const copied = [];
  const h = streamHarness({ clipboard: { writeText: async (text) => { copied.push(text); } } });
  h.emit("delta", { delta: "工具前的结论\n" });
  h.seal();
  const first = h.box.children[0];
  h.emit("delta", { delta: "工具后的结论\n" });
  assert.equal(h.box.querySelectorAll("[data-copy-answer]").length, 1);
  assert.equal(await h.tap(first.querySelector("[data-copy-answer]")), true);
  h.emit("done");
  assert.equal(h.box.querySelectorAll("[data-copy-answer]").length, 2);
  assert.equal(h.box.children[0], first, "completion preserves answer order and nodes");
  assert.equal(await h.tap(h.box.children[1].querySelector("[data-copy-answer]")), true);
  assert.deepEqual(copied, ["工具前的结论\n", "工具后的结论\n"]);
});

test("recovered partial answers do not expose a copy control", () => {
  const h = harness();
  vm.runInContext(between("function recoveredProgressView(", "function handleReplayEvent("), h.context);
  const view = h.context.recoveredProgressView();
  view.delta("未完成的回答\n");
  view.replace("仍在输出");
  assert.equal(view.box.querySelectorAll("[data-copy-answer]").length, 0);
});

for (const phone of [true, false]) {
  test((phone ? "phone" : "desktop") + " final rendering keeps throughput outside copied answer text", async () => {
    const copied = [];
    const h = streamHarness({ phone, clipboard: { writeText: async (text) => copied.push(text) } });
    h.emit("delta", { delta: "provisional answer" });
    // app_stream's phone generation tests also keep one styled rate surface visible.
    h.emit("rate", { phase: "output", tokens: 40, elapsed_ms: 2000 });
    assert.equal(h.box.querySelectorAll(".turn-rate").length, 1);
    const raw = "Final answer\nNext line";
    h.emit("replace", { text: raw });
    h.emit("done");
    const footers = h.box.querySelectorAll(".turn-rate");
    assert.equal(footers.length, 1, "final markdown rendering removed or duplicated throughput");
    const footer = footers[0], answer = footer.parentNode;
    assert.ok(answer.classList.contains("msg") && answer.classList.contains("asst"));
    assert.equal(footer.textContent, "↳ speed · output 40 tok @ 20 tok/s");
    assert.doesNotMatch(answer.querySelector(".answer-body").textContent, /speed|tok\/s/);
    assert.equal(h.box.querySelectorAll("[data-copy-answer]").length, 1);
    assert.equal(await h.tap(answer.querySelector("[data-copy-answer]")), true);
    assert.deepEqual(copied, [raw], "the new rate footer leaked into copied markdown");
    answer.querySelector(".answer-body").renderedText = raw;
    h.context.answer = answer;
    vm.runInContext("answerCopyTexts.delete(answer);", h.context);
    assert.equal(await h.tap(answer.querySelector("[data-copy-answer]")), true);
    assert.deepEqual(copied, [raw, raw], "rendered-text copy included throughput");
  });
}
