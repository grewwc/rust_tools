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
    replaceChild(next, current) {
      const index = this.children.indexOf(current);
      assert.ok(index >= 0, "the replaced node is a child");
      next.remove();
      this.children[index] = next;
      current.parentNode = null;
      next.parentNode = this;
      return current;
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
  vm.runInContext(between("function handleReplayEvent(", "function showEventWarning("), h.context);
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
  vm.runInContext(between("// BEGIN execution event dispatcher", "// END execution event dispatcher"), h.context);
  vm.runInContext(between("function renderHistoryRow(", "// A session the server reports"), h.context);
  const raw = "\n# 结论\n\n完整的回答\n\n";
  h.context.renderHistoryRow({ message: { role: "assistant", content: raw }, text: raw.trim(), historyIndex: 2 }, h.msgs);
  const answer = h.msgs.lastElementChild;
  assert.equal(answer.querySelector(".answer-body").innerHTML, h.context.renderMarkdown(raw.trim()));
  assert.equal(await h.tap(answer.querySelector("[data-copy-answer]")), true);
  assert.deepEqual(copied, [raw]);
  assert.equal(answer.dataset.canonicalIndex, "2", "history indexing remains on the answer wrapper");
});

test("timestamp parsing uses storage seconds or explicit wire times and never invents missing values", () => {
  const h = harness(), c = h.context;
  const instant = new Date(2026, 10, 13, 9, 5).getTime();
  assert.equal(c.messageTimestampOf({ created_at: instant / 1000, timestamp: instant + 60000 }), instant);
  assert.equal(c.messageTimestampOf({ timestamp: instant }), instant);
  assert.equal(c.messageTimestampOf({ timestamp: new Date(instant).toISOString() }), instant);
  for (const source of [null, {}, [], { timestamp: "" }, { timestamp: false }, { timestamp: "09:05" },
    { timestamp: "2026-11-13T09:05:00" }, { timestamp: "2026-02-30T09:05:00Z" },
    { created_at: Infinity }, { timestamp: NaN }, { timestamp: 1e20 }]) {
    assert.equal(c.messageTimestampOf(source), null);
  }
  assert.equal(c.messageTimestampOf({ created_at: "bad", timestamp: instant }), instant);
  assert.equal(c.formatMessageTime(instant, new Date(2026, 10, 13, 23, 59).getTime()), "09:05");
  assert.equal(c.formatMessageTime(instant, new Date(2026, 10, 14).getTime()), "11/13 09:05");
  assert.equal(c.formatMessageTime(instant, new Date(2027, 0, 1).getTime()), "2026/11/13 09:05");
  assert.equal(c.formatMessageTime(null, instant), "");
});

for (const phone of [false, true]) test(`history user and answer timestamps stay outside body and copy text (phone=${phone})`, async () => {
  const copied = [], h = harness({ phone, clipboard: { writeText: async (text) => { copied.push(text); } } });
  const instant = new Date(2026, 10, 13, 9, 5).getTime();
  h.context.textOf = (content) => content;
  vm.runInContext(between("// BEGIN execution event dispatcher", "// END execution event dispatcher"), h.context);
  vm.runInContext(between("function attachUserRewind(", "function findUserHistoryIndex("), h.context);
  vm.runInContext(between("function renderHistoryRow(", "// A session the server reports"), h.context);
  const user = { role: "user", content: "问题\n原文", created_at: instant / 1000 };
  const answer = { role: "assistant", content: "\n# 最终结论\n", timestamp: instant + 60000 };
  for (const [index, message] of [user, answer].entries()) {
    h.context.renderHistoryRow({ message, text: message.content.trim(), historyIndex: index }, h.msgs);
  }
  const userBubble = h.msgs.querySelector(".msg.user"), answerBubble = h.msgs.querySelector(".msg.md");
  assert.equal(userBubble.dataset.question, user.content);
  assert.equal(userBubble.dataset.historyIndex, "0");
  assert.equal(userBubble.querySelectorAll(".message-time").length, 1);
  assert.equal(userBubble.querySelector(".message-time").getAttribute("datetime"), new Date(instant).toISOString());
  assert.equal(answerBubble.querySelectorAll(".message-time").length, 1);
  assert.equal(answerBubble.querySelector(".message-time").parentNode, answerBubble.querySelector(".answer-actions"));
  assert.equal(answerBubble.querySelector(".answer-body").querySelector(".message-time"), null);
  h.context.setMessageTimestamp(answerBubble, { timestamp: instant + 120000 });
  assert.equal(answerBubble.querySelectorAll(".message-time").length, 1, "metadata updates cannot duplicate the label");
  assert.equal(answerBubble.querySelectorAll("[data-copy-answer]").length, 1);
  assert.equal(await h.tap(answerBubble.querySelector("[data-copy-answer]")), true);
  assert.deepEqual(copied, [answer.content]);
  h.context.answerBubble = answerBubble;
  vm.runInContext("answerCopyTexts.delete(answerBubble)", h.context);
  await h.tap(answerBubble.querySelector("[data-copy-answer]"));
  assert.deepEqual(copied, [answer.content, answerBubble.querySelector(".answer-body").innerText]);
});

test("missing message metadata leaves existing user and assistant layouts unchanged", () => {
  const h = harness();
  const user = h.context.addMsg("user", "问题", "user", h.msgs, {});
  const answer = h.context.addMsgMd("回答", h.msgs, "回答", {});
  const tool = h.context.addMsg("tool", "结果", "asst", h.msgs, { timestamp: Date.now() });
  const pending = h.context.addMsgMd("", h.msgs, "", { timestamp: Date.now() });
  assert.equal(h.msgs.querySelectorAll(".message-time").length, 0);
  assert.equal(user.innerHTML, "问题");
  assert.equal(answer.children.length, 2);
  assert.equal(tool.querySelector(".message-time"), null);
  assert.equal(pending.querySelector(".answer-actions"), null);
});

test("timestamp metadata refreshes history render keys without changing fold or message identity", () => {
  const h = harness();
  h.context.textOf = (content) => content;
  h.context.isRuntimeSyntheticUser = () => false;
  vm.runInContext(between("function historyRowsOf(", "function historyExecutionOf("), h.context);
  const before = { role: "assistant", content: "最终回答" };
  const after = { ...before, created_at: 1700000000 };
  const first = h.context.historyRowsOf([before], 1, true).rows[0];
  const second = h.context.historyRowsOf([after], 1, true).rows[0];
  assert.notEqual(first.key, second.key, "new metadata cannot be lost by retaining the old DOM row");
  assert.equal(first.identity, second.identity, "timestamps must not change stored disclosure identities");
  assert.equal(first.historyIndex, second.historyIndex);
  assert.equal(first.text, second.text);
});

test("live timestamp metadata appears once only on the finished answer and resets between segments", async () => {
  const copied = [], h = streamHarness({ clipboard: { writeText: async (text) => { copied.push(text); } } });
  const instant = new Date(2026, 10, 13, 9, 5).getTime();
  h.emit("delta", { delta: "前序正文", timestamp: instant });
  assert.equal(h.box.querySelectorAll(".message-time").length, 0);
  h.seal();
  h.emit("delta", { delta: "最终", timestamp: instant + 60000 });
  h.emit("replace", { text: "最终回答", timestamp: instant + 120000 });
  assert.equal(h.box.querySelectorAll(".message-time").length, 0);
  h.emit("done", { timestamp: instant + 180000 });
  const final = h.box.querySelectorAll(".msg.md").at(-1);
  assert.equal(h.box.querySelectorAll(".message-time").length, 1);
  assert.equal(final.querySelector(".message-time").getAttribute("datetime"), new Date(instant + 180000).toISOString());
  await h.tap(final.querySelector("[data-copy-answer]"));
  assert.deepEqual(copied, ["最终回答"]);
});

for (const phone of [false, true]) test(`canonical done metadata timestamps the matching live input and final answer (phone=${phone})`, async () => {
  const copied = [], h = streamHarness({ phone, clipboard: { writeText: async (text) => { copied.push(text); } } });
  const row = h.document.createElement("div");
  h.msgs.appendChild(row);
  const user = h.context.addMsg("user", "同一个问题", "user", row);
  h.context.liveEntry.userInput = { row };
  h.emit("delta", { delta: "最终回答" });
  h.emit("replace", { text: "最终回答" });
  assert.equal(h.msgs.querySelectorAll(".message-time").length, 0);
  h.emit("done", { message_id: 14, created_at: 1700000004, user_message: { message_id: 13, created_at: 1700000003 } });
  const answer = h.box.querySelectorAll(".msg.md").at(-1);
  assert.equal(user.querySelectorAll(".message-time").length, 1);
  assert.equal(answer.querySelectorAll(".message-time").length, 1);
  assert.equal(user.querySelector(".message-time").getAttribute("datetime"), "2023-11-14T22:13:23.000Z");
  assert.equal(answer.querySelector(".message-time").getAttribute("datetime"), "2023-11-14T22:13:24.000Z");
  await h.tap(answer.querySelector("[data-copy-answer]"));
  assert.deepEqual(copied, ["最终回答"]);
});

test("timezone-qualified wire timestamps align with canonical Unix seconds", () => {
  const h = harness();
  const c = h.context;
  const ms = Date.parse("2026-11-13T01:05:00Z");
  assert.equal(c.messageTimestampOf({ timestamp: "2026-11-13T09:05:00+08:00" }), ms);
  assert.equal(c.messageTimestampOf({ created_at: ms / 1000 }), ms);
  assert.equal(c.formatMessageTime(ms, ms), String(new Date(ms).getHours()).padStart(2, "0") + ":05");
});

test("timestamp styles use secondary text in normal flow on both layouts", () => {
  assert.match(html, /\.message-time \{[^}]*color:var\(--dim\);[^}]*font-size:11px;[^}]*font-variant-numeric:tabular-nums/);
  assert.match(html, /\.msg\.user > \.message-time \{[^}]*margin-top:4px;[^}]*color:inherit/);
  assert.match(html, /\.answer-actions > \.message-time \{ margin-right:auto;/);
  const phone = html.slice(html.indexOf("@media (max-width:700px)"));
  assert.match(phone, /\.message-time \{ font-size:12px;/);
  assert.doesNotMatch(html.match(/\.message-time \{([^}]*)\}/)[1], /(?:^|;)\s*(?:position|height):/);
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
