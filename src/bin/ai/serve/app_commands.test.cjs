// Run with: node --test src/bin/ai/serve/app_commands.test.cjs
// Execute the composer's slash-command completion block without a browser.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const start = script.indexOf("// Slash-command completion.");
const end = script.indexOf("async function loadInfo()");
assert.ok(start >= 0 && end > start, "command completion block is present");
const block = script.slice(start, end);

function node(tag = "div") {
  const element = {
    tagName: tag,
    id: "",
    className: "",
    type: "",
    hidden: false,
    children: [],
    attrs: {},
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    setAttribute(name, value) { this.attrs[name] = String(value); },
    removeAttribute(name) { delete this.attrs[name]; },
    contains(candidate) {
      if (candidate === this) return true;
      return this.children.some((child) => child === candidate || (child.contains && child.contains(candidate)));
    },
    scrollIntoView() {},
  };
  Object.defineProperty(element, "textContent", {
    get() { return (element._text || "") + element.children.map((child) => child.textContent).join(""); },
    set(value) { element._text = String(value); element.children = []; },
  });
  return element;
}

function harness(commands) {
  const prompt = node("textarea");
  prompt.value = "";
  prompt.selectionStart = 0;
  prompt.selectionEnd = 0;
  prompt.setSelectionRange = (startAt, endAt) => {
    prompt.selectionStart = startAt;
    prompt.selectionEnd = endAt;
  };
  const elements = {
    prompt,
    cmdmenu: node("div"),
    cmdoptions: node("div"),
    cmdempty: node("div"),
  };
  elements.cmdmenu.hidden = true;
  elements.cmdempty.hidden = true;
  const document = {
    activeElement: prompt,
    createElement: (tag) => node(tag),
  };
  const context = {
    promptComposing: false,
    store: { base: "http://serve" },
    api: async () => ({ json: async () => commands }),
    $: (id) => elements[id],
    document,
    setPromptValue: (value) => { prompt.value = value; return true; },
    growPrompt() {},
    persistDraft() {},
  };
  vm.createContext(context);
  vm.runInContext(block, context);
  const type = (value, caret = value.length) => {
    prompt.value = value;
    prompt.selectionStart = caret;
    prompt.selectionEnd = caret;
    context.refreshCommandMenu();
  };
  return { context, prompt, elements, document, type };
}

const catalog = [
  { name: "audit", usage: "[instruction]", description: "Run the audit subagent" },
  { name: "archive", usage: "[id]", description: "Toggle the session archive flag" },
  { name: "cd", usage: "[dir]", description: "Change the working directory" },
];

test("commandContext only accepts a first-token command word", () => {
  const { context } = harness(catalog);
  const fragment = context.commandContext("/au", 3);
  assert.equal(fragment.start, 0);
  assert.equal(fragment.end, 3);
  assert.equal(fragment.prefix, "/");
  assert.equal(fragment.query, "au");
  assert.equal(context.commandContext("  /audit", 8).start, 2);
  assert.equal(context.commandContext("hello /audit", 12), null);
  assert.equal(context.commandContext("/tmp/foo", 8), null);
  assert.equal(context.commandContext("/audit rest", 6).end, 6);
  assert.equal(context.commandContext("/audit rest", 9), null);
  assert.equal(context.commandContext("\n/audit", 7), null);
  assert.equal(context.commandContext(":au", 3).prefix, ":");
  assert.equal(context.commandContext("/", 1).query, "");
});

test("normalizeCommands strips prefixes and rejects malformed entries", () => {
  const { context } = harness(catalog);
  const out = context.normalizeCommands([
    { name: "/audit", usage: "x", description: "d" },
    { name: "audit", usage: "duplicate", description: "d" },
    { name: "bad/name" },
    { name: "cd" },
    { name: 42 },
    null,
  ]);
  assert.deepEqual(Array.from(out, (item) => item.name), ["audit", "cd"]);
});

test("loadCommands opens and filters the menu from the server catalog", async () => {
  const h = harness(catalog);
  await h.context.loadCommands();
  h.type("/a");
  assert.equal(h.elements.cmdmenu.hidden, false);
  assert.equal(h.elements.cmdempty.hidden, true);
  assert.deepEqual(Array.from(h.elements.cmdoptions.children, (row) => row.id), ["cmdopt-0", "cmdopt-1"]);
  assert.match(h.elements.cmdoptions.children[0].textContent, /^\/audit/);
  assert.match(h.elements.cmdoptions.children[1].textContent, /^\/archive/);
  assert.equal(h.prompt.attrs["aria-expanded"], "true");
  assert.equal(h.prompt.attrs["aria-activedescendant"], "cmdopt-0");
});

test("Tab and Enter accept the active command", async () => {
  const h = harness(catalog);
  await h.context.loadCommands();
  h.type("/a");
  const tab = { key: "Tab", preventDefault() { this.prevented = true; } };
  assert.equal(h.context.commandKeydown(tab), true);
  assert.equal(tab.prevented, true);
  assert.equal(h.prompt.value, "/audit ");
  assert.equal(h.prompt.selectionStart, 7);
  assert.equal(h.elements.cmdmenu.hidden, true);
  h.type("/arch");
  const enter = { key: "Enter", preventDefault() { this.prevented = true; } };
  assert.equal(h.context.commandKeydown(enter), true);
  assert.equal(enter.prevented, true);
  assert.equal(h.prompt.value, "/archive ");
});

test("Arrow keys move the active option and wrap", async () => {
  const h = harness(catalog);
  await h.context.loadCommands();
  h.type("/a");
  assert.equal(h.elements.cmdoptions.children[0].attrs["data-active"], "true");
  assert.equal(h.context.commandKeydown({ key: "ArrowDown", preventDefault() {} }), true);
  assert.equal(h.elements.cmdoptions.children[0].attrs["data-active"], "false");
  assert.equal(h.elements.cmdoptions.children[1].attrs["data-active"], "true");
  assert.equal(h.prompt.attrs["aria-activedescendant"], "cmdopt-1");
  h.context.commandKeydown({ key: "ArrowDown", preventDefault() {} });
  assert.equal(h.elements.cmdoptions.children[0].attrs["data-active"], "true");
  h.context.commandKeydown({ key: "ArrowUp", preventDefault() {} });
  assert.equal(h.elements.cmdoptions.children[1].attrs["data-active"], "true");
});

test("Escape closes and ctrl-enter stays available for send", async () => {
  const h = harness(catalog);
  await h.context.loadCommands();
  h.type("/a");
  const escape = { key: "Escape", preventDefault() { this.prevented = true; } };
  assert.equal(h.context.commandKeydown(escape), true);
  assert.equal(escape.prevented, true);
  assert.equal(h.elements.cmdmenu.hidden, true);
  h.type("/a");
  const ctrlEnter = { key: "Enter", ctrlKey: true, preventDefault() { this.prevented = true; } };
  assert.equal(h.context.commandKeydown(ctrlEnter), false);
  assert.equal(ctrlEnter.prevented, undefined);
});

test("non-command text and paths do not open the menu", async () => {
  const h = harness(catalog);
  await h.context.loadCommands();
  h.type("hello /aud");
  assert.equal(h.elements.cmdmenu.hidden, true);
  h.type("/tmp/foo");
  assert.equal(h.elements.cmdmenu.hidden, true);
  h.type("  /audit rest", 8);
  assert.equal(h.elements.cmdmenu.hidden, false);
  h.type("  /audit rest", 12);
  assert.equal(h.elements.cmdmenu.hidden, true);
});

test("an unknown command word shows the empty state", async () => {
  const h = harness(catalog);
  await h.context.loadCommands();
  h.type("/zzz");
  assert.equal(h.elements.cmdmenu.hidden, false);
  assert.equal(h.elements.cmdempty.hidden, false);
  assert.equal(h.elements.cmdoptions.children.length, 0);
  const enter = { key: "Enter", preventDefault() { this.prevented = true; } };
  assert.equal(h.context.commandKeydown(enter), false);
  assert.equal(enter.prevented, undefined);
});

test("accepting preserves text after the command", async () => {
  const h = harness(catalog);
  await h.context.loadCommands();
  h.type("/aud rest", 4);
  assert.equal(h.elements.cmdmenu.hidden, false);
  assert.equal(h.context.acceptCommand(0), true);
  assert.equal(h.prompt.value, "/audit rest");
  assert.equal(h.prompt.selectionStart, 6);
});

test("a failed catalog load closes completion", async () => {
  const h = harness(catalog);
  h.context.api = async () => { throw new Error("offline"); };
  h.type("/a");
  assert.equal(h.elements.cmdmenu.hidden, false, "the empty state is visible before loading");
  await h.context.loadCommands();
  assert.equal(h.elements.cmdmenu.hidden, true);
});

test("commandScore ranks prefix before substring and description", () => {
  const { context } = harness(catalog);
  const audit = { name: "audit", description: "Run the audit subagent" };
  const task = { name: "task", description: "Run a multi-worker task room" };
  assert.equal(context.commandScore(audit, "aud"), 0);
  assert.equal(context.commandScore(audit, "dit"), 1);
  assert.equal(context.commandScore(audit, "subagent"), 2);
  assert.equal(context.commandScore(task, "zzz"), -1);
  assert.equal(context.commandScore(audit, "a"), 0);
  assert.equal(context.commandScore(task, "a"), -1);
});

test("command menu disables animation for reduced motion", () => {
  assert.match(
    html,
    /@media \(prefers-reduced-motion:reduce\) \{ #cmdmenu \{ animation:none; \} \}/,
  );
});
