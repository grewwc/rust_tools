// Run with: env -u NODE_OPTIONS node --test src/bin/ai/serve/app_commands_desktop.test.cjs
// Execute the real completion helpers and composer events in a small DOM.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function between(from, to) {
  const start = script.indexOf(from), end = script.indexOf(to, start);
  assert.ok(start >= 0 && end > start, "missing source slice: " + from);
  return script.slice(start, end);
}
const commands = [
  { name: "audit", usage: "[-f|--fast] [--deep] [instruction]", description: "Audit code" },
  { name: "archive", usage: "[<id>|current|last]", description: "Archive a session" },
  { name: "skills", usage: "[name]...", description: "Select skills" },
  { name: "agent", usage: "[name]", description: "Select agent" },
  { name: "use", usage: "[backend] [model]", description: "Select backend" },
  { name: "cd", usage: "[dir]", description: "Change directory" },
  { name: "git", usage: "[commit|checkout|branch|help]", description: "Git checkpoints" },
  { name: "help", usage: "[commands|shortcuts]", description: "Help" },
  { name: "proc", usage: "[list|kill <id>]", description: "Processes" },
  { name: "task", usage: "[N] <goal> | status", description: "Tasks" },
];
function node(tag = "div") {
  const handlers = new Map();
  const element = {
    tagName: tag, hidden: false, children: [], attrs: {}, dataset: {}, value: "",
    appendChild(child) { this.children.push(child); return child; },
    append(...children) { this.children.push(...children); },
    setAttribute(name, value) { this.attrs[name] = String(value); },
    getAttribute(name) { return this.attrs[name] ?? null; },
    removeAttribute(name) { delete this.attrs[name]; },
    contains(other) { return this === other || this.children.some((child) => child.contains(other)); },
    querySelectorAll(selector) {
      assert.equal(selector, '[role="option"]');
      return this.children.filter((child) => child.attrs.role === "option");
    },
    scrollIntoView() {},
    addEventListener(type, callback) {
      if (!handlers.has(type)) handlers.set(type, []);
      handlers.get(type).push(callback);
    },
    dispatch(type, fields = {}) {
      const event = { target: this, preventDefault() { this.prevented = true; }, ...fields };
      for (const callback of handlers.get(type) || []) callback(event);
      return event;
    },
  };
  Object.defineProperty(element, "textContent", {
    get() { return (this._text || "") + this.children.map((child) => child.textContent).join(""); },
    set(value) { this._text = String(value); this.children = []; },
  });
  return element;
}
function harness({ width = 701, catalog = commands, media = "known", respond } = {}) {
  const prompt = node("textarea"), elements = { prompt, cmdmenu: node(), cmdoptions: node(), cmdempty: node() };
  elements.cmdmenu.hidden = true;
  elements.cmdmenu.appendChild(elements.cmdoptions);
  elements.cmdmenu.appendChild(elements.cmdempty);
  for (const [id, names] of [["omodel", ["model-one", "model-two"]], ["oagent", ["build", "review"]], ["oeffort", ["high", "xhigh"]]]) {
    elements[id] = node("select");
    elements[id].options = ["", ...names].map((value) => ({ value, textContent: value }));
    Object.defineProperty(elements[id], "selectedIndex", { get() { return this.options.findIndex((option) => option.value === this.value); } });
  }
  for (const id of ["omodelbtn", "omodelbtnlabel", "modelpick", "modelpicklist", "modelpicksearch", "modelpickempty", "modelpicksearchclear"]) elements[id] = node();
  elements.modelpick.hidden = true;
  const document = node("document");
  document.activeElement = prompt;
  const focusable = (element) => { element.focus = () => { document.activeElement = element; }; return element; };
  Object.values(elements).forEach(focusable);
  document.createElement = (tag) => focusable(node(tag));
  const window = node("window"), mq = node("media");
  Object.defineProperty(mq, "matches", { get: () => width >= 701 });
  if (media === "known") window.matchMedia = (query) => {
    assert.equal(query, "(min-width:701px)");
    return mq;
  };
  else if (media === "throws") window.matchMedia = () => { throw new Error("unsupported"); };
  const calls = [], effects = [], timers = [];
  prompt.selectionStart = prompt.selectionEnd = 0;
  prompt.setSelectionRange = (start, end) => {
    effects.push(["selection", start, end]);
    prompt.selectionStart = start; prompt.selectionEnd = end;
  };
  prompt.setRangeText = (text, start, end) => {
    prompt.value = prompt.value.slice(0, start) + text + prompt.value.slice(end);
    prompt.selectionStart = prompt.selectionEnd = start + text.length;
  };
  const context = vm.createContext({
    $, document, window, promptComposing: false, composingText: "", sessionId: "s-one",
    store: { base: "http://server-one", token: "" },
    skillCatalog: [{ name: "review" }, { name: "release" }, { name: "中文🚀" }],
    lastItems: [{ id: "s-one" }, { id: "s-two" }],
    TEAM_FIELDS: { backend: { kernel: "teamkernel", traecli: "teamtraecli" } },
    workerModels: null, workerModelsBase: "", workerModelsState: "idle",
    api: async (url, opts) => {
      calls.push([url, opts]);
      const body = url === "/commands" ? catalog : respond ? await respond(url) :
        url === "/subagent-models" ? { models: ["worker-one", "worker-two"] } : { candidates: ["src/", "scripts/"] };
      return { json: async () => body };
    },
    setPromptValue(value) { if (context.promptComposing) return false; effects.push(["value", value]); prompt.value = value; return true; },
    growPrompt() { effects.push(["grow"]); }, scheduleGrow() {}, persistDraft() { effects.push(["draft", prompt.value]); },
    traceMark() {}, promptHistory: { capture() {}, record() {}, recordInput() {}, replay() {}, beforeInput() {} },
    queueImageFiles(files) { effects.push(["images", files]); }, restoreDraft() {},
    setTimeout(fn) { timers.push(fn); }, isTurnRunning: () => false,
    send() { effects.push(["send"]); }, sendSideNote() { effects.push(["side-note"]); },
    pauseHistoryRetry() {}, abortEvents() {}, updateExecutionModel() {}, syncStopButton() {},
    resetQuestionNavigator() {}, restoreSkillPick() {}, restoreTeamPick() {},
    historyLoadSeq: 0, ensureTranscriptOwner() {},
    syncPickTrigger() {}, syncOvrSummary() {}, note(text) { effects.push(["note", text]); },
    openPickPop(kind) { effects.push(["picker", kind]); },
    localStorage: { setItem() {}, removeItem() {} },
  });
  function $(id) { return elements[id] || null; }
  vm.runInContext(between("async function persistPicks()", "function fillSelect("), context);
  vm.runInContext(between("let modelSheetReturnFocus", "// Skills are pending next-turn choices"), context);
  vm.runInContext(between("// Slash-command completion.", "async function loadInfo()"), context);
  if (typeof context.wireDesktopCommandCompletion === "function") context.wireDesktopCommandCompletion();
  vm.runInContext(between("function insertPromptText(", "// --- native phone actions"), context);
  vm.runInContext(between("const IMAGE_EXTENSION_MIMES =", "function renderComposerAttachments("), context);
  vm.runInContext(between("function handlePromptPaste(", "function textOf("), context);
  vm.runInContext(between('$("prompt").addEventListener("input"', '$("prompt").addEventListener("click"'), context);
  const pointer = script.indexOf('  document.addEventListener("pointerdown", (e) => {\n    if (!$("cmdmenu")');
  vm.runInContext(script.slice(script.indexOf('$("prompt").addEventListener("click"'), script.indexOf("\n  });", pointer) + 7), context);
  vm.runInContext(between("// BEGIN session switch", "// END session switch"), context);
  const type = (value, start = value.length, end = start) => {
    prompt.value = value; prompt.selectionStart = start; prompt.selectionEnd = end;
    prompt.dispatch("input");
  };
  return {
    context, prompt, elements, document, window, calls, effects, type,
    key: (key, fields) => prompt.dispatch("keydown", { key, ...fields }),
    state: () => vm.runInContext("desktopCompletion.session", context),
    resize(next) { width = next; mq.dispatch("change", { matches: width >= 701 }); window.dispatch("resize"); },
    flush() { while (timers.length) timers.shift()(); },
  };
}
async function ready(options) { const h = harness(options); await h.context.loadCommands(); return h; }
async function settle() { await new Promise((resolve) => setImmediate(resolve)); }
const values = (h) => h.elements.cmdoptions.children.map((row) => row.textContent);

test("desktop single slash candidate stays open and repeated Tab is idempotent", async () => {
  const h = await ready({ catalog: [commands[0]] }); h.type("/");
  assert.equal(h.key("Tab").prevented, true); assert.equal(h.prompt.value, "/audit ");
  assert.equal(h.elements.cmdmenu.hidden, false);
  h.key("Tab"); assert.equal(h.prompt.value, "/audit ");
});
test("desktop Tab cycles all matches and wraps using the original range", async () => {
  const h = await ready({ catalog: commands.slice(0, 2) }); h.type("/");
  h.key("Tab"); assert.equal(h.prompt.value, "/audit ");
  h.key("Tab"); assert.equal(h.prompt.value, "/archive ");
  assert.equal(h.prompt.attrs["aria-activedescendant"], "cmdopt-1");
  h.key("Tab"); assert.equal(h.prompt.value, "/audit ");
  assert.equal(h.state().original, "/");
  assert.equal(h.state().context.end, 1);
});
test("first desktop Tab starts at the first candidate after arrow navigation", async () => {
  const h = await ready(); h.type("/a"); h.key("ArrowDown"); h.key("Tab");
  assert.equal(h.prompt.value, "/audit ");
});
test("Enter confirms a cycle without submitting or inserting a newline", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab"); h.key("Tab");
  assert.equal(h.key("Enter").prevented, true); assert.equal(h.prompt.value, "/archive ");
  assert.equal(h.elements.cmdmenu.hidden, true); assert.equal(h.state(), null);
  assert.equal(h.effects.some(([kind]) => kind === "send"), false);
});
test("point selection confirms another candidate from the frozen snapshot", async () => {
  const h = await ready(); h.type("/a tail", 2); h.key("Tab");
  h.elements.cmdoptions.children[1].onclick();
  assert.equal(h.prompt.value, "/archive tail"); assert.equal(h.state(), null);
  assert.equal(h.elements.cmdmenu.hidden, true);
});
test("Escape restores original text and selection and ends the cycle", async () => {
  const h = await ready(); h.type("/au tail", 2, 3); h.key("Tab");
  assert.equal(h.key("Escape").prevented, true); assert.equal(h.prompt.value, "/au tail");
  assert.equal(h.prompt.selectionStart, 2); assert.equal(h.prompt.selectionEnd, 3);
  assert.equal(h.state(), null); assert.equal(h.elements.cmdmenu.hidden, true);
});
test("Enter and point selection also accept candidates before cycling", async () => {
  const h = await ready(); h.type("/a"); h.elements.cmdoptions.children[1].onclick();
  assert.equal(h.prompt.value, "/archive ");
  h.type("/a"); h.key("Enter"); assert.equal(h.prompt.value, "/audit ");
});
test("skills parameters cycle names and preserve completed parameters", async () => {
  const h = await ready(); h.type("/skills re");
  assert.match(values(h)[0], /review/); h.key("Tab"); assert.equal(h.prompt.value, "/skills review ");
  h.key("Tab"); assert.equal(h.prompt.value, "/skills release ");
  h.type("/skills review re"); h.key("Tab"); assert.equal(h.prompt.value, "/skills review release ");
});
test("agent parameters come from info select values", async () => {
  const h = await ready(); h.type("/agent r"); h.key("Tab");
  assert.equal(h.prompt.value, "/agent review ");
});
test("use completes registered backends and remote worker models", async () => {
  const h = await ready(); h.type("/use tr"); h.key("Tab");
  assert.equal(h.prompt.value, "/use traecli ");
  h.type("/use traecli worker-"); await settle(); h.key("Tab");
  assert.equal(h.prompt.value, "/use traecli worker-one ");
  h.key("Tab"); assert.equal(h.prompt.value, "/use traecli worker-two ");
  assert.equal(h.calls.filter(([url]) => url === "/subagent-models").length, 1);
});
test("a name-only model command usage uses info model values", async () => {
  const h = await ready({ catalog: [{ name: "use", usage: "[name]" }] });
  h.type("/use model-"); h.key("Tab"); assert.equal(h.prompt.value, "/use model-one ");
});
test("static usage alternatives and optional flags respect the argument position", async () => {
  const h = await ready(); h.type("/git ch"); h.key("Tab"); assert.equal(h.prompt.value, "/git checkout ");
  h.type("/audit --"); h.key("Tab"); assert.equal(h.prompt.value, "/audit --fast ");
  h.key("Tab"); assert.equal(h.prompt.value, "/audit --deep ");
  h.type("/audit -f --"); h.key("Tab"); assert.equal(h.prompt.value, "/audit -f --deep ");
  h.type("/proc "); assert.deepEqual(values(h), ["list", "kill"]);
  h.type("/proc kill "); assert.equal(h.elements.cmdoptions.children.length, 0);
  h.type("/task sta"); h.key("Tab"); assert.equal(h.prompt.value, "/task status ");
});
test("static grammar preserves top-level alternatives and angle-bracket metavars", async () => {
  const h = await ready();
  assert.deepEqual(Array.from(h.context.desktopStaticArguments("[N] <goal> | status", [], 0)), ["status"]);
  assert.deepEqual(Array.from(h.context.desktopStaticArguments("[list|current|use <name>]", [], 0)), ["list", "current", "use"]);
  assert.deepEqual(Array.from(h.context.desktopStaticArguments("[list|current|use <name>]", ["use"], 1)), []);
  assert.deepEqual(Array.from(h.context.desktopStaticArguments("<number|id-prefix|id>", [], 0)), []);
});
test("parameter sources never add commands outside GET /commands", async () => {
  const h = await ready({ catalog: [commands[0]] }); h.type("/skills re");
  assert.equal(h.elements.cmdmenu.hidden, true); assert.equal(h.key("Tab").prevented, undefined);
  h.type("/cd s"); await settle(); assert.equal(h.calls.length, 1);
});
test("cd fetches session candidates once and cycles directory replacements", async () => {
  const h = await ready(); h.type("/cd s"); await settle();
  assert.equal(h.calls[1][0], "/sessions/s-one/cd-candidates?frag=s");
  h.key("Tab"); assert.equal(h.prompt.value, "/cd src/");
  h.key("Tab"); assert.equal(h.prompt.value, "/cd scripts/");
  h.key("Tab"); assert.equal(h.prompt.value, "/cd src/"); assert.equal(h.calls.length, 2);
});
test("cd handles spaces and UTF-16 fragments without touching later lines", async () => {
  const h = await ready({ respond: () => ({ candidates: ["中文🚀 folder/", "中文🚀 files/"] }) });
  h.type("intro🚀\n/cd 中文🚀 f\nnext🚀", "intro🚀\n/cd 中文🚀 f".length); await settle(); h.key("Tab");
  assert.equal(h.prompt.value, "intro🚀\n/cd 中文🚀 folder/\nnext🚀");
  assert.equal(h.prompt.selectionStart, "intro🚀\n/cd 中文🚀 folder/".length);
  assert.match(h.calls[1][0], /frag=%E4%B8%AD%E6%96%87%F0%9F%9A%80%20f$/);
});
for (const width of [320, 700]) test(width + "px retains the old accept-and-close behavior and parameters", async () => {
  const h = await ready({ width }); h.type("/a"); h.key("Tab");
  assert.equal(h.prompt.value, "/audit "); assert.equal(h.elements.cmdmenu.hidden, true);
  assert.equal(h.key("Tab").prevented, undefined);
  h.type("/skills re"); assert.equal(h.elements.cmdmenu.hidden, true);
  h.type("\n/a"); assert.equal(h.elements.cmdmenu.hidden, true);
});
for (const media of ["missing", "throws"]) test("unknown media environment " + media + " retains the old path", async () => {
  const h = await ready({ media }); h.type("/a"); h.key("Tab");
  assert.equal(h.elements.cmdmenu.hidden, true); assert.equal(h.prompt.value, "/audit ");
});
test("shrinking from 701 to 700 clears the desktop cycle without changing its text", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab"); h.resize(700);
  assert.equal(h.state(), null); assert.equal(h.elements.cmdmenu.hidden, true);
  assert.equal(h.prompt.value, "/audit "); assert.equal(h.key("Tab").prevented, undefined);
  h.resize(701); h.type("/a"); h.key("Tab"); h.key("Tab"); assert.equal(h.prompt.value, "/archive ");
});
test("shrinking before the first Tab clears desktop candidate objects before the phone path", async () => {
  const h = await ready(); h.type("/a"); h.resize(700);
  assert.equal(h.elements.cmdmenu.hidden, true); assert.equal(h.key("Tab").prevented, undefined);
  assert.equal(h.prompt.value, "/a"); h.type("/a"); h.key("Tab"); assert.equal(h.prompt.value, "/audit ");
});
test("a stale desktop menu after direct server or value changes cannot insert old candidates", async () => {
  const h = await ready(); h.type("/a"); h.context.store.base = "http://server-two";
  assert.equal(h.key("Tab").prevented, undefined); assert.equal(h.prompt.value, "/a");
  h.type("/a"); h.prompt.value = "/arch"; h.prompt.selectionStart = h.prompt.selectionEnd = 5;
  assert.equal(h.key("Enter").prevented, undefined); assert.equal(h.prompt.value, "/arch");
});
for (const guard of ["promptComposing", "isComposing", "keyCode"]) test("IME " + guard + " guards keydown before any mutation", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab"); const before = h.effects.length;
  if (guard === "promptComposing") h.context.promptComposing = true;
  for (const key of ["Tab", "Enter", "Escape", "ArrowDown"]) {
    const event = h.key(key, guard === "isComposing" ? { isComposing: true } : guard === "keyCode" ? { keyCode: 229 } : {});
    assert.equal(event.prevented, undefined);
  }
  assert.equal(h.effects.length, before); assert.equal(h.prompt.value, "/audit ");
});
test("multiline commands preserve leading text and middle-caret suffix", async () => {
  const h = await ready(); const head = "中文🚀\n  /a";
  h.type(head + "rchive tail🚀\nnext", head.length); h.key("Tab");
  assert.equal(h.prompt.value, "中文🚀\n  /audit tail🚀\nnext"); h.key("Tab");
  assert.equal(h.prompt.value, "中文🚀\n  /archive tail🚀\nnext");
});
test("noncollapsed parameter selection replaces only its token with correct UTF-16 caret", async () => {
  const h = await ready(); const start = "中文🚀\n/skills ".length;
  h.type("中文🚀\n/skills 中文🚀 tail", start, start + "中文🚀".length); h.key("Tab");
  assert.equal(h.prompt.value, "中文🚀\n/skills review tail");
  assert.equal(h.prompt.selectionStart, start + "review".length);
  h.key("Escape"); assert.equal(h.prompt.selectionEnd, start + "中文🚀".length);
});
test("a selection spanning tokens or lines and ordinary paths preserve native Tab", async () => {
  const h = await ready();
  for (const [text, start, end] of [["/a tail", 1, 5], ["/a\nnext", 1, 4], ["/tmp/file", 4, 4], ["hello /a", 8, 8], ["/agent   build", 7, 9]]) {
    h.type(text, start, end); assert.equal(h.key("Tab").prevented, undefined); assert.equal(h.prompt.value, text);
  }
});
test("empty, hidden and zero-candidate menus preserve browser Tab", async () => {
  const h = await ready();
  for (const text of ["", "ordinary", "/zzz", "/skills unavailable"]) {
    h.type(text); assert.equal(h.key("Tab").prevented, undefined); assert.equal(h.prompt.value, text);
  }
  h.type("/a"); h.context.closeCommandMenu(); assert.equal(h.key("Tab").prevented, undefined);
});
for (const flag of ["ctrlKey", "metaKey", "altKey", "shiftKey"]) test(flag + "+Tab preserves browser focus navigation", async () => {
  const h = await ready(); h.type("/a"); assert.equal(h.key("Tab", { [flag]: true }).prevented, undefined);
  assert.equal(h.prompt.value, "/a");
});
test("ordinary editing invalidates the cycle even when input restores identical text", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab"); h.prompt.dispatch("input");
  assert.equal(h.state(), null); h.key("Escape"); assert.equal(h.prompt.value, "/audit ");
});
test("programmatic selection events preserve a cycle and user caret movement invalidates it", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab"); h.prompt.dispatch("select"); h.document.dispatch("selectionchange");
  assert.ok(h.state()); h.prompt.selectionStart = h.prompt.selectionEnd = 1; h.document.dispatch("selectionchange");
  assert.equal(h.state(), null); assert.equal(h.elements.cmdmenu.hidden, true);
});
test("cursor keys, click and blur end the cycle", async () => {
  const h = await ready();
  for (const mode of ["ArrowLeft", "Home", "click", "blur"]) {
    h.type("/a"); h.key("Tab");
    if (mode === "click" || mode === "blur") h.prompt.dispatch(mode); else h.key(mode);
    assert.equal(h.state(), null, mode);
  }
});
test("the actual paste path invalidates text and image completion without requiring input", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab");
  const pasted = h.prompt.dispatch("paste", { clipboardData: { getData: () => "尾🚀", items: [], files: [] } });
  assert.equal(pasted.prevented, true); assert.equal(h.prompt.value, "/audit 尾🚀"); assert.equal(h.state(), null);
  h.type("/a"); h.key("Tab"); const image = { type: "image/png" };
  h.prompt.dispatch("paste", { clipboardData: { getData: () => "", items: [], files: [image] } });
  assert.equal(h.state(), null); assert.equal(h.prompt.value, "/audit ");
});
test("switchSession clears the cycle synchronously before restoring another draft", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab"); h.context.switchSession("s-two");
  assert.equal(h.state(), null); assert.equal(h.elements.cmdmenu.hidden, true); assert.equal(h.context.sessionId, "s-two");
});
test("server catalog reload and observed server change invalidate the cycle", async () => {
  const h = await ready(); h.type("/a"); h.key("Tab"); h.context.store.base = "http://server-two";
  assert.equal(h.key("Tab").prevented, undefined); assert.equal(h.state(), null);
  h.type("/a"); h.key("Tab"); await h.context.loadCommands(); assert.equal(h.state(), null);
});
test("slow directory responses cannot reopen a menu after edits, session change, blur, Escape or shrink", async () => {
  for (const action of ["edit", "switch", "blur", "escape", "shrink"]) {
    let resolve; const pending = new Promise((yes) => { resolve = yes; });
    const h = await ready({ respond: () => pending }); h.type("/cd s");
    if (action === "edit") h.type("ordinary");
    else if (action === "switch") h.context.switchSession("s-two");
    else if (action === "blur") { h.document.activeElement = h.document; h.prompt.dispatch("blur"); }
    else if (action === "escape") h.key("Escape");
    else h.resize(700);
    resolve({ candidates: ["stale/"] }); await settle();
    assert.equal(h.elements.cmdmenu.hidden, true, action); assert.equal(h.prompt.value.includes("stale"), false);
  }
});
test("pure replacement and cycle functions preserve UTF-16 indexes and wrap", async () => {
  const h = await ready();
  const context = h.context.desktopCommandContext("🚀\n/cd 文😀\nnext", 10, 10, commands);
  assert.ok(context); const plan = h.context.desktopReplacementPlan("🚀\n/cd 文😀\nnext", context, { display: "文件/", replacement: "文件/" });
  assert.equal(plan.value, "🚀\n/cd 文件/\nnext"); assert.equal(plan.caret, "🚀\n/cd 文件/".length);
  assert.equal(h.context.desktopCycleIndex(-1, 2), 0); assert.equal(h.context.desktopCycleIndex(1, 2), 0);
  assert.equal(h.context.desktopCycleIndex(0, 1), 0); assert.equal(h.context.desktopCycleIndex(0, 0), -1);
});
test("the full client remains a valid classic script", () => {
  assert.doesNotThrow(() => new vm.Script(script)); assert.doesNotMatch(html, /<script[^>]*type=["']module/);
});

const modelCatalog = [{ name: "subagent", usage: "[backend] [model]", description: "Set the subagent backend and model" }];
const configWrites = (h) => h.calls.filter(([url, options]) => url.endsWith("/config") && options?.method === "POST")
  .map(([url, options]) => [url, JSON.parse(options.body)]);
function assertModelPrompt(h, value, caret, hidden) {
  assert.equal(h.prompt.value, value);
  assert.equal(h.prompt.selectionStart, caret);
  assert.equal(h.prompt.selectionEnd, caret);
  assert.equal(h.elements.cmdmenu.hidden, hidden);
  assert.equal(h.effects.some(([kind]) => kind === "send" || kind === "side-note" || kind === "prepare-turn"), false);
  assert.equal(h.calls.some(([url]) => url.endsWith("/turn") || url.endsWith("/side-note")), false);
}
for (const word of ["mod", "mode", "model", "models"]) test("desktop /" + word + " prioritizes the local model command over description matches", async () => {
  const h = await ready({ catalog: modelCatalog }); h.type("/" + word);
  assert.match(values(h)[0], new RegExp("^/" + (word === "models" ? "models" : "model") + "(?: |$)"));
  assert.equal(h.elements.cmdempty.hidden, true);
  assert.equal(vm.runInContext("commandCatalog.some((command) => command.name === 'model' || command.name === 'models')", h.context), false);
  assert.equal(h.calls.length, 1);
});
for (const alias of ["model", "models"]) {
  for (const key of ["Tab", "Enter"]) test("desktop /" + alias + " direct " + key + " opens the existing model picker without a turn", async () => {
    const h = await ready({ catalog: modelCatalog }); h.type("/" + alias);
    assert.equal(h.key(key).prevented, true);
    assertModelPrompt(h, "/" + alias + " ", alias.length + 2, true);
    assert.equal(h.elements.modelpick.hidden, false);
    assert.equal(configWrites(h).length, 0);
    h.elements.modelpicklist.children[2].onclick();
    assert.equal(h.elements.omodel.value, "model-two");
    assert.equal(h.elements.modelpick.hidden, true);
    assert.deepEqual(configWrites(h), [["/sessions/s-one/config", { model: "model-two", agent: "", reasoning_effort: "" }]]);
  });
  for (const key of ["Tab", "Enter"]) test("desktop /" + alias + " parameter " + key + " commits the option value to the current session", async () => {
    const h = await ready({ catalog: modelCatalog });
    h.elements.omodel.options[1].textContent = "Model One label";
    h.elements.oagent.value = "review"; h.elements.oeffort.value = "high";
    h.type("/" + alias + " model-o"); assert.match(values(h)[0], /^Model One label/);
    assert.equal(h.key(key).prevented, true);
    assertModelPrompt(h, "/" + alias + " model-one ", alias.length + 12, key === "Enter");
    assert.equal(h.elements.omodel.value, "model-one");
    assert.deepEqual(configWrites(h), [["/sessions/s-one/config", { model: "model-one", agent: "review", reasoning_effort: "high" }]]);
  });
  test("desktop /" + alias + " parameters cycle, wrap, and Enter confirms the persisted model", async () => {
    const h = await ready({ catalog: modelCatalog }); h.type("/" + alias + " model-");
    h.key("Tab"); assertModelPrompt(h, "/" + alias + " model-one ", alias.length + 12, false);
    h.key("Tab"); assertModelPrompt(h, "/" + alias + " model-two ", alias.length + 12, false);
    assert.equal(h.elements.omodel.value, "model-two");
    h.key("Tab"); assert.equal(h.elements.omodel.value, "model-one");
    assert.equal(h.key("Enter").prevented, true);
    assertModelPrompt(h, "/" + alias + " model-one ", alias.length + 12, true);
    assert.equal(configWrites(h).length, 3, "confirming an unchanged model does not post it again");
  });
  test("desktop /" + alias + " offers existing model subcommands and effort values", async () => {
    const h = await ready({ catalog: modelCatalog }); h.type("/" + alias + " ");
    assert.deepEqual(values(h), ["list", "current", "help", "effort", "model-one", "model-two"]);
    h.type("/" + alias + " effort x"); h.key("Enter");
    assertModelPrompt(h, "/" + alias + " effort xhigh ", alias.length + 15, true);
    assert.equal(h.elements.oeffort.value, "xhigh");
    assert.deepEqual(configWrites(h), [["/sessions/s-one/config", { model: "", agent: "", reasoning_effort: "xhigh" }]]);
    h.type("/" + alias + " effort xhigh "); assert.equal(h.elements.cmdoptions.children.length, 0);
  });
}
test("desktop /mod cycles model aliases before Enter opens the picker", async () => {
  const h = await ready({ catalog: modelCatalog }); h.type("/mod");
  h.key("Tab"); assertModelPrompt(h, "/model ", 7, false);
  h.key("Tab"); assertModelPrompt(h, "/models ", 8, false);
  assert.equal(h.elements.modelpick.hidden, true);
  h.key("Enter"); assertModelPrompt(h, "/models ", 8, true);
  assert.equal(h.elements.modelpick.hidden, false);
});
for (const [input, caret, expected, expectedCaret, model] of [
  ["/models", 4, "/model ", 7, ""],
  ["/models model-two", 4, "/model model-two", 6, "model-two"],
  ["/models model-two", 7, "/models model-two", 7, "model-two"],
  ["/models model-two tail🚀", 7, "/models model-two tail🚀", 7, "model-two"],
  ["/models ", 7, "/models ", 7, ""],
  ["/models  ", 7, "/models  ", 7, ""],
  ["/models  model-o tail🚀", 16, "/models  model-one tail🚀", 18, "model-one"],
  ["中文🚀\n  /models model-o\nnext🚀", "中文🚀\n  /models model-o".length,
    "中文🚀\n  /models model-one\nnext🚀", "中文🚀\n  /models model-one".length, "model-one"],
]) for (const key of ["Tab", "Enter"]) test("desktop model " + key + " preserves the replacement contract: " + JSON.stringify([input, caret]), async () => {
  const h = await ready({ catalog: modelCatalog }); h.type(input, caret);
  assert.equal(h.key(key).prevented, true);
  const direct = caret === 7;
  assertModelPrompt(h, expected, expectedCaret, key === "Enter" || direct);
  assert.equal(h.elements.omodel.value, model);
  assert.equal(h.elements.modelpick.hidden, !!model || (key === "Tab" && !direct));
  assert.equal(configWrites(h).length, model ? 1 : 0);
});
for (const spaces of [" ", "  "]) for (const key of ["Tab", "Enter"]) test("desktop /models with " + spaces.length + " trailing spaces accepts a subcommand via " + key, async () => {
  const h = await ready({ catalog: modelCatalog }); h.type("/models" + spaces);
  assert.equal(h.key(key).prevented, true);
  const expected = "/models" + spaces + "list ";
  assertModelPrompt(h, expected, expected.length, key === "Enter");
  assert.equal(h.elements.modelpick.hidden, key === "Tab");
  assert.equal(configWrites(h).length, 0);
});
for (const subcommand of ["list", "current", "help", "effort"]) test("desktop model " + subcommand + " is handled locally on Enter", async () => {
  const h = await ready({ catalog: modelCatalog }); h.type("/models " + subcommand); h.key("Enter");
  assertModelPrompt(h, "/models " + subcommand + " ", 9 + subcommand.length, true);
  if (subcommand === "effort") assert.deepEqual(h.effects.filter(([kind]) => kind === "picker"), [["picker", "effort"]]);
  else assert.equal(h.elements.modelpick.hidden, false);
  assert.equal(configWrites(h).length, 0);
});
for (const width of [320, 700]) test(width + "px receives no local model commands or picker side effects", async () => {
  const h = await ready({ width, catalog: modelCatalog }); h.type("/mod"); h.key("Tab");
  assert.equal(h.prompt.value, "/subagent ");
  assert.equal(h.elements.cmdmenu.hidden, true);
  h.type("/models"); assert.equal(h.key("Tab").prevented, undefined);
  assertModelPrompt(h, "/models", 7, false);
  h.type("/models model-"); assert.equal(h.key("Enter").prevented, undefined);
  assertModelPrompt(h, "/models model-", 14, true);
  assert.equal(h.elements.modelpick.hidden, true); assert.equal(configWrites(h).length, 0);
});
test("desktop send dispatcher consumes model commands before preparing a chat turn", async () => {
  const h = await ready({ catalog: modelCatalog });
  Object.assign(h.context, {
    canSend: () => true, pendingImages: [], validateSkillInput: () => true, sendBusy: false,
    runCdCommand: async () => false, runGitCommand: async () => false,
    prepareTurnSession: async () => { h.effects.push(["prepare-turn"]); return ""; },
  });
  vm.runInContext(between("// BEGIN composer send", "// END composer send"), h.context);
  h.type("/models model-two"); await h.context.send();
  assert.equal(h.elements.omodel.value, "model-two");
  assert.deepEqual(configWrites(h), [["/sessions/s-one/config", { model: "model-two", agent: "", reasoning_effort: "" }]]);
  assertModelPrompt(h, "/models model-two", 17, true);
  h.type("/model unavailable"); await h.context.send();
  assert.equal(h.elements.omodel.value, "model-two");
  assert.equal(h.elements.modelpick.hidden, false);
  assert.equal(configWrites(h).length, 1);
  assertModelPrompt(h, "/model unavailable", 18, true);
  h.context.closeModelSheet(); h.prompt.focus();
  const multiline = "中文🚀\n/models model-one tail🚀\nnext";
  h.type(multiline); await h.context.send();
  assert.equal(h.elements.omodel.value, "model-one");
  assertModelPrompt(h, multiline, multiline.length, true);
  assert.equal(configWrites(h).length, 2);
});
for (const width of [320, 700]) test(width + "px send dispatcher keeps the original model-text path", async () => {
  const h = await ready({ width, catalog: modelCatalog });
  Object.assign(h.context, {
    canSend: () => true, pendingImages: [], validateSkillInput: () => true, sendBusy: false,
    runCdCommand: async () => false, runGitCommand: async () => false,
    prepareTurnSession: async () => { h.effects.push(["prepare-turn"]); return ""; },
  });
  vm.runInContext(between("// BEGIN composer send", "// END composer send"), h.context);
  h.type("/models model-two"); await h.context.send();
  assert.equal(h.effects.filter(([kind]) => kind === "prepare-turn").length, 1);
  assert.equal(h.elements.modelpick.hidden, true); assert.equal(configWrites(h).length, 0);
  assert.equal(h.prompt.value, "/models model-two");
});
