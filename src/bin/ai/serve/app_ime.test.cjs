// Run with: node --test src/bin/ai/serve/app_ime.test.cjs
// IME contract of the embedded client: nothing the page does on its own may
// write the composer's value or resize it while the user is assembling a
// word. On Android WebView both cancel the composition, so the pinyin the
// user typed never reaches the field and the keyboard stops committing.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// The composer helpers, in source order: the composition flag, the guarded
// value write, the focus probe and the resize.
const imeSrc = script.slice(
  script.indexOf("let promptComposing = false;"),
  script.indexOf("function insertPromptText("),
);
const restoreSrc = script.slice(
  script.indexOf("function restoreDraft("),
  script.indexOf("function clearDraft("),
);
const loadSessionsSrc = script.slice(
  script.indexOf("async function loadSessions()"),
  script.indexOf("// History is append-only"),
);
const traceSrc = script.slice(
  script.indexOf("function createInputTrace("),
  script.indexOf("let uiWired = false;"),
);

function makeContext(extra) {
  const prompt = {
    value: "",
    style: { height: "44px" },
    scrollHeight: 120,
    selectionStart: 0,
    selectionEnd: 0,
    setRangeText(text, start, end) {
      this.value = this.value.slice(0, start) + text + this.value.slice(end);
      this.selectionStart = this.selectionEnd = start + text.length;
    },
  };
  const document = { activeElement: null };
  const context = vm.createContext(Object.assign({
    document,
    window: { matchMedia: () => ({ matches: false }), innerHeight: 800 },
    requestAnimationFrame: (fn) => { context.frames.push(fn); return context.frames.length; },
    localStorage: {
      store: new Map(),
      getItem(k) { return this.store.has(k) ? this.store.get(k) : null; },
      setItem(k, v) { this.store.set(k, v); },
      removeItem(k) { this.store.delete(k); },
    },
    $: (id) => (id === "prompt" ? prompt : null),
    draftKey: (id) => "draft:" + id,
  }, extra || {}));
  context.frames = [];
  context.prompt = prompt;
  context.marks = [];
  vm.runInContext(imeSrc, context);
  return context;
}

function run(context, code) {
  return vm.runInContext(code, context);
}

test("the composer is not resized while a word is being composed", () => {
  const context = makeContext();
  context.prompt.style.height = "44px";
  run(context, "promptComposing = true;");
  run(context, "growPrompt();");
  assert.equal(context.prompt.style.height, "44px", "the composer was resized mid-composition");
  // Once the word is committed the composer grows again.
  run(context, "promptComposing = false;");
  context.prompt.value = "hello";
  run(context, "growPrompt();");
  assert.notEqual(context.prompt.style.height, "44px");
});

test("the composer's value is not written while a word is being composed", () => {
  const context = makeContext();
  context.prompt.value = "ni";
  run(context, "promptComposing = true;");
  assert.equal(run(context, "setPromptValue('overwritten')"), false);
  assert.equal(context.prompt.value, "ni", "the value was written mid-composition");
  run(context, "promptComposing = false;");
  assert.equal(run(context, "setPromptValue('written')"), true);
  assert.equal(context.prompt.value, "written");
});

test("a draft restore does not overwrite the text being typed", () => {
  const context = makeContext();
  vm.runInContext(restoreSrc, context);
  context.localStorage.store.set("draft:old", "stored draft");
  context.prompt.value = "正在输入";
  run(context, "promptComposing = true;");
  run(context, "restoreDraft('old');");
  assert.equal(context.prompt.value, "正在输入", "the restore replaced the text being typed");
  // With the composer free the draft is restored as before.
  run(context, "promptComposing = false;");
  run(context, "restoreDraft('old');");
  assert.equal(context.prompt.value, "stored draft");
});

test("an automatic session switch does not replace the composer's text", async () => {
  const switched = [];
  const notes = [];
  const context = makeContext({
    api: async () => ({ json: async () => [{ id: "new", title: "new" }] }),
    switchSession: (id) => { switched.push(id); },
    note: (text) => { notes.push(text); },
    renderSessionList: () => {},
    renderSessionBar: () => {},
    pollConfirm: () => {},
    loadHistory: async () => {},
    loadSessionConfig: async () => {},
    resetPicks: () => {},
  });
  vm.runInContext(loadSessionsSrc, context);
  vm.runInContext("let lastItems = []; let pendingNew = ''; let sessionId = 'gone'; let turnRunning = false; let sessFilter = '';", context);

  // The composer has focus: the user is typing, so no automatic switch.
  context.document.activeElement = context.prompt;
  await run(context, "loadSessions()");
  assert.deepEqual(switched, [], "the page switched sessions while the composer was in use");
  assert.equal(notes.length, 1);

  // With the composer idle the switch still happens.
  context.document.activeElement = null;
  await run(context, "loadSessions()");
  assert.deepEqual(switched, ["new"]);
});

test("programmatic writes and resizes are marked for the input trace", () => {
  const context = makeContext();
  run(context, "traceMark = (what) => { marks.push(what); };");
  run(context, "promptComposing = true;");
  run(context, "setPromptValue('x');");
  assert.deepEqual(context.marks, [], "a refused write was still marked");
  run(context, "promptComposing = false;");
  run(context, "setPromptValue('x');");
  run(context, "growPrompt();");
  assert.ok(context.marks.includes("valueWrite"), "the value write was not marked");
  assert.ok(context.marks.includes("resize"), "the resize was not marked");
});

test("the resize after a composition is deferred out of the IME commit", () => {
  const context = makeContext();
  context.prompt.value = "你好";
  run(context, "scheduleGrow();");
  assert.equal(context.frames.length, 1, "the resize was not deferred to a frame");
  const before = context.prompt.style.height;
  context.frames[0]();
  assert.notEqual(context.prompt.style.height, before, "the deferred resize did not run");
});

test("the composer box is not rewritten when its height does not change", () => {
  const context = makeContext();
  run(context, "traceMark = (what) => { marks.push(what); };");
  context.prompt.value = "hello";
  run(context, "growPrompt();");
  const height = context.prompt.style.height;
  assert.equal(height, "120px");
  const marks = context.marks.length;
  // Another character on the same line does not change the content height, so
  // the box must not be written again: a layout change on the focused field
  // is what turns into a lost Chinese word.
  context.prompt.value = "hello!";
  run(context, "growPrompt();");
  assert.equal(context.prompt.style.height, height, "the composer box was rewritten without a height change");
  assert.equal(context.marks.length, marks, "a no-op resize was still marked");
});

test("the composer box gives its height back when the text is deleted", () => {
  const context = makeContext();
  context.prompt.value = "hello";
  run(context, "growPrompt();");
  assert.equal(context.prompt.style.height, "120px");
  context.prompt.scrollHeight = 40;
  context.prompt.value = "hi";
  run(context, "growPrompt();");
  assert.equal(context.prompt.style.height, "40px", "the composer box kept the height of the deleted line");
});

test("the input trace reports whether the keyboard ever composed a word", () => {
  const prompt = { value: "", style: { height: "44px" } };
  const context = vm.createContext({
    document: { activeElement: prompt },
    window: { innerHeight: 600 },
    navigator: { userAgent: "WebView-test" },
  });
  vm.runInContext(traceSrc, context);

  // Latin letters committed directly: the keyboard never composed, so the
  // report has to say so instead of leaving it to guesswork.
  const direct = context.createInputTrace(prompt);
  direct.record({ type: "input", inputType: "insertText", isComposing: false });
  const directReport = JSON.parse(direct.export());
  assert.equal(directReport.ime.compositionEvents, 0);
  assert.equal(directReport.ime.directInputs, 1);
  assert.match(directReport.ime.hint, /direct \(latin\) input mode/);

  // A real pinyin session announces itself with composition events.
  const composed = context.createInputTrace(prompt);
  composed.record({ type: "compositionstart" });
  composed.record({ type: "input", inputType: "insertCompositionText", isComposing: true });
  composed.record({ type: "compositionend" });
  const composedReport = JSON.parse(composed.export());
  assert.equal(composedReport.ime.compositionEvents, 2);
  assert.equal(composedReport.ime.compositionInputs, 1);
  assert.equal(composedReport.ime.hint, "");

  // The buffer stays bounded, and clearing it isolates a fresh reproduction.
  for (let i = 0; i < 400; i++) composed.record({ type: "input", inputType: "insertText" });
  assert.equal(JSON.parse(composed.export()).events.length, 300);
  composed.clear();
  assert.equal(JSON.parse(composed.export()).events.length, 0);
});
