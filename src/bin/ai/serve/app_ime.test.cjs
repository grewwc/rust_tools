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
// The commit path: the input handler that carries a committed word, the
// composition handler that ends a word, and the turn state that resizes the
// composer around the focused field.
const inputHandlerSrc = script.slice(
  script.indexOf('$("prompt").addEventListener("input"'),
  script.indexOf('$("prompt").addEventListener("paste"'),
);
const compositionEndSrc = script.slice(
  script.indexOf('$("prompt").addEventListener("compositionend"'),
  script.indexOf('$("prompt").addEventListener("input"'),
);
const syncStopSrc = script.slice(
  script.indexOf("function syncStopButton() {"),
  script.indexOf("// Transient note through the existing #connstat line"),
);
const noteSrc = script.slice(
  script.indexOf("// Transient note through the existing #connstat line"),
  script.indexOf("async function sendSideNote("),
);

function makeContext(extra) {
  const listeners = new Map();
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
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    dispatch(type, event) {
      for (const handler of listeners.get(type) || []) handler(event || {});
    },
    getBoundingClientRect() { return { top: 0, left: 0, width: 320, height: 44 }; },
  };
  const document = { activeElement: null };
  const context = vm.createContext(Object.assign({
    document,
    window: { matchMedia: () => ({ matches: false }), innerHeight: 800 },
    requestAnimationFrame: (fn) => { context.frames.push(fn); return context.frames.length; },
    setTimeout: (fn) => { context.timers.push(fn); return context.timers.length; },
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
  context.timers = [];
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

test("native field-sizing engines never get a scripted height write", () => {
  const context = makeContext({ CSS: { supports: (_prop, _value) => true } });
  context.prompt.value = "multi\nline\ntext";
  run(context, "traceMark = (what) => { marks.push(what); };");
  run(context, "growPrompt();");
  assert.equal(context.prompt.style.height, "44px", "the focused field's geometry was rewritten");
  assert.deepEqual(context.marks, [], "a resize mark fired on a native-sizing engine");
});

test("the input trace reports DOM observations without inferring keyboard mode", () => {
  const prompt = { value: "", style: { height: "44px" } };
  const context = vm.createContext({
    document: { activeElement: prompt },
    window: { innerHeight: 600 },
    navigator: { userAgent: "WebView-test" },
  });
  vm.runInContext(traceSrc, context);

  // insertText can carry Chinese without DOM composition events; it cannot
  // identify an input language or a native keyboard mode.
  const direct = context.createInputTrace(prompt);
  prompt.value = "中文";
  direct.record({ type: "input", inputType: "insertText", isComposing: false });
  const directReport = JSON.parse(direct.export());
  assert.equal(directReport.ime.compositionEvents, 0);
  assert.equal(directReport.ime.directInputs, 1);
  assert.equal(directReport.ime.hint,
    "no DOM composition events observed; insertText alone does not identify the input language or keyboard mode");
  assert.ok(!direct.export().includes("中文"), "diagnostics leaked the input text");

  // Recorded composition events remain visible in the summary.
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
  const clearedReport = JSON.parse(composed.export());
  assert.equal(clearedReport.events.length, 0);
  assert.deepEqual(clearedReport.ime,
    { compositionEvents: 0, compositionInputs: 0, directInputs: 0, hint: "" });
  composed.record({ type: "input", inputType: "insertText" });
  assert.deepEqual(JSON.parse(composed.export()).ime, directReport.ime);
});

test("the committed word is not resized inside the commit task", () => {
  const context = makeContext({ persistDraft: () => {} });
  vm.runInContext(inputHandlerSrc, context);
  context.prompt.value = "你好";
  context.prompt.scrollHeight = 200;
  context.prompt.style.height = "44px";
  // An IME word arrives as an input event. Resizing the focused field inside
  // that task is what makes the keyboard drop the word.
  context.prompt.dispatch("input", { isComposing: false, inputType: "insertText" });
  assert.equal(context.prompt.style.height, "44px", "the composer was resized inside the commit task");
  assert.equal(context.frames.length, 1, "the resize was not deferred to a frame");
  context.frames[0]();
  assert.notEqual(context.prompt.style.height, "44px", "the deferred resize did not run");
});

test("a layout change waits for the composition to end", () => {
  const context = makeContext();
  run(context, "promptComposing = true;");
  run(context, "holdLayout('sideNote', () => { marks.push('sideNote'); });");
  assert.deepEqual(context.marks, [], "the page changed while a word was being composed");
  run(context, "promptComposing = false;");
  run(context, "flushLayout();");
  assert.deepEqual(context.marks, ["sideNote"], "the held change was not applied afterwards");
});

test("flushing layout cannot bypass an active composition", () => {
  const context = makeContext();
  run(context, "promptComposing = true; holdLayout('status', () => marks.push('status'));");
  run(context, "flushLayout();");
  assert.deepEqual(context.marks, [], "a direct flush bypassed the composition guard");
  run(context, "promptComposing = false; flushLayout();");
  assert.deepEqual(context.marks, ["status"], "the deferred update was discarded");
});

test("a previous commit callback preserves work queued during the next composition", () => {
  const context = makeContext({ persistDraft: () => {}, note: () => {} });
  vm.runInContext(script.slice(
    script.indexOf('$("prompt").addEventListener("compositionstart"'),
    script.indexOf('$("prompt").addEventListener("input"'),
  ), context);
  context.document.activeElement = context.prompt;
  context.prompt.dispatch("compositionstart");
  context.prompt.value = "你";
  context.prompt.dispatch("compositionend", { data: "你" });
  const previousCommit = context.timers.shift();

  context.prompt.dispatch("compositionstart");
  run(context, "pendingShare = ' share'; pendingAppend = ' image';");
  run(context, "holdLayout('status', () => marks.push('old'));");
  run(context, "holdLayout('status', () => marks.push('latest'));");
  previousCommit();
  assert.deepEqual(context.marks, [], "the earlier commit changed the next composition's layout");
  assert.equal(context.prompt.value, "你");
  assert.equal(run(context, "pendingShare"), " share", "queued share was lost");
  assert.equal(run(context, "pendingAppend"), " image", "queued attachment was lost");

  context.prompt.value = "你好";
  context.prompt.dispatch("compositionend", { data: "好" });
  context.timers.shift()();
  assert.deepEqual(context.marks, ["latest"]);
  assert.equal(context.prompt.value, "你好 share image");
  assert.equal(run(context, "pendingShare + pendingAppend"), "");
});

test("a transient note does not move the composer during a composition", () => {
  const classes = new Set();
  const connstat = {
    textContent: "",
    classList: {
      add(...names) { names.forEach((name) => classes.add(name)); },
      remove(...names) { names.forEach((name) => classes.delete(name)); },
    },
  };
  const context = makeContext({
    $: (id) => (id === "connstat" ? connstat : null),
  });
  run(context, "let noteTimer = null;");
  vm.runInContext(noteSrc, context);
  run(context, "promptComposing = true;");
  run(context, "note('working');");
  assert.equal(connstat.textContent, "", "the note appeared while a word was being composed");
  assert.equal(context.timers.length, 0, "the note timer started while a word was being composed");
  run(context, "promptComposing = false;");
  run(context, "flushLayout();");
  assert.equal(connstat.textContent, "working");
  assert.ok(classes.has("show") && classes.has("note"), "the held note was not shown");
  run(context, "promptComposing = true;");
  context.timers.shift()();
  assert.ok(classes.has("show"), "the note hid while a word was being composed");
  run(context, "promptComposing = false;");
  run(context, "flushLayout();");
  assert.ok(!classes.has("show"), "the note stayed after its timer");
});

test("the side note column does not resize the composer during a composition", () => {
  const toggles = [];
  const composer = { classList: { toggle: (name, on) => toggles.push("composer." + name + "=" + on) } };
  const sidenote = {
    classList: { toggle: (name, on) => toggles.push("sidenote." + name + "=" + on) },
    disabled: false,
    setAttribute() {},
    tabIndex: -1,
  };
  const send = { textContent: "", classList: { toggle() {} }, setAttribute() {}, title: "" };
  const status = { textContent: "", hidden: true };
  const context = makeContext({
    $: (id) => ({ send, sidenote, composer, turnstatus: status })[id] || null,
    isTurnRunning: () => true,
    note: () => {},
  });
  vm.runInContext(
    "let sideNoteBusy = false; let eventActive = false; let eventSession = '';" +
    "let sessionId = 's1'; let turnRunning = false; let turnRunningAt = 0;" +
    "const RUNNING_SAMPLE_TTL_MS = 1000; const turns = new Set();",
    context,
  );
  vm.runInContext(syncStopSrc, context);
  run(context, "promptComposing = true;");
  vm.runInContext("syncStopButton();", context);
  assert.deepEqual(toggles, [], "the composer grid changed during the composition");
  run(context, "promptComposing = false;");
  vm.runInContext("flushLayout();", context);
  assert.ok(toggles.includes("composer.side-note-ready=true"), "the side note column was not applied afterwards");
});

test("a dropped composition puts the word back instead of losing it", () => {
  const notes = [];
  const context = makeContext({
    persistDraft: () => {},
    note: (text) => { notes.push(text); },
  });
  vm.runInContext(compositionEndSrc, context);
  context.document.activeElement = context.prompt;
  context.prompt.value = "";
  run(context, "promptComposing = true; composingText = \"pu'sh\"; composingStartValue = '';");
  // The browser ends a composition it dropped with an empty payload: the word
  // the user assembled never reached the field.
  context.prompt.dispatch("compositionend", { data: "" });
  assert.equal(context.prompt.value, "pu'sh", "the word being composed was lost");
  assert.equal(notes.length, 0, "the note changed layout inside the commit task");
  assert.ok(context.timers.length >= 1, "the deferred composition work was not scheduled");
  context.timers.shift()();
  assert.equal(notes.length, 1, "the drop was not reported");
  assert.match(notes[0], /输入被打断/);
});

test("a committed word is not put back into the field", () => {
  const context = makeContext({ persistDraft: () => {}, note: () => {} });
  vm.runInContext(compositionEndSrc, context);
  context.document.activeElement = context.prompt;
  context.prompt.value = "朴实";
  run(context, "promptComposing = true; composingText = \"pu'sh\"; composingStartValue = '';");
  context.prompt.dispatch("compositionend", { data: "朴实" });
  assert.equal(context.prompt.value, "朴实", "the committed word was replaced");
});
