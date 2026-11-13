// Run with: env -u NODE_OPTIONS node --test --test-reporter=tap src/bin/ai/serve/app_phase3_pure_boundaries.test.cjs
// Optional ASERVE_PHASE3_BASELINE compares the same observed input sequences with a saved page.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const appFile = process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html");
const html = fs.readFileSync(appFile, "utf8");
const baseline = process.env.ASERVE_PHASE3_BASELINE
  ? fs.readFileSync(process.env.ASERVE_PHASE3_BASELINE, "utf8") : null;
const plain = (value) => JSON.parse(JSON.stringify(value));

function loadFixtures(file, names, source = html) {
  const filename = path.join(__dirname, file);
  const fixtureVm = { ...vm };
  const fixtureFs = {
    ...fs,
    readFileSync(file, ...args) {
      if (String(file) === path.join(__dirname, "app.html") || String(file) === appFile) {
        return args[0] ? source : Buffer.from(source);
      }
      return fs.readFileSync(file, ...args);
    },
  };
  return new vm.Script(fs.readFileSync(filename, "utf8") +
    "\n;({" + names.join(",") + "});", { filename }).runInNewContext({
    require: (name) => name === "node:test" ? { test() {} } : name === "node:fs" ? fixtureFs
      : name === "node:vm" ? fixtureVm : require(name),
    __dirname, __filename: filename, console, Buffer,
    process: { env: { ...process.env, ASERVE_TEST_APP_HTML: appFile } },
    TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
    setTimeout, clearTimeout, setInterval, clearInterval, setImmediate, queueMicrotask,
  });
}

function domSnapshot(elements, document) {
  const node = (element) => {
    const result = {};
    for (const key of ["tagName", "id", "className", "type", "hidden", "disabled", "value",
      "selectionStart", "selectionEnd", "tabIndex", "title", "textContent", "attrs", "focusOptions",
      "dataset", "style", "scrollTop", "scrollHeight", "clientHeight", "open", "isConnected", "_taskResult"]) {
      if (element[key] !== undefined) result[key] = element[key];
    }
    if (element.children) result.children = Array.from(element.children, node);
    return result;
  };
  return plain({
    elements: Object.fromEntries(Object.entries(elements).map(([id, element]) => [id, node(element)])),
    focus: Object.keys(elements).find((id) => elements[id] === document?.activeElement) || null,
  });
}

const commandCatalog = [
  { name: "description", description: "Audit instructions" },
  { name: "zaudit", description: "" },
  { name: "audit", description: "" },
  { name: "auditor", description: "" },
  { name: "cd", description: "" },
];
const commandNames = (h) => Array.from(vm.runInContext("commandMatches", h.context), (item) => item.name);
const commandSnapshot = (h, trace = [], result = null) => plain({
  dom: domSnapshot(h.elements, h.document), names: commandNames(h),
  active: vm.runInContext("commandActive", h.context), trace, result,
});
async function commands(source) {
  const h = loadFixtures("app_commands.test.cjs", ["harness"], source).harness(commandCatalog);
  await h.context.loadCommands();
  return h;
}

async function completionRanking(source) {
  const h = await commands(source), snapshots = [];
  for (const [value, expected] of [
    ["/aud", ["audit", "auditor", "zaudit", "description"]],
    ["/a", ["audit", "auditor"]],
    ["/dit", ["zaudit", "audit", "auditor", "description"]],
    ["/", ["description", "zaudit", "audit", "auditor", "cd"]],
    ["/zzz", []],
  ]) {
    h.type(value);
    assert.deepEqual(commandNames(h), expected);
    assert.equal(vm.runInContext("commandActive", h.context), expected.length ? 0 : -1);
    snapshots.push(commandSnapshot(h));
  }
  h.type(":aud");
  h.context.setCommandActive(3);
  snapshots.push(commandSnapshot(h));
  h.context.refreshCommandMenu();
  assert.equal(vm.runInContext("commandActive", h.context), 0, "refresh always resets the command active row");
  snapshots.push(commandSnapshot(h));
  h.context.promptComposing = true;
  for (const key of ["ArrowDown", "ArrowUp", "Tab", "Enter", "Escape"]) {
    const event = { key, preventDefault() { this.prevented = true; } };
    assert.equal(h.context.commandKeydown(event), false);
    assert.equal(event.prevented, undefined);
    snapshots.push(commandSnapshot(h));
  }
  h.context.refreshCommandMenu();
  assert.equal(h.elements.cmdmenu.hidden, true);
  snapshots.push(commandSnapshot(h));
  return snapshots;
}

const replacementCases = [
  { value: "/au", caret: 3, output: "/audit ", next: 7 },
  { value: "  :au rest", caret: 5, output: "  :audit rest", next: 8 },
  { value: "/audition tail", caret: 4, output: "/audit tail", next: 6 },
  { value: "\t/au\r\nrest", caret: 4, output: "\t/audit\r\nrest", next: 7 },
  { value: "/au\targ", caret: 3, output: "/audit\targ", next: 6 },
  { value: "/au\n中🙂", caret: 3, output: "/audit\n中🙂", next: 6 },
  { value: " \r/au", caret: 5, output: " \r/audit ", next: 9 },
];
async function completionReplacement(source) {
  const snapshots = [];
  for (const scenario of replacementCases) {
    const h = await commands(source), trace = [], c = h.context;
    h.type(scenario.value, scenario.caret);
    const setValue = c.setPromptValue, select = h.prompt.setSelectionRange, close = c.closeCommandMenu;
    c.setPromptValue = (value) => { trace.push(["value", value]); return setValue(value); };
    h.prompt.setSelectionRange = (a, b) => { trace.push(["caret", a, b]); select(a, b); };
    c.closeCommandMenu = () => { trace.push(["close"]); close(); };
    c.growPrompt = () => trace.push(["grow"]);
    c.persistDraft = () => trace.push(["persist"]);
    const result = c.acceptCommand({ name: "audit" });
    assert.equal(result, true);
    assert.equal(h.prompt.value, scenario.output);
    assert.equal(h.prompt.selectionStart, scenario.next);
    assert.deepEqual(trace, [["value", scenario.output], ["caret", scenario.next, scenario.next], ["close"], ["grow"], ["persist"]]);
    snapshots.push(commandSnapshot(h, trace, result));
  }
  return snapshots;
}

async function completionExceptions(source) {
  const snapshots = [];
  for (const branch of ["refused", "value throws", "selection throws", "grow throws", "invalid context", "missing target"]) {
    const h = await commands(source), trace = [], c = h.context;
    h.type("/au"); const close = c.closeCommandMenu;
    c.closeCommandMenu = () => { trace.push(["close"]); close(); };
    c.setPromptValue = (value) => {
      trace.push(["value", value]);
      if (branch === "value throws") throw new Error("value failure");
      if (branch === "refused") return false;
      h.prompt.value = value; return true;
    };
    h.prompt.setSelectionRange = (a, b) => {
      trace.push(["caret", a, b]);
      if (branch === "selection throws") throw new Error("selection failure");
      h.prompt.selectionStart = a; h.prompt.selectionEnd = b;
    };
    c.growPrompt = () => { trace.push(["grow"]); if (branch === "grow throws") throw new Error("grow failure"); };
    c.persistDraft = () => trace.push(["persist"]);
    if (branch === "invalid context") h.prompt.value = "hello /au";
    let result;
    try { result = c.acceptCommand(branch === "missing target" ? 99 : { name: "audit" }); }
    catch (error) { result = error.message; }
    const expected = {
      refused: { result: false, trace: [["value", "/audit "]] },
      "value throws": { result: "value failure", trace: [["value", "/audit "]] },
      "selection throws": { result: true, trace: [["value", "/audit "], ["caret", 7, 7], ["close"], ["grow"], ["persist"]] },
      "grow throws": { result: "grow failure", trace: [["value", "/audit "], ["caret", 7, 7], ["close"], ["grow"]] },
      "invalid context": { result: false, trace: [["close"]] },
      "missing target": { result: false, trace: [] },
    }[branch];
    assert.deepEqual({ result, trace }, expected);
    assert.equal(h.elements.cmdmenu.hidden, ["selection throws", "grow throws", "invalid context"].includes(branch));
    snapshots.push(commandSnapshot(h, trace, result));
  }
  return snapshots;
}

async function skillRankingAndSelection(source) {
  const catalog = [
    { name: "build", description: "Audito help" },
    { name: "auditor", description: "" },
    { name: "audit_own_changes", description: "" },
    { name: "Auditor-extra", description: "" },
    { name: "auditor-long", description: "" },
  ];
  const h = loadFixtures("app_skills.test.cjs", ["harness"], source).harness({ catalog });
  await h.context.loadSkills(); h.open();
  const snapshots = [], capture = () => snapshots.push(plain({
    dom: domSnapshot(h.elements, h.context.document), names: h.names(), selected: h.selected(),
    active: vm.runInContext("skillActive", h.context), storage: Array.from(h.storage),
  }));
  assert.deepEqual(Array.from(h.names()), catalog.map((skill) => skill.name)); capture();
  h.context.setSkillActive(3);
  h.elements.skillsearch.value = " AUD ";
  h.context.renderSkillMenu();
  assert.deepEqual(Array.from(h.names()), ["auditor", "audit_own_changes", "Auditor-extra", "auditor-long", "build"]);
  assert.equal(vm.runInContext("skillActive", h.context), 2, "render retains the skill name, independent of command reset");
  capture();
  h.choose("audit_own_changes"); h.choose("auditor"); h.choose("audit_own_changes"); h.choose("audit_own_changes");
  assert.deepEqual(Array.from(h.selected()), ["auditor", "audit_own_changes"]); capture();
  h.type("audito");
  assert.deepEqual(Array.from(h.names()), ["auditor", "Auditor-extra", "auditor-long", "audit_own_changes", "build"]);
  assert.equal(vm.runInContext("skillActive", h.context), -1, "typing resets the skill active row");
  capture();
  h.elements.skillsearch.oncompositionstart(); h.type("审", true);
  for (const key of ["ArrowDown", "ArrowUp", "Home", "End", "Enter", "Escape"]) {
    assert.equal(h.key(key).prevented, undefined); capture();
  }
  h.elements.skillsearch.oncompositionend(); h.type("not-found");
  assert.deepEqual(Array.from(h.names()), []);
  assert.deepEqual(Array.from(h.selected()), ["auditor", "audit_own_changes"]); capture();
  return snapshots;
}

async function skillLateAcceptance(source) {
  const h = loadFixtures("app_skills.test.cjs", ["harness"], source).harness();
  await h.context.loadSkills();
  const snapshots = [], capture = () => snapshots.push(plain({
    dom: domSnapshot(h.elements, h.context.document), selected: h.selected(),
    storage: Array.from(h.storage), requests: h.requests,
  }));
  let resolve;
  h.context.fetch = () => new Promise((yes) => { resolve = yes; });
  h.choose("review"); const first = h.context.submit(); capture();
  h.switchTo("beta"); h.choose("build"); capture();
  resolve({ ok: true }); await first;
  assert.deepEqual(Array.from(h.selected()), ["build"]); capture();
  h.switchTo("alpha"); h.choose("review"); const second = h.context.submit();
  h.choose(""); h.choose("review"); h.type("bui"); capture();
  resolve({ ok: true }); await second;
  assert.deepEqual(Array.from(h.selected()), ["review"]);
  assert.equal(h.elements.skillsearch.value, "bui"); capture();
  return snapshots;
}

const completionScenarios = [
  ["command stable ranking, reset and composition", completionRanking],
  ["command replacement text, whitespace and caret", completionReplacement],
  ["command replacement refusal and exception order", completionExceptions],
  ["skill stable ranking, active retention and ordered selection", skillRankingAndSelection],
  ["skill late acceptance preserves session and pick identity", skillLateAcceptance],
];

test("pure completion: command ranking preserves references and frozen catalog order", async () => {
  const h = await commands(html);
  assert.equal(typeof h.context.rankCommandMatches, "function");
  const catalog = Object.freeze(commandCatalog.map((item) => Object.freeze({ ...item })));
  const before = plain(catalog);
  for (const [query, names] of [
    ["aud", ["audit", "auditor", "zaudit", "description"]],
    ["a", ["audit", "auditor"]],
    ["", before.map((item) => item.name)],
    ["AUD", []], ["none", []],
  ]) {
    const ranked = h.context.rankCommandMatches(catalog, query);
    assert.deepEqual(Array.from(ranked, (item) => item.name), names);
    assert.ok(ranked.every((item) => catalog.includes(item)), "ranked entries retain catalog identity");
  }
  assert.deepEqual(plain(catalog), before);
  assert.throws(() => h.context.rankCommandMatches([{ name: "missing" }], "xyz"), /toLowerCase/);
});

test("pure completion: skill ranking preserves ties and does not normalize the caller query", () => {
  const h = loadFixtures("app_skills.test.cjs", ["harness"]).harness();
  assert.equal(typeof h.context.rankSkillMatches, "function");
  const catalog = Object.freeze([
    Object.freeze({ name: "unrelated", description: "audit" }),
    Object.freeze({ name: "audit_own_changes" }),
    Object.freeze({ name: "auditor" }),
    Object.freeze({ name: "audit-other" }),
    Object.freeze({ name: "", description: "audit" }),
  ]);
  for (const [query, indices] of [["aud", [1, 2, 3, 0, 4]], ["", [0, 1, 2, 3, 4]], ["AUD", []]]) {
    const ranked = h.context.rankSkillMatches(catalog, query);
    assert.deepEqual(Array.from(ranked, (item) => catalog.indexOf(item)), indices);
  }
});

test("pure completion: replacement plans preserve token tails and frozen inputs", async () => {
  const h = await commands(html), c = h.context;
  assert.equal(typeof c.commandReplacementPlan, "function");
  const command = Object.freeze({ name: "audit" });
  for (const scenario of replacementCases) {
    const fragment = Object.freeze(c.commandContext(scenario.value, scenario.caret));
    const before = plain(fragment);
    assert.deepEqual(plain(c.commandReplacementPlan(scenario.value, fragment, command)),
      { value: scenario.output, caret: scenario.next });
    assert.deepEqual(plain(fragment), before);
  }
  assert.throws(() => c.commandReplacementPlan("/a", null, command), /end/);
});

for (const [name, run] of completionScenarios) {
  test("characterization completion: " + name, () => run(html));
  if (baseline) test("differential completion: " + name, async () => {
    assert.deepEqual(await run(html), await run(baseline));
  });
}

async function settle() {
  for (let i = 0; i < 32; i++) await Promise.resolve();
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function imageHarness(source, phone = false) {
  const fixtures = loadFixtures("app_ime.test.cjs", ["makeContext", "run", "script"], source);
  const readers = [], trace = [];
  const c = fixtures.makeContext({
    DRAFT_PREFIX: "draft:", sessionId: "old", pendingAppend: "",
    isPhoneLayout: () => phone, note: (text) => trace.push(["note", text]),
    addMsg: (role, text, cls) => trace.push(["message", role, text, cls]),
    scrollToLatest: () => trace.push(["scroll"]),
    FileReader: class {
      readAsDataURL(file) { readers.push(this); trace.push(["read", file.name, file.type]); }
    },
  });
  const slice = (start, end) => fixtures.script.slice(fixtures.script.indexOf(start), fixtures.script.indexOf(end));
  fixtures.run(c, slice("const composerDrafts = new Map();", "function closeQuestionPanel("));
  fixtures.run(c, slice("function uniqueImageFilename(", "// Entry point for the Android share target."));
  for (const [name, label] of [["setPromptValue", "value"], ["growPrompt", "grow"], ["persistDraft", "persist"]]) {
    const original = c[name];
    c[name] = (...args) => { trace.push([label, ...args]); return original(...args); };
  }
  const ids = new WeakMap();
  let nextId = 0;
  const arrayId = (value) => {
    if (!ids.has(value)) ids.set(value, ++nextId);
    return ids.get(value);
  };
  const snapshot = () => plain({
    dom: domSnapshot({ prompt: c.prompt }, c.document), style: c.prompt.style,
    sessionId: c.sessionId, append: fixtures.run(c, "pendingAppend"), trace,
    pendingImages: { id: arrayId(c.pendingImages), items: c.pendingImages },
    drafts: Array.from(fixtures.run(c, "composerDrafts")),
    imageDrafts: Array.from(fixtures.run(c, "composerImageDrafts"), ([key, images]) => [key, arrayId(images), images]),
    localStorage: Array.from(c.localStorage.store), frames: c.frames.length,
  });
  return { c, readers, trace, run: (code) => fixtures.run(c, code), snapshot };
}
async function finishImage(h, index, value = "data:image/png;base64,QQ==") {
  assert.ok(h.readers[index], "real queue created the next reader");
  h.readers[index].result = value; h.readers[index].onload(); await settle();
}

async function attachmentMarkerText(source) {
  const snapshots = [];
  for (const value of ["", "text", "text\n", "text\r", "text\r\n", "中🙂"]) {
    const h = imageHarness(source), images = h.c.pendingImages;
    h.c.prompt.value = value;
    const pending = h.c.queueImageFiles([
      { name: "image.png", type: "image/png" }, { name: "image.png", type: "image/png" },
    ]);
    snapshots.push(h.snapshot());
    await finishImage(h, 0);
    assert.equal(h.c.prompt.value, value, "marker text waits for the whole batch");
    snapshots.push(h.snapshot());
    await finishImage(h, 1); await pending;
    const append = (value && !value.endsWith("\n") ? "\n" : "") + "[[image:image.png]]\n[[image:image-2.png]]";
    assert.equal(h.c.prompt.value, value + append);
    assert.equal(h.c.pendingImages, images, "queue retains the active attachment array");
    assert.equal(h.run("composerImageDrafts.get(draftKey(sessionId))"), images);
    assert.deepEqual(h.trace.slice(-3), [["value", value + append], ["grow"], ["persist"]]);
    snapshots.push(h.snapshot());
  }
  return snapshots;
}

async function attachmentCompositionAndFailures(source) {
  const snapshots = [];
  const composing = imageHarness(source);
  composing.c.prompt.value = "ni"; composing.run("promptComposing = true;");
  const pending = composing.c.queueImageFiles([{ name: "p.png", type: "image/png" }]);
  await finishImage(composing, 0); await pending;
  assert.equal(composing.c.prompt.value, "ni");
  assert.equal(composing.run("pendingAppend"), "\n[[image:p.png]]");
  assert.deepEqual(composing.trace.slice(-2), [["value", "ni\n[[image:p.png]]"], ["note", "图片已附上，请先完成当前输入"]]);
  assert.equal(composing.trace.some(([kind]) => kind === "grow" || kind === "persist"), false);
  snapshots.push(composing.snapshot());
  const failed = imageHarness(source); failed.c.prompt.value = "unchanged";
  const badBatch = failed.c.queueImageFiles([null, { name: "text", type: "text/plain" },
    { name: "bad.png", type: "image/png" }, { name: "empty.png", type: "image/png" }]);
  await settle();
  failed.readers[0].error = new Error("read failed");
  failed.readers[0].onerror(); await settle();
  await finishImage(failed, 1, "data:image/png;base64,"); await badBatch;
  assert.equal(failed.c.pendingImages.length, 0);
  assert.equal(failed.c.prompt.value, "unchanged");
  assert.deepEqual(failed.trace, [
    ["message", "err", "无法附加 未命名图片：不支持的图片类型，请选择图片文件", "asst err"], ["scroll"],
    ["message", "err", "无法附加 text：不支持的图片类型，请选择图片文件", "asst err"], ["scroll"],
    ["read", "bad.png", "image/png"], ["message", "err", "无法附加 bad.png：read failed", "asst err"], ["scroll"],
    ["read", "empty.png", "image/png"], ["message", "err", "无法附加 empty.png：图片内容为空或无法读取", "asst err"], ["scroll"],
  ]);
  snapshots.push(failed.snapshot());
  const capped = imageHarness(source); capped.c.prompt.value = "full";
  for (let i = 0; i < 9; i++) capped.c.pendingImages.push({ filename: i + ".png", data_base64: "QQ==" });
  const fullBatch = capped.c.queueImageFiles([{ name: "late.png", type: "image/png" }, { name: "next.png", type: "image/png" }]);
  capped.c.pendingImages.push({ filename: "tenth.png", data_base64: "QQ==" });
  await finishImage(capped, 0); await fullBatch;
  assert.equal(capped.readers.length, 1, "capacity is checked again after the read");
  assert.equal(capped.c.pendingImages.length, 10);
  assert.equal(capped.c.prompt.value, "full");
  assert.deepEqual(capped.trace.slice(-4), [
    ["message", "err", "无法附加 late.png：最多只能附带 10 张图片", "asst err"], ["scroll"],
    ["message", "err", "无法附加 next.png：最多只能附带 10 张图片", "asst err"], ["scroll"],
  ]);
  snapshots.push(capped.snapshot());
  return snapshots;
}

async function attachmentSessionOwnership(source) {
  const snapshots = [];
  for (const phone of [false, true]) {
    const h = imageHarness(source, phone), c = h.c;
    c.prompt.value = "old draft"; c.pendingImages.push({ filename: "old.png", data_base64: "QQ==" });
    c.persistDraft(); const oldImages = c.pendingImages;
    const pending = c.queueImageFiles([{ name: "late.png", type: "image/png" }, { name: "again.png", type: "image/png" }]);
    h.run("persistDraft(); sessionId = 'new'; restoreDraft();");
    c.prompt.value = "new draft"; c.pendingImages.push({ filename: "new.png", data_base64: "Qg==" });
    c.persistDraft(); const newImages = c.pendingImages; snapshots.push(h.snapshot());
    await finishImage(h, 0); snapshots.push(h.snapshot());
    await finishImage(h, 1); await pending;
    assert.equal(c.pendingImages, newImages);
    assert.equal(h.run("composerImageDrafts.get(draftKey('old'))"), oldImages);
    assert.equal(c.prompt.value, "new draft"); snapshots.push(h.snapshot());
    h.run("persistDraft(); sessionId = 'old'; restoreDraft();");
    assert.equal(c.prompt.value, "old draft\n[[image:late.png]]\n[[image:again.png]]");
    assert.deepEqual(Array.from(c.pendingImages, (image) => image.filename), ["old.png", "late.png", "again.png"]);
    snapshots.push(h.snapshot());
  }
  return snapshots;
}

function previewSnapshot(h) {
  return plain({
    sessionId: h.context.sessionId, requested: h.requested, revoked: h.revoked,
    slots: Array.from(h.slots, (slot) => ({ dataset: slot.dataset, children: slot.children, replacement: slot.replacement })),
    urls: Array.from(h.context.previewUrls, ([slot, url]) => [h.slots.indexOf(slot), url]),
  });
}
function previews(source, paths, api) {
  return loadFixtures("app_math.test.cjs", ["previewHarness"], source).previewHarness(paths, api);
}
async function previewPathAttempts(source) {
  const snapshots = [];
  for (const [path, attempts] of [
    ["dir/result.png", ["dir/result.png", "result.png"]],
    ["bare.png", ["bare.png"]], ["/result.png", ["/result.png"]],
    ["./result.png", ["./result.png", "result.png"]],
    ["a//b.svg", ["a//b.svg", "b.svg"]], ["dir/", ["dir/", ""]],
    ["dir/a b#?.PNG", ["dir/a b#?.PNG", "a b#?.PNG"]],
  ]) {
    const h = previews(source, [path], () => ({ ok: false, status: 404 }));
    await h.context.loadPreviews(h.root);
    assert.deepEqual(Array.from(h.requested), attempts.map((p) => "/sessions/s/file?path=" + encodeURIComponent(p)));
    assert.equal(h.slots[0].replacement.text, path);
    snapshots.push(previewSnapshot(h));
  }
  for (const path of ["dir/result.png", undefined, null]) {
    const h = previews(source, [path]);
    await h.context.loadPreviews(h.root); await h.context.loadPreviews(h.root);
    assert.equal(h.requested.length, 1, "successful and held slots never calculate a fallback");
    assert.equal(h.slots[0].children[0].tag, "img"); snapshots.push(previewSnapshot(h));
  }
  for (const path of [undefined, null]) {
    const h = previews(source, [path], () => ({ ok: false, status: 404 }));
    await h.context.loadPreviews(h.root);
    assert.equal(h.requested.length, 1, "malformed fallback paths retain the original failure branch");
    assert.equal(h.slots[0].replacement.text, path); snapshots.push(previewSnapshot(h));
  }
  const rejectedBlob = previews(source, ["dir/retry.png"], () => ({
    ok: true, blob: () => rejectedBlob.requested.length === 1 ? Promise.reject(new Error("blob failed")) : Promise.resolve({}),
  }));
  await rejectedBlob.context.loadPreviews(rejectedBlob.root);
  assert.deepEqual(Array.from(rejectedBlob.requested), ["/sessions/s/file?path=dir%2Fretry.png", "/sessions/s/file?path=retry.png"]);
  snapshots.push(previewSnapshot(rejectedBlob));
  return snapshots;
}

async function previewSessionAndRetirement(source) {
  const snapshots = [], first = deferred();
  const h = previews(source, ["dir/one.png", "second.mp3"], () => {
    return h.requested.length === 1 ? first.promise : { ok: true, blob: async () => ({}) };
  });
  const pending = h.context.loadPreviews(h.root); await settle();
  assert.equal(h.requested.length, 1, "the first slot finishes before the second starts"); snapshots.push(previewSnapshot(h));
  h.context.sessionId = "other"; first.reject(new Error("offline")); await pending;
  assert.deepEqual(Array.from(h.requested), ["/sessions/s/file?path=dir%2Fone.png", "/sessions/s/file?path=one.png", "/sessions/s/file?path=second.mp3"]);
  assert.equal(h.slots[1].children[0].tag, "audio");
  assert.equal(h.slots[1].children[0].controls, true);
  assert.equal(h.slots[1].children[0].preload, "none"); snapshots.push(previewSnapshot(h));
  const failed = deferred();
  const retired = previews(source, ["dir/one.png"], () => failed.promise);
  const retiring = retired.context.loadPreviews(retired.root); await settle();
  retired.context.revokePreviews(); retired.context.sessionId = "other"; snapshots.push(previewSnapshot(retired));
  failed.reject(new Error("not found")); await retiring;
  assert.equal(retired.requested.length, 1);
  assert.equal(retired.slots[0].replacement, undefined); snapshots.push(previewSnapshot(retired));
  const blob = deferred();
  const late = previews(source, ["result.png", "other.png"], () => ({ ok: true, blob: () => blob.promise }));
  const loading = late.context.loadPreviews(late.root); await settle(); snapshots.push(previewSnapshot(late));
  late.context.revokePreviews(); blob.resolve({}); await loading;
  assert.deepEqual(Array.from(late.revoked), ["blob:1"]);
  assert.equal(late.requested.length, 1);
  assert.equal(late.slots[0].children.length, 0);
  assert.equal(late.context.previewUrls.size, 0); snapshots.push(previewSnapshot(late));
  return snapshots;
}

const attachmentScenarios = [
  ["marker newline, filenames and batch timing", attachmentMarkerText],
  ["composition, failed reads and capacity after await", attachmentCompositionAndFailures],
  ["phone and desktop delayed reads retain session and array ownership", attachmentSessionOwnership],
  ["preview candidates, encoding and held slots", previewPathAttempts],
  ["preview sequential slots, captured session and retirement", previewSessionAndRetirement],
];

test("pure attachment: marker text preserves newline quirks and frozen image inputs", () => {
  const h = imageHarness(html);
  assert.equal(typeof h.c.imageMarkerAppend, "function");
  const images = Object.freeze([Object.freeze({ filename: "raw [中].png" }), Object.freeze({ filename: "two.png" })]);
  const markers = "[[image:raw [中].png]]\n[[image:two.png]]";
  for (const [value, prefix] of [["", ""], ["text", "\n"], ["text\r", "\n"], ["text\n", ""], ["text\r\n", ""]]) {
    assert.equal(h.c.imageMarkerAppend(value, images), prefix + markers);
  }
  assert.equal(h.c.imageMarkerAppend("text", Object.freeze([])), "\n", "an empty calculation retains the original expression");
  assert.equal(h.c.imageMarkerAppend("", Object.freeze([])), "");
  assert.throws(() => h.c.imageMarkerAppend(1, images), /endsWith/);
  assert.throws(() => h.c.imageMarkerAppend("", null), /map/);
});

test("pure attachment: preview path candidates preserve basename and invalid path behavior", () => {
  const h = previews(html, []);
  assert.equal(typeof h.context.previewPathCandidates, "function");
  for (const [input, expected] of [
    ["dir/a.png", ["dir/a.png", "a.png"]], ["a.png", ["a.png"]], ["/a.png", ["/a.png"]],
    ["a//b.svg", ["a//b.svg", "b.svg"]], ["dir/", ["dir/", ""]], ["", [""]],
  ]) assert.deepEqual(Array.from(h.context.previewPathCandidates(input)), expected);
  assert.throws(() => h.context.previewPathCandidates(undefined), /lastIndexOf/);
  assert.throws(() => h.context.previewPathCandidates(null), /lastIndexOf/);
});

for (const [name, run] of attachmentScenarios) {
  test("characterization attachment: " + name, () => run(html));
  if (baseline) test("differential attachment: " + name, async () => {
    assert.deepEqual(await run(html), await run(baseline));
  });
}

function teamHarness(source, options = {}) {
  const fixtures = loadFixtures("app_team_mode.test.cjs", ["harness", "NORMAL", "TRAECLI", "response"], source);
  return { ...fixtures, ...fixtures.harness(options) };
}
function teamSnapshot(h, trace = [], result = null) {
  const state = vm.runInContext("teamPicks.get(teamPickKey())", h.context);
  return plain({
    dom: domSnapshot(h.elements, h.context.document),
    state: state ? { ...state, saving: Boolean(state.saving) } : null,
    draft: vm.runInContext("teamDraft", h.context),
    history: { pushes: h.history.pushes, backs: h.history.backs },
    sid: h.context.sessionId, requests: h.requests, notes: h.notes, stored: Array.from(h.stored), trace, result,
  });
}
const teamDisplayCases = [
  { sid: "alpha", confirmed: null, mode: "normal", label: "对话模式", detail: "", status: "未同步", dataMode: "unknown", phase: "ready" },
  { sid: "", confirmed: null, mode: "normal", label: "普通对话", detail: "", status: "", dataMode: "normal", phase: "ready" },
  { sid: "alpha", confirmed: {}, mode: "team", backend: "kernel", model: "ignored", label: "subagent", detail: " · Kernel", status: "", dataMode: "team", phase: "ready" },
  { sid: "alpha", confirmed: {}, mode: "team", backend: "traecli", model: " worker ", label: "subagent", detail: " · TraeCLI", status: "", dataMode: "team", phase: "ready" },
  { sid: "alpha", confirmed: null, mode: "team", pending: true, label: "subagent", detail: " · TraeCLI", status: "待保存", dataMode: "team", phase: "pending" },
  { sid: "alpha", confirmed: null, mode: "team", pending: true, saving: {}, error: "offline", label: "subagent", detail: " · TraeCLI", status: "保存中", dataMode: "team", phase: "saving" },
  { sid: "alpha", confirmed: null, mode: "team", saving: {}, label: "对话模式", detail: " · TraeCLI", status: "保存中", dataMode: "unknown", phase: "saving" },
  { sid: "alpha", confirmed: null, mode: "normal", error: "offline", label: "对话模式", detail: "", status: "重试", dataMode: "unknown", phase: "error" },
  { sid: "alpha", confirmed: {}, mode: "normal", pending: true, error: "save failure", label: "普通对话", detail: "", status: "重试", dataMode: "normal", phase: "error" },
];
function displayState(scenario) {
  return {
    choice: { mode: scenario.mode, backend: scenario.backend || "traecli", model: scenario.model || "" },
    confirmed: scenario.confirmed, pending: Boolean(scenario.pending), saving: scenario.saving || null, error: scenario.error || "",
  };
}
async function teamChipDisplay(source) {
  const snapshots = [];
  for (const scenario of teamDisplayCases) {
    const h = teamHarness(source, { sid: scenario.sid }), c = h.context;
    const state = Object.assign(h.state(), displayState(scenario)), before = plain(state);
    c.syncTeamTrigger();
    assert.equal(h.elements.teamlabel.textContent, scenario.label);
    assert.equal(h.elements.teamstatus.textContent, scenario.status);
    assert.equal(h.elements.teamtrigger.dataset.mode, scenario.dataMode);
    assert.equal(h.elements.teamtrigger.dataset.state, scenario.phase);
    assert.equal(h.elements.teamtrigger.attrs["aria-busy"], String(Boolean(scenario.saving)));
    assert.equal(h.elements.teamtrigger.attrs["aria-label"],
      scenario.label + scenario.detail + (scenario.status ? "，" + scenario.status : "") + "，选择对话模式");
    const model = scenario.mode === "team" && (scenario.backend || "traecli") === "traecli" && scenario.model
      ? " · 子 Agent 模型：" + scenario.model : "";
    assert.equal(h.elements.teamtrigger.title, scenario.error || scenario.label + scenario.detail + model);
    assert.deepEqual(plain(state), before, "sync only projects the state");
    snapshots.push(teamSnapshot(h));
  }
  return snapshots;
}

async function teamChipTimingAndExceptions(source) {
  const h = teamHarness(source), c = h.context, snapshots = [], trace = [], callbacks = [];
  const stateOf = c.teamPickState;
  c.teamPickState = (...args) => { trace.push(["state", c.sessionId]); return stateOf(...args); };
  c.holdLayout = (key, fn) => { trace.push(["hold", key]); callbacks.push(fn); };
  const before = domSnapshot(h.elements, c.document);
  c.syncTeamTrigger();
  assert.deepEqual(trace, [["hold", "teamMode"]]);
  assert.deepEqual(domSnapshot(h.elements, c.document), before); snapshots.push(teamSnapshot(h, trace));
  c.sessionId = "beta";
  const state = stateOf("beta");
  Object.assign(state, displayState({ confirmed: {}, mode: "team", backend: "kernel" }));
  callbacks[0]();
  assert.deepEqual(trace, [["hold", "teamMode"], ["state", "beta"]]);
  assert.equal(h.elements.teamlabel.textContent, "subagent"); snapshots.push(teamSnapshot(h, trace));
  c.holdLayout = (_, fn) => fn();
  state.choice = { mode: "team", backend: "traecli", model: Symbol("model") };
  let result;
  try { c.syncTeamTrigger(); } catch (error) { result = error.message; }
  assert.match(result, /Symbol/);
  assert.equal(h.elements.teamtrigger.title, "subagent · Kernel", "title failure follows earlier DOM writes");
  assert.equal(h.elements.teamtrigger.attrs["aria-label"], "subagent · TraeCLI，选择对话模式");
  snapshots.push(teamSnapshot(h, trace, result));
  state.choice = null;
  const prior = domSnapshot(h.elements, c.document);
  assert.throws(() => c.syncTeamTrigger(), /mode/);
  assert.deepEqual(domSnapshot(h.elements, c.document), prior, "an invalid choice fails before DOM writes");
  snapshots.push(teamSnapshot(h, trace));
  return snapshots;
}

async function geometryHarness(source) {
  const fixtures = loadFixtures("app_team_mode.test.cjs", ["harness", "TRAECLI"], source);
  const h = fixtures.harness({ cfg: fixtures.TRAECLI });
  await h.context.loadSessionConfig(); h.context.openTeamSheet(); await settle();
  vm.runInContext("teamDraft.workerOpen = true;", h.context);
  h.elements.teamworkerlist.style.bottom = "fallback-bottom";
  h.elements.teamworkerlist.style.maxHeight = "fallback-height";
  h.elements.teamworkeroptions.style.maxHeight = "fallback-options";
  return h;
}
async function workerGeometryBoundaries(source) {
  const snapshots = [];
  for (const scenario of [
    { wrap: { height: 112, bottom: 498 }, panel: { top: 60, bottom: 900 }, view: 800, bottom: "412px", height: "422px", options: "358px" },
    { wrap: { height: .5, bottom: 499.6 }, panel: { top: 60.1, bottom: 900.2 }, view: 800, bottom: "411px", height: "423px", options: "359px" },
    { wrap: { height: -1, bottom: 100 }, panel: { top: 80, bottom: 110 }, view: 100, bottom: "20px", height: "240px", options: "176px" },
    { wrap: { height: 112, bottom: 1500 }, panel: { top: 0, bottom: 1520 }, view: 300, bottom: "30px", height: "288px", options: "224px" },
    { wrap: { height: 112, bottom: 1500 }, panel: { top: 0, bottom: 1520 }, view: "300", bottom: "30px", height: "288px", options: "224px" },
    { wrap: { height: 112, bottom: 1000 }, panel: { top: 0, bottom: 1200 }, view: 0, bottom: "210px", height: "614px", options: "550px" },
    { wrap: { height: 112, bottom: 1000 }, panel: { top: 0, bottom: 1200 }, view: -3, bottom: "210px", height: "614px", options: "550px" },
    { wrap: { height: 112, bottom: 1000 }, panel: { top: 0, bottom: 1200 }, view: undefined, bottom: "210px", height: "614px", options: "550px" },
    { wrap: { height: 112, bottom: 1500 }, panel: { top: 0, bottom: 1520 }, view: Infinity, bottom: "30px", height: "1484px", options: "1420px" },
    { wrap: { height: 112, bottom: 100 }, panel: { top: 80, bottom: 0 }, view: 800, bottom: "6px", height: "240px", options: "176px" },
    { wrap: { height: 112, bottom: NaN }, panel: { top: 80, bottom: 110 }, view: 800, bottom: "NaNpx", height: "NaNpx", options: "NaNpx" },
    { wrap: { height: 0, bottom: 100 }, panel: { top: 80, bottom: 110 }, view: 800, bottom: "fallback-bottom", height: "fallback-height", options: "fallback-options" },
    { wrap: null, panel: { top: 80, bottom: 110 }, view: 800, bottom: "fallback-bottom", height: "fallback-height", options: "fallback-options" },
    { wrap: { height: 112, bottom: 100 }, panel: null, view: 800, bottom: "fallback-bottom", height: "fallback-height", options: "fallback-options" },
  ]) {
    const h = await geometryHarness(source), trace = [];
    h.elements.teamworkerwrap.getBoundingClientRect = () => { trace.push(["rect", "wrap"]); return scenario.wrap; };
    h.elements.teampickpanel.getBoundingClientRect = () => { trace.push(["rect", "panel"]); return scenario.panel; };
    Object.defineProperty(h.context.window, "innerHeight", { get() { trace.push(["view"]); return scenario.view; } });
    const before = plain(h.state());
    h.context.placeWorkerList();
    assert.equal(h.elements.teamworkerlist.style.bottom, scenario.bottom);
    assert.equal(h.elements.teamworkerlist.style.maxHeight, scenario.height);
    assert.equal(h.elements.teamworkeroptions.style.maxHeight, scenario.options);
    assert.deepEqual(plain(h.state()), before);
    const reads = scenario.wrap?.height && scenario.panel ? Number(scenario.view) > 0 ? 2 : 1 : 0;
    assert.equal(trace.filter(([name]) => name === "view").length, reads);
    snapshots.push(teamSnapshot(h, trace));
  }
  return snapshots;
}

async function workerGeometryExceptions(source) {
  const snapshots = [];
  for (const branch of ["closed", "wrap throws", "view throws", "bottom write throws", "second bottom throws", "options lookup throws"]) {
    const h = await geometryHarness(source), c = h.context, trace = [];
    let bottomReads = 0;
    const wrap = { height: 112, get bottom() {
      trace.push(["wrap bottom", ++bottomReads]);
      if (branch === "second bottom throws" && bottomReads === 2) throw new Error("second bottom failure");
      return 498;
    } };
    const panel = { get bottom() { trace.push(["panel bottom"]); return 900; },
      get top() { trace.push(["panel top"]); return 60; } };
    h.elements.teamworkerwrap.getBoundingClientRect = () => {
      trace.push(["rect", "wrap"]); if (branch === "wrap throws") throw new Error("wrap failure"); return wrap;
    };
    h.elements.teampickpanel.getBoundingClientRect = () => { trace.push(["rect", "panel"]); return panel; };
    Object.defineProperty(c.window, "innerHeight", { get() {
      trace.push(["view"]); if (branch === "view throws") throw new Error("view failure"); return 800;
    } });
    for (const [id, name] of [["teamworkerlist", "list"], ["teamworkeroptions", "options"]]) {
      h.elements[id].style = new Proxy(h.elements[id].style, { set(target, key, value) {
        trace.push(["write", name, key, value]);
        if (branch === "bottom write throws" && key === "bottom") throw new Error("bottom write failure");
        target[key] = value; return true;
      } });
    }
    const lookup = c.$;
    c.$ = (id) => {
      if (id === "teamworkeroptions") {
        trace.push(["lookup", id]);
        if (branch === "options lookup throws") throw new Error("options lookup failure");
      }
      return lookup(id);
    };
    if (branch === "closed") vm.runInContext("teamDraft.workerOpen = false;", c);
    let result = null;
    try { c.placeWorkerList(); } catch (error) { result = error.message; }
    if (branch === "closed") assert.deepEqual(trace, []);
    else if (branch === "wrap throws") assert.deepEqual(trace, [["rect", "wrap"]]);
    else if (branch === "view throws") assert.deepEqual(trace, [["rect", "wrap"], ["rect", "panel"], ["view"]]);
    else {
      const prefix = [["rect", "wrap"], ["rect", "panel"], ["view"], ["view"],
        ["panel bottom"], ["wrap bottom", 1], ["write", "list", "bottom", "412px"]];
      if (branch === "second bottom throws") prefix.push(["wrap bottom", 2]);
      if (branch === "options lookup throws") prefix.push(["wrap bottom", 2], ["panel top"],
        ["write", "list", "maxHeight", "422px"], ["lookup", "teamworkeroptions"]);
      assert.deepEqual(trace, prefix);
    }
    const expected = { closed: null, "wrap throws": "wrap failure", "view throws": "view failure",
      "bottom write throws": "bottom write failure", "second bottom throws": "second bottom failure",
      "options lookup throws": "options lookup failure" };
    assert.equal(result, expected[branch]);
    snapshots.push(teamSnapshot(h, trace, result));
  }
  return snapshots;
}

async function teamSaveAndEpochs(source) {
  const save = deferred(), h = teamHarness(source, { configPost: () => save.promise }), c = h.context, snapshots = [];
  await c.loadSessionConfig();
  const applying = h.choose("team", "traecli", "worker-model"); await settle();
  const state = h.state(), saving = state.saving, revision = state.revision, fullEpoch = state.fullLoadEpoch;
  assert.equal(typeof saving.then, "function");
  assert.equal(h.elements.teamstatus.textContent, "保存中");
  snapshots.push(teamSnapshot(h));
  h.elements.omodel.value = "main-model"; h.elements.oagent.value = "build"; h.elements.oeffort.value = "high";
  await c.loadSessionConfig({ teamOnly: true });
  assert.equal(state.saving, saving, "team-only hydration retains the save promise");
  assert.equal(state.revision, revision);
  assert.equal(state.fullLoadEpoch, fullEpoch);
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["main-model", "build", "high"]);
  snapshots.push(teamSnapshot(h));
  h.configs.set("alpha", plain(h.TRAECLI)); save.resolve(h.response(h.TRAECLI));
  assert.equal(await applying, true); await settle();
  assert.equal(state.saving, null); assert.equal(state.pending, false);
  assert.equal(state.revision, revision + 1);
  assert.equal(h.elements.teamstatus.textContent, "");
  assert.equal(h.history.pushes, 1); assert.equal(h.history.backs, 1);
  assert.equal(c.document.activeElement, h.elements.teamtrigger);
  assert.equal(h.teamPosts().length, 1);
  assert.deepEqual(plain(h.teamPosts()[0].body), { subagents: { action: "use", backend: "traecli", model: "worker-model" } });
  snapshots.push(teamSnapshot(h));
  return snapshots;
}

const teamScenarios = [
  ["chip labels, details and status priority", teamChipDisplay],
  ["chip layout deferral and partial writes on exceptions", teamChipTimingAndExceptions],
  ["worker measured geometry, floor, cap and missing boxes", workerGeometryBoundaries],
  ["worker geometry read and style-write exception order", workerGeometryExceptions],
  ["save identity, revision, team-only epochs, focus and history", teamSaveAndEpochs],
];

test("pure team: chip values preserve status priority and frozen state", () => {
  const h = teamHarness(html), c = h.context;
  assert.equal(typeof c.teamChipValues, "function");
  for (const scenario of teamDisplayCases) {
    const input = displayState(scenario);
    Object.freeze(input.choice); Object.freeze(input);
    const before = plain(input);
    assert.deepEqual(plain(c.teamChipValues(input, scenario.sid)), {
      unknown: scenario.dataMode === "unknown" ? true : scenario.sid ? false : "",
      label: scenario.label, detail: scenario.detail, status: scenario.status,
    });
    assert.deepEqual(plain(input), before);
  }
  assert.throws(() => c.teamChipValues({ choice: null, confirmed: {} }, "alpha"), /mode/);
});

test("pure team: worker bottom keeps the floor, rounding and numeric coercion", () => {
  const h = teamHarness(html), c = h.context;
  assert.equal(typeof c.workerListBottom, "function");
  for (const [panel, wrap, expected] of [[900, 498, 412], [900.2, 499.6, 411], [0, 100, 6], ["900.2", "499.6", 411]]) {
    assert.equal(c.workerListBottom(panel, wrap), expected);
  }
  assert.ok(Number.isNaN(c.workerListBottom(900, NaN)));
  assert.throws(() => c.workerListBottom(Symbol("bottom"), 100), /Symbol/);
});

test("pure team: worker heights use passed geometry without DOM or viewport reads", () => {
  const h = teamHarness(html), c = h.context;
  assert.equal(typeof c.workerListHeights, "function");
  c.$ = () => assert.fail("pure geometry must not look up DOM");
  Object.defineProperty(c.window, "innerHeight", { get() { assert.fail("pure geometry must use the supplied viewport"); } });
  for (const [wrap, panel, view, expected] of [
    [498, 60, 800, { height: 422, optionsHeight: 358 }],
    [499.6, 60.1, 800, { height: 423, optionsHeight: 359 }],
    [100, 80, 100, { height: 240, optionsHeight: 176 }],
    [1500, 0, 300, { height: 288, optionsHeight: 224 }],
    [1500, 0, Infinity, { height: 1484, optionsHeight: 1420 }],
    [1000, 0, 0, { height: 240, optionsHeight: 176 }],
  ]) assert.deepEqual(plain(c.workerListHeights(wrap, panel, view)), expected);
  const invalid = c.workerListHeights(NaN, 60, 800);
  assert.ok(Number.isNaN(invalid.height) && Number.isNaN(invalid.optionsHeight));
  assert.throws(() => c.workerListHeights(Symbol("bottom"), 60, 800), /Symbol/);
});

for (const [name, run] of teamScenarios) {
  test("characterization team: " + name, () => run(html));
  if (baseline) test("differential team: " + name, async () => {
    assert.deepEqual(await run(html), await run(baseline));
  });
}

function taskHarness(source, options) {
  const fixtures = loadFixtures("app_task_results.test.cjs", ["harness", "reportedTask", "report", "taskCard"], source);
  return { ...fixtures, ...fixtures.harness(options) };
}
const taskValue = (value) => JSON.parse(JSON.stringify(value, (_, item) => item === undefined ? { undefined: true }
  : typeof item === "number" && !Number.isFinite(item) ? { number: String(item) } : item));
function observeTasks(h) {
  const c = h.context, trace = [], observed = { trace, records: null, value: null, nowCalls: 0 };
  c.Date.now = () => { observed.nowCalls++; trace.push(["now", h.clock.now]); return h.clock.now; };
  const merge = c.mergeTaskReport, project = c.projectTaskReport;
  c.mergeTaskReport = (records, report, at) => {
    const result = merge(records, report, at);
    observed.records = records; trace.push(["merge", taskValue(result), taskValue(Array.from(records))]);
    return result;
  };
  c.projectTaskReport = (value, ...args) => {
    observed.value = value;
    trace.push(["project", taskValue(value), args[0] !== false]);
    return project(value, ...args);
  };
  for (const name of ["setInterval", "clearInterval"]) {
    const original = c[name];
    c[name] = (...args) => { trace.push([name, ...args.map((item) => typeof item === "function" ? "callback" : item)]); return original(...args); };
  }
  return observed;
}
function taskSnapshot(h, observed) {
  const panels = h.root.querySelectorAll(".task-output");
  return taskValue({
    dom: domSnapshot({ root: h.root }, h.context.document), storage: Array.from(h.storage),
    records: observed.records ? Array.from(observed.records) : null,
    projected: observed.value, trace: observed.trace, nowCalls: observed.nowCalls,
    clocks: panels.map((panel) => vm.runInContext("subagentClocks.has(observedBox)",
      Object.assign(h.context, { observedBox: panel.parentNode }))),
    ticking: vm.runInContext("tickingSubagents.size", h.context),
    taskStates: Array.from(h.context.executionState().taskStates),
  });
}
function realTaskClock(h, mode) {
  h.emit(mode, "initial raw", "task_spawn", { task_report: h.report([
    h.reportedTask("initial", { active: true, output: undefined, status: "running" }),
  ], { operation: "task_spawn" }) });
  const box = h.taskCard(h).parentNode;
  const clock = vm.runInContext("subagentClocks.get(observedBox)", Object.assign(h.context, { observedBox: box }));
  assert.ok(clock, "the real source created a task clock");
  return clock;
}
async function taskAgingNumeric(source) {
  const snapshots = [];
  for (const mode of ["live", "replay"]) for (const phone of [false, true]) {
    const h = taskHarness(source, { phone }), observed = observeTasks(h), clock = realTaskClock(h, mode);
    const tasks = [
      h.reportedTask("numeric", { active: true, output: undefined, elapsed_ms: 5000,
        progress: { stale_for_ms: 1500, elapsed_ms: 250, timeline: [{ elapsed_ms: 100 }] } }),
      h.reportedTask("negative", { active: true, output: undefined, elapsed_ms: -20, progress: { stale_for_ms: -100 } }),
      h.reportedTask("nonfinite", { active: true, output: undefined, elapsed_ms: Infinity, progress: { stale_for_ms: NaN } }),
      h.reportedTask("strings", { active: true, output: undefined, elapsed_ms: "1000", progress: { stale_for_ms: "750" } }),
      h.reportedTask("done", { active: false, elapsed_ms: 8000, progress: { stale_for_ms: 3000 }, output: "  finished bytes\n" }),
    ];
    const report = h.report(tasks, { snapshot: true, wait: { reason: "parked", waited_ms: 500 } }), before = taskValue(report);
    clock.set({ text: "raw FAILED after 999s", task_report: report });
    const details = h.taskCard(h).parentNode.querySelector("details"); details.open = false; details.fire("toggle");
    const recordSnapshot = taskValue(Array.from(observed.records)); snapshots.push(taskSnapshot(h, observed));
    for (const [now, elapsed, stale, negative, negativeStale] of [
      [0, 5000, 1500, -20, -100], [2500, 6500, 3000, 1480, 1400], [6000, 10000, 6500, 4980, 4900], [6000, 10000, 6500, 4980, 4900],
    ]) {
      const calls = observed.nowCalls;
      clock.paint(now);
      assert.equal(observed.nowCalls, calls, "paint uses the supplied time");
      const value = observed.value;
      assert.deepEqual(Array.from(value.tasks, (task) => task.task_id), ["numeric", "negative", "nonfinite", "strings", "done"]);
      assert.equal(value.tasks[0].elapsed_ms, elapsed); assert.equal(value.tasks[0].progress.stale_for_ms, stale);
      assert.equal(value.tasks[0].progress.elapsed_ms, 250);
      assert.equal(value.tasks[0].progress.timeline, observed.records.get("numeric").task.progress.timeline);
      assert.equal(value.tasks[1].elapsed_ms, negative); assert.equal(value.tasks[1].progress.stale_for_ms, negativeStale);
      assert.equal(value.tasks[2].elapsed_ms, Infinity); assert.ok(Number.isNaN(value.tasks[2].progress.stale_for_ms));
      assert.equal(value.tasks[3].elapsed_ms, "1000"); assert.equal(value.tasks[3].progress.stale_for_ms, "750");
      assert.equal(value.tasks[4], observed.records.get("done").task, "inactive tasks retain their reference");
      assert.equal(value.tasks[4].elapsed_ms, 8000); assert.equal(value.tasks[4].progress.stale_for_ms, 3000);
      assert.equal(value.wait.waited_ms, 500); assert.deepEqual(Array.from(value.currentTaskIndexes), [4]);
      assert.notEqual(value.tasks[0], observed.records.get("numeric").task);
      assert.equal(details.open, false, "painting preserves the reader's disclosure");
      assert.deepEqual(taskValue(Array.from(observed.records)), recordSnapshot, "painting never updates stored ages or current flags");
      assert.deepEqual(taskValue(report), before);
      snapshots.push(taskSnapshot(h, observed));
    }
  }
  return snapshots;
}

async function taskAgingProgressShapes(source) {
  const h = taskHarness(source), observed = observeTasks(h), clock = realTaskClock(h, "live");
  const shapes = [undefined, null, false, 0, "", "go", []];
  clock.set({ task_report: h.report(shapes.map((progress, index) =>
    h.reportedTask("shape-" + index, { active: true, elapsed_ms: 100, output: undefined, progress })), { snapshot: true }) });
  const before = taskValue(Array.from(observed.records)), snapshots = [taskSnapshot(h, observed)];
  clock.paint(2500);
  for (let i = 0; i < 5; i++) assert.equal(observed.value.tasks[i].progress, shapes[i]);
  assert.deepEqual(Object.keys(observed.value.tasks[5].progress), ["0", "1", "stale_for_ms"]);
  assert.equal(observed.value.tasks[5].progress[0], "g"); assert.equal(observed.value.tasks[5].progress[1], "o");
  assert.equal(observed.value.tasks[5].progress.stale_for_ms, undefined);
  assert.deepEqual(Object.keys(observed.value.tasks[6].progress), ["stale_for_ms"]);
  assert.ok(observed.value.tasks.every((task) => task.elapsed_ms === 1600));
  assert.deepEqual(taskValue(Array.from(observed.records)), before);
  snapshots.push(taskSnapshot(h, observed)); return snapshots;
}

async function taskMergeOrderAndProtection(source) {
  const h = taskHarness(source), records = new Map(), snapshots = [];
  const active = (id) => h.reportedTask(id, { active: true, output: undefined, status: "running" });
  const a = active("a"), b = active("b"), d = active("d");
  const doneA = { ...a, active: false, output: "  a result\n" }, doneB = { ...b, active: false, output: "b result\n" };
  const anonymous = h.reportedTask(undefined, { index: 0, output: "anonymous result\n" });
  const stages = [
    [h.report([a, b, d], { operation: "task_spawn" }), ["a", "b", "d"], []],
    [h.report([doneB]), ["a", "b", "d"], [1]],
    [h.report([doneA, anonymous]), ["a", "b", "d", "task_wait:0"], [0, 3]],
    [h.report([{ ...a, elapsed_ms: 999999 }], { snapshot: true }), ["a", "b", "task_wait:0"], [0, 2]],
    [h.report([{ ...doneB, output: "b", output_truncated: true }]), ["a", "b", "task_wait:0"], [0, 2]],
    [h.report([h.reportedTask("error", { output: undefined, error: "current failure", status: "failed" })]), ["a", "b", "task_wait:0", "error"], [3]],
    [h.report([{ ...doneB, output: "second b result\n" }]), ["a", "b", "task_wait:0", "error"], [1]],
  ];
  for (const [index, [incoming, keys, current]] of stages.entries()) {
    const result = h.context.mergeTaskReport(records, incoming, (index + 1) * 1000);
    assert.deepEqual(Array.from(records.keys()), keys);
    assert.deepEqual(Array.from(result.currentTaskIndexes), current);
    assert.deepEqual(Array.from(result.tasks, (task) => task.task_id || "task_wait:" + task.index), keys);
    if (index >= 3) {
      assert.equal(records.get("a").task.output, "  a result\n"); assert.equal(records.get("a").task.active, false);
      assert.equal(records.get("a").at, 3000, "rejected progress does not rebase the delivered result");
    }
    if (index === 4) assert.equal(records.get("b").task.output, "b result\n");
    snapshots.push(taskValue({ result, records: Array.from(records), projection: h.context.projectTaskReport(result) }));
  }
  return snapshots;
}

async function taskHistoryDoesNotAge(source) {
  const snapshots = [];
  for (const mode of ["history", "cold"]) for (const phone of [false, true]) {
    const h = taskHarness(source, { phone }), observed = observeTasks(h);
    h.context.createSubagentClock = () => assert.fail("canonical history must not create an activity clock");
    h.context.agedTaskReport = () => assert.fail("canonical history must not invoke live aging");
    h.context.Date.now = () => assert.fail("canonical history uses stored task ages");
    h.context.setInterval = () => assert.fail("canonical history must not start the activity ticker");
    h.clock.now = 1000000000;
    const task = h.reportedTask("frozen", { active: true, output: undefined, elapsed_ms: 5000, progress: { stale_for_ms: 1000 } });
    h.emit(mode, "history raw 999s", "task_status", { task_report: h.report([task], { snapshot: true, operation: "task_status" }) });
    const panel = h.taskCard(h);
    assert.match(panel.querySelector(".task-identity").textContent, /5s/);
    assert.match(panel.querySelector(".task-meta").nextElementSibling?.textContent || panel.textContent, /1s ago/);
    assert.equal(vm.runInContext("tickingSubagents.size", h.context), 0);
    assert.ok(observed.records && Array.from(observed.records.values()).every((row) => row.at === 0));
    snapshots.push(taskSnapshot(h, observed));
  }
  return snapshots;
}

async function taskAgingExceptionsAndLegacy(source) {
  const h = taskHarness(source), observed = observeTasks(h), clock = realTaskClock(h, "live"), snapshots = [];
  const before = taskSnapshot(h, observed), traceLength = observed.trace.length;
  assert.throws(() => clock.paint(Symbol("time")), /Symbol/);
  assert.equal(observed.trace.length, traceLength, "age calculation fails before labels or projection");
  assert.deepEqual(taskSnapshot(h, observed), before);
  snapshots.push(taskSnapshot(h, observed));
  clock.set("FAILED after 999s\nraw legacy tail  ");
  assert.equal(clock.busy(), null);
  assert.equal(observed.records.size, 0);
  assert.equal(observed.value, null);
  assert.equal(h.taskCard(h), undefined);
  assert.ok(h.root.textContent.includes("FAILED after 999s\nraw legacy tail  "));
  assert.equal(vm.runInContext("tickingSubagents.size", h.context), 0);
  snapshots.push(taskSnapshot(h, observed));
  clock.paint(500000);
  assert.equal(observed.value, null);
  snapshots.push(taskSnapshot(h, observed)); return snapshots;
}

const taskScenarios = [
  ["numeric ages, rollback, repeated paint, references and raw fields", taskAgingNumeric],
  ["missing and non-object progress values", taskAgingProgressShapes],
  ["Map order, anonymous tasks, delivery protection and current batch", taskMergeOrderAndProtection],
  ["canonical history retains ages and never starts the activity clock", taskHistoryDoesNotAge],
  ["age failure precedes paint effects and legacy clears records", taskAgingExceptionsAndLegacy],
];

test("pure task: aged reports preserve frozen records, terminal identity and report metadata", () => {
  const c = taskHarness(html).context;
  assert.equal(typeof c.agedTaskReport, "function");
  c.Date.now = () => assert.fail("pure aging must use its now argument");
  const timeline = Object.freeze([Object.freeze({ elapsed_ms: 100 })]);
  const active = Object.freeze({ task_id: "z-active", status: "running", active: true, elapsed_ms: 5000,
    progress: Object.freeze({ stale_for_ms: 1000, elapsed_ms: 250, timeline }) });
  const done = Object.freeze({ task_id: "a-done", status: "completed", active: false, elapsed_ms: 8000, output: "  exact result\n" });
  const records = new Map([
    ["z-active", Object.freeze({ task: active, at: 1000, current: true })],
    ["a-done", Object.freeze({ task: done, at: 1000, current: false })],
  ]);
  for (const method of ["set", "delete", "clear"]) records[method] = () => assert.fail("pure aging must not mutate the Map");
  Object.freeze(records);
  const input = Object.freeze({ schema: "ai.task-report", version: 1, operation: "task_wait", snapshot: true,
    tasks: Object.freeze([]), currentTaskIndexes: Object.freeze([1]),
    wait: Object.freeze({ reason: "parked", waited_ms: 500 }), error: "", extra: "producer metadata" });
  const before = taskValue([input, Array.from(records)]);
  const result = c.agedTaskReport(input, records, 6000);
  assert.notEqual(result, input);
  assert.deepEqual(Array.from(result.tasks, (task) => task.task_id), ["z-active", "a-done"]);
  assert.notEqual(result.tasks[0], active); assert.notEqual(result.tasks[0].progress, active.progress);
  assert.equal(result.tasks[0].elapsed_ms, 10000); assert.equal(result.tasks[0].progress.stale_for_ms, 6000);
  assert.equal(result.tasks[0].progress.elapsed_ms, 250); assert.equal(result.tasks[0].progress.timeline, timeline);
  assert.equal(result.tasks[1], done);
  assert.equal(result.currentTaskIndexes, input.currentTaskIndexes);
  assert.equal(result.wait, input.wait); assert.equal(result.extra, "producer metadata");
  assert.deepEqual(taskValue([input, Array.from(records)]), before);
  const projection = c.projectTaskReport(result);
  assert.equal(projection.busy, true);
  assert.deepEqual(plain(projection.taskStates), [["z-active", true], ["a-done", false]]);
  assert.ok(projection.rows.filter((row) => row.taskId === "a-done").every((row) => row.currentTask === true),
    "report provenance remains authoritative over the record's current flag");
});

test("pure task: aging keeps null guards, numeric domains and destructuring failures", () => {
  const c = taskHarness(html).context;
  assert.equal(typeof c.agedTaskReport, "function");
  const poison = { values() { assert.fail("a missing report must not inspect records"); } };
  for (const report of [null, undefined, false, "", 0]) assert.equal(c.agedTaskReport(report, poison, 6000), report);
  const report = Object.freeze({ schema: "ai.task-report", version: 1, tasks: Object.freeze([]) });
  for (const [now, expected] of [[500, 100], [2500, 1600], ["2500", 1600], [NaN, NaN], [Infinity, Infinity]]) {
    const records = new Map([["a", { at: 1000, task: { status: "running", active: true, elapsed_ms: 100, progress: null } }]]);
    assert.equal(c.agedTaskReport(report, records, now).tasks[0].elapsed_ms, expected);
  }
  for (const value of [undefined, null, "100", NaN, Infinity, -Infinity]) {
    const records = new Map([["a", { at: 1000, task: { status: "running", active: true, elapsed_ms: value,
      progress: { stale_for_ms: value } } }]]);
    const result = c.agedTaskReport(report, records, 2500).tasks[0];
    assert.equal(result.elapsed_ms, value); assert.equal(result.progress.stale_for_ms, value);
  }
  const terminal = { task: { status: "completed", active: false }, get at() { throw new Error("inactive timestamp failure"); } };
  assert.throws(() => c.agedTaskReport(report, new Map([["a", terminal]]), 6000), /inactive timestamp failure/);
  assert.throws(() => c.agedTaskReport(report, new Map([["a", null]]), 6000), /null/);
});

for (const [name, run] of taskScenarios) {
  test("characterization task: " + name, () => run(html));
  if (baseline) test("differential task: " + name, async () => {
    assert.deepEqual(await run(html), await run(baseline));
  });
}
