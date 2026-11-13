// Run with: node --test src/bin/ai/serve/app_commands.test.cjs
// Execute the composer's slash-command completion block without a browser.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
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

function loadCommandPreludeFixtures() {
  const filename = path.join(__dirname, "app_desktop.test.cjs");
  // Reuse the existing DOM, draft stores, execution model and real send fixture.
  return new vm.Script(fs.readFileSync(filename, "utf8") + "\n;({harness});", { filename })
    .runInNewContext({
      require: (name) => name === "node:test" ? { test() {} } : require(name),
      __dirname, __filename: filename, console, Buffer, TextDecoder,
      process: { env: { ...process.env, ASERVE_TEST_APP_HTML: process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html") } },
      setTimeout, clearTimeout, setInterval, clearInterval, setImmediate,
    });
}
const commandPreludeFixtures = loadCommandPreludeFixtures();
const commandPlain = (value) => JSON.parse(JSON.stringify(value));
const commandSettle = () => new Promise((resolve) => setImmediate(resolve));
function commandDeferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function commandResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300, status, statusText: "Unavailable",
    headers: { get: (name) => name === "content-disposition" ? 'attachment; filename="session.zip"' : null },
    json: async () => commandPlain(body), blob: async () => ({ type: "application/zip", size: 3 }),
  };
}

function commandPreludeHarness({ sid = "open-session", images = [], respond, readError = false, transfer = true } = {}) {
  const h = commandPreludeFixtures.harness({ sid }), c = h.context;
  const effects = [], requests = [], timers = new Map();
  let timerId = 0, picker = null;
  const slice = (from, to) => {
    const a = script.indexOf(from), b = script.indexOf(to, a);
    assert.ok(a >= 0 && b > a, "missing command prelude slice: " + from);
    return script.slice(a, b);
  };
  for (const [id, tag] of [
    ["gitconfirmwrap", "div"], ["gitconfirmpanel", "section"], ["gitconfirmtext", "div"],
    ["gitcancel", "button"], ["gitaccept", "button"],
  ]) h.elements[id] = h.document.createElement(tag);
  h.elements.gitconfirmwrap.hidden = true;
  h.elements.gitconfirmwrap.appendChild(h.elements.gitconfirmpanel);
  for (const id of ["gitconfirmtext", "gitcancel", "gitaccept"]) h.elements.gitconfirmpanel.appendChild(h.elements[id]);
  h.document.body = h.document.createElement("body");
  for (const id of ["msgs", "composer", "gitconfirmwrap"]) h.document.body.appendChild(h.elements[id]);
  const create = h.document.createElement;
  h.document.createElement = (tag) => {
    const element = create(tag);
    if (tag === "input") {
      picker = element;
      element.handlers = {};
      element.files = null;
      element.addEventListener = (type, listener) => { element.handlers[type] = listener; };
      element.click = () => effects.push(["picker-open"]);
    } else if (tag === "a") element.click = () => effects.push(["download", element.download]);
    return element;
  };
  Object.assign(c, {
    pendingImages: images,
    updateQuestionButton() {},
    shortLine: (text, length) => text.slice(0, length),
    nativeBridgeReady: () => false, navigator: {},
    URL: {
      createObjectURL: () => { effects.push(["object-url"]); return "blob:archive"; },
      revokeObjectURL: (url) => effects.push(["revoke-object-url", url]),
    },
    FileReader: class {
      readAsDataURL(file) {
        effects.push(["read-file", file.name]);
        if (readError) { this.error = new Error("read failed"); this.onerror(); return; }
        this.result = "data:application/zip;base64," + Buffer.from(file.bytes).toString("base64");
        this.onload();
      }
    },
    setTimeout(fn, delay) {
      const id = ++timerId;
      effects.push(["set-timeout", id, delay]); timers.set(id, { fn, delay, kind: "timeout" }); return id;
    },
    clearTimeout(id) { effects.push(["clear-timeout", id]); timers.delete(id); },
    setInterval(fn, delay) {
      const id = ++timerId;
      effects.push(["set-interval", id, delay]); timers.set(id, { fn, delay, kind: "interval" }); return id;
    },
    clearInterval(id) { effects.push(["clear-interval", id]); timers.delete(id); },
    fetch: async (url, opts = {}) => {
      const request = { path: new URL(url).pathname, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : null };
      requests.push(request); effects.push(["request", commandPlain(request)]);
      if (respond) {
        const result = await respond(request);
        if (result) return result;
      }
      if (request.path === "/sessions") return commandResponse(request.method === "POST" ? { id: "fresh-session", cwd: "/srv" } : c.lastItems);
      if (request.path.endsWith("/cd")) return commandResponse({ stdout: "Working directory: /tmp\n", stderr: "", cwd: "/tmp" });
      if (request.path.endsWith("/checkpoints")) return commandResponse({ output: "On branch main\n", confirmation: null });
      if (request.path.endsWith("/export") || request.path.endsWith("/export-md")) return commandResponse({});
      if (request.path === "/sessions/import") return commandResponse({ id: "imported-session" });
      assert.ok(request.path.endsWith("/turns/stream"), "unexpected command request: " + request.path);
      let read = false;
      return { ...commandResponse({}), body: { getReader: () => ({
        read: async () => read ? { done: true } : (read = true, { done: false, value: Buffer.from("event: done\n\n") }),
        cancel() { effects.push(["reader-cancel"]); },
      }) } };
    },
  });
  vm.runInContext(slice("// BEGIN composer cd command", "// END composer cd command"), c);
  vm.runInContext(slice("function addFold(", "// Create a fold only"), c);
  vm.runInContext(slice("// BEGIN composer export/import commands", "// END composer export/import commands"), c);
  const escape = script.indexOf("const esc =");
  vm.runInContext(script.slice(escape, script.indexOf("\n", escape)), c);
  const message = script.indexOf("function addMsg(");
  vm.runInContext(script.slice(message, script.indexOf("// Assistant replies render as markdown", message)), c);
  const addMsg = c.addMsg;
  c.addMsg = (role, text, cls, parent) => {
    h.messages.push({ role, text, cls });
    return addMsg(role, text, cls, parent);
  };
  for (const name of ["setPromptValue", "clearDraft", "closeCommandMenu", "runCdCommand", "runGitCommand", "runTransferCommand", "prepareTurnSession", "scrollToLatest"]) {
    const original = c[name];
    c[name] = (...args) => {
      effects.push([name, ...args.slice(0, name === "setPromptValue" ? 2 : 1).map((arg) => typeof arg === "string" || typeof arg === "boolean" ? arg : null)]);
      return original(...args);
    };
  }
  if (!transfer) c.runTransferCommand = undefined;
  h.elements.prompt.style.height = "61px";
  h.elements.prompt.focus();
  const imageDrafts = vm.runInContext("composerImageDrafts", c);
  function input(text, { stored = true } = {}) {
    h.elements.prompt.value = text;
    if (stored) c.persistDraft();
    else c.clearDraft(c.sessionId);
  }
  function pick(file) {
    assert.ok(picker, "the archive picker is open");
    effects.push(["picker-result", file ? file.name : null]);
    if (file) { picker.files = [file]; picker.handlers.change(); }
    else picker.handlers.cancel();
  }
  function resetEffects() { effects.length = 0; requests.length = 0; h.messages.length = 0; }
  return { ...h, effects, requests, timers, images, imageDrafts, input, pick, resetEffects, picker: () => picker,
    draft: (id = c.sessionId) => h.drafts.get(c.draftKey(id)),
    hasDraft: (id = c.sessionId) => h.drafts.has(c.draftKey(id)),
    hasImageDraft: (id = c.sessionId) => imageDrafts.has(c.draftKey(id)),
    send(text, options) { input(text, options); resetEffects(); return c.send(); },
  };
}

for (const [name, text, endpoint] of [
  ["cd", "  /cd /tmp  ", "/cd"], ["git", "  /git status  ", "/checkpoints"], ["transfer", "  /export  ", "/export"],
]) test("send command prelude: " + name + " clears its draft before its endpoint and returns without a turn", async () => {
  const reply = commandDeferred(), requested = commandDeferred();
  const h = commandPreludeHarness({ respond: (request) => { if (request.path.endsWith(endpoint)) { requested.resolve(); return reply.promise; } } });
  const sending = h.send(text);
  await requested.promise;
  assert.equal(h.context.sendBusy, true);
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.elements.prompt.style.height, "auto");
  assert.equal(h.hasDraft(), false);
  assert.equal(h.hasImageDraft(), false);
  const clear = h.effects.findIndex(([kind]) => kind === "clearDraft");
  const request = h.effects.findIndex(([kind]) => kind === "request");
  assert.ok(clear >= 0 && clear < request, "draft clearing precedes the command request");
  reply.resolve(commandResponse({ stdout: "Working directory: /tmp\n", output: "On branch main\n", confirmation: null }));
  await sending;
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].path, "/sessions/open-session" + endpoint);
  assert.equal(h.effects.some(([kind]) => kind === "prepareTurnSession"), false);
  assert.equal(h.context.sendBusy, false);
  assert.equal(h.messages[0].text, name === "transfer" ? text : text.trim());
  assert.equal(h.messages.some((message) => message.role === "err"), false);
  if (name !== "transfer") assert.ok(h.elements.msgs.children.some((node) => node.classList.contains("tool-activity")));
});

for (const text of ["  hello  ", "/cdx", "/github", "/exportx", "/git status\n", "/export\nexplain"]) {
  test("send command prelude: unmatched " + JSON.stringify(text) + " probes cd, git and transfer in order", async () => {
    const h = commandPreludeHarness();
    await h.send(text);
    assert.deepEqual(h.effects.filter(([kind]) => /^run(Cd|Git|Transfer)Command$/.test(kind) || kind === "prepareTurnSession").map(([kind]) => kind),
      ["runCdCommand", "runGitCommand", "runTransferCommand", "prepareTurnSession"]);
    assert.deepEqual(h.requests.map((request) => request.path), ["/sessions/open-session/turns/stream", "/sessions"]);
    assert.equal(h.requests[0].path, "/sessions/open-session/turns/stream");
    assert.equal(h.requests[0].body.prompt, text);
    assert.equal(h.elements.prompt.value, "");
    assert.equal(h.hasDraft(), false);
    assert.equal(h.context.sendBusy, false);
  });
}

test("send command prelude: cd with an attachment falls through and consumes the normal turn input", async () => {
  const images = [{ filename: "photo.png", data: "image" }], h = commandPreludeHarness({ images });
  await h.send("/cd /tmp");
  assert.equal(h.requests[0].path, "/sessions/open-session/turns/stream");
  assert.equal(h.requests[0].body.prompt, "/cd /tmp\n[[image:photo.png]]");
  assert.deepEqual(h.requests[0].body.images, images);
  assert.equal(h.context.pendingImages.length, 0);
  assert.equal(h.hasDraft(), false);
  assert.equal(h.hasImageDraft(), false);
});

for (const text of ["/git status", "/export", "/export-md notes.md"]) {
  test("send command prelude: " + text + " retains pending attachment identities while clearing draft storage", async () => {
    const images = [{ filename: "photo.png", data: "image" }], h = commandPreludeHarness({ images });
    await h.send(text);
    assert.equal(h.requests.length, 1);
    assert.equal(h.requests[0].path.endsWith("/turns/stream"), false);
    assert.equal(h.context.pendingImages, images);
    assert.equal(h.context.pendingImages[0], images[0]);
    assert.equal(h.elements.prompt.value, "");
    assert.equal(h.hasDraft(), false);
    assert.equal(h.hasImageDraft(), false);
  });
}

for (const stored of [false, true]) test("send command prelude: canceling import preserves " + (stored ? "the stored" : "an unstored") + " draft and attachments", async () => {
  const images = [{ filename: "photo.png", data: "image" }], h = commandPreludeHarness({ images });
  const sending = h.send("  /import as=copy  ", { stored });
  await commandSettle();
  assert.ok(h.picker());
  assert.equal(h.context.sendBusy, true);
  assert.equal(h.elements.prompt.value, "  /import as=copy  ");
  assert.equal(h.elements.prompt.style.height, "61px");
  assert.equal(h.hasDraft(), stored);
  assert.equal(h.hasImageDraft(), stored);
  h.pick(null);
  await sending;
  assert.equal(h.elements.prompt.value, "  /import as=copy  ");
  assert.equal(h.hasDraft(), stored);
  assert.equal(h.context.pendingImages, images);
  assert.equal(h.effects.some(([kind]) => kind === "clearDraft" || kind === "setPromptValue" || kind === "prepareTurnSession"), false);
  assert.equal(h.requests.length, 0);
  assert.equal(h.messages.length, 0);
  assert.equal(h.context.sendBusy, false);
});

for (const [text, sid, error] of [
  ["/import as=", "open-session", "import failed: empty as= id"],
  ["/export last", "open-session", "/export last is not available"],
  ["/export", "", "export failed: no session selected"],
]) test("send command prelude: " + text + " validation error retains input and draft", async () => {
  const h = commandPreludeHarness({ sid });
  await h.send(text);
  assert.equal(h.elements.prompt.value, text);
  assert.equal(h.elements.prompt.style.height, "61px");
  assert.equal(h.draft(), text);
  assert.equal(h.requests.length, 0);
  assert.equal(h.picker(), null);
  assert.equal(h.effects.some(([kind]) => kind === "clearDraft" || kind === "setPromptValue" || kind === "prepareTurnSession"), false);
  assert.ok(h.messages.some((message) => message.role === "err" && message.text.startsWith(error)));
  assert.equal(h.context.sendBusy, false);
});

for (const cancel of ["button", "Escape", "scrim"]) test("send command prelude: git " + cancel + " cancels after the submitted draft was cleared", async () => {
  const images = [{ filename: "photo.png", data: "image" }];
  const h = commandPreludeHarness({ images, respond: () => commandResponse({ output: "", confirmation: "Discard?" }) });
  const sending = h.send("/git clear");
  await commandSettle();
  assert.equal(h.elements.gitconfirmwrap.hidden, false);
  assert.equal(h.document.activeElement, h.elements.gitaccept);
  assert.equal(h.context.sendBusy, true);
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.hasDraft(), false);
  assert.equal(h.context.pendingImages, images);
  h.input("next unsent draft");
  if (cancel === "button") h.elements.gitcancel.onclick();
  else if (cancel === "scrim") h.elements.gitconfirmwrap.onclick({ target: h.elements.gitconfirmwrap });
  else h.elements.gitconfirmpanel.onkeydown({ key: "Escape", preventDefault() {}, stopPropagation() {} });
  await sending;
  assert.equal(h.requests.length, 1);
  assert.equal(h.elements.prompt.value, "next unsent draft");
  assert.equal(h.draft(), "next unsent draft");
  assert.equal(h.elements.gitconfirmwrap.hidden, true);
  assert.equal(h.document.activeElement, h.elements.prompt);
  assert.equal(h.messages.at(-1).text, "Canceled: nothing was changed.");
  assert.equal(h.context.sendBusy, false);
});

for (const text of ["/cd /tmp", "/git status"]) test("send command prelude: lazy creation failure for " + text + " preserves the original draft", async () => {
  const h = commandPreludeHarness({ sid: "", respond: () => commandResponse({ error: "cannot create" }, 503) });
  await h.send(text);
  assert.equal(h.requests.length, 1);
  assert.equal(h.context.sessionId, "");
  assert.equal(h.elements.prompt.value, text);
  assert.equal(h.draft(), text);
  assert.equal(h.effects.some(([kind]) => kind === "clearDraft" || kind === "setPromptValue"), false);
  assert.ok(h.messages.some((message) => message.text.includes("create session failed:")));
  assert.equal(h.context.sendBusy, false);
});

for (const readError of [false, true]) test("send command prelude: import " + (readError ? "read failure" : "upload") + " clears only after a file is picked", async () => {
  const images = [{ filename: "photo.png", data: "image" }], h = commandPreludeHarness({ images, readError });
  const sending = h.send("/import as=copy");
  await commandSettle();
  assert.equal(h.elements.prompt.value, "/import as=copy");
  assert.equal(h.hasDraft(), true);
  h.pick({ name: "archive.zip", bytes: "zip-bytes" });
  await sending;
  const picked = h.effects.findIndex(([kind]) => kind === "picker-result");
  const clear = h.effects.findIndex(([kind]) => kind === "clearDraft");
  const read = h.effects.findIndex(([kind]) => kind === "read-file");
  assert.ok(picked < clear && clear < read, "the archive selection precedes clearing, which precedes reading");
  assert.equal(h.elements.prompt.value, "");
  if (readError) {
    assert.equal(h.hasDraft("open-session"), false);
    assert.equal(h.hasImageDraft("open-session"), false);
    assert.equal(h.context.pendingImages, images);
    assert.equal(h.requests.length, 0);
    assert.equal(h.context.sessionId, "open-session");
    assert.equal(h.messages.at(-1).text, "import failed: read failed");
  } else {
    assert.equal(h.requests[0].path, "/sessions/import");
    assert.deepEqual(h.requests[0].body, { filename: "archive.zip", data: Buffer.from("zip-bytes").toString("base64"), id: "copy" });
    assert.equal(h.context.sessionId, "imported-session");
    // switchSession saves the cleared outgoing input and its remaining images.
    assert.equal(h.draft("open-session"), "");
    assert.equal(h.imageDrafts.get(h.context.draftKey("open-session")), images);
    assert.equal(h.context.pendingImages.length, 0);
  }
  assert.equal(h.context.sendBusy, false);
});

test("send command prelude: an unavailable transfer bridge falls through after cd and git", async () => {
  const h = commandPreludeHarness({ transfer: false });
  await h.send("/export");
  assert.equal(h.requests[0].path, "/sessions/open-session/turns/stream");
  assert.deepEqual(h.effects.filter(([kind]) => /^run(Cd|Git|Transfer)Command$/.test(kind)).map(([kind]) => kind), ["runCdCommand", "runGitCommand"]);
});

test("send command prelude: an unexpected command rejection releases sendBusy without clearing the input", async () => {
  const h = commandPreludeHarness();
  h.context.runCdCommand = async () => { throw new Error("command rejection"); };
  await assert.rejects(h.send("hello"), /command rejection/);
  assert.equal(h.context.sendBusy, false);
  assert.equal(h.elements.prompt.value, "hello");
  assert.equal(h.draft(), "hello");
  assert.equal(h.requests.length, 0);
});
