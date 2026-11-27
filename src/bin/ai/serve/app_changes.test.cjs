// Run with: node --test src/bin/ai/serve/app_changes.test.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);

function between(from, to, source = script) {
  const start = source.indexOf(from), end = source.indexOf(to, start);
  assert.ok(start >= 0 && end > start, "missing source slice: " + from);
  return source.slice(start, end);
}
function dom() {
  const elements = new Map(), captures = new Map();
  function node(tagName) {
    const handlers = new Map();
    const element = {
      tagName, hidden: false, children: [], attributes: {}, dataset: {}, className: "", parent: null,
      style: { setProperty(key, value) { this[key] = value; }, removeProperty(key) { delete this[key]; } },
      clientWidth: 520, clientHeight: 480, offsetWidth: 100, offsetHeight: 44,
      getBoundingClientRect() { return this.bounds ?? { left: 16, top: 48, bottom: 92 }; },
      get parentNode() { return this.parent; },
      get firstElementChild() { return this.children.find((child) => !child.tagName.startsWith("#")) ?? null; },
      set tabIndex(value) { this.setAttribute("tabindex", value); },
      appendChild(child) {
        if (child.tagName === "#fragment") { this.append(...child.children); return child; }
        child.parent = this;
        this.children.push(child);
        return child;
      },
      removeChild(child) { this.children.splice(this.children.indexOf(child), 1); child.parent = null; },
      append(...children) { for (const child of children) this.appendChild(typeof child === "string" ? document.createTextNode(child) : child); },
      replaceChildren(...children) { this.textContent = ""; this.append(...children); },
      setAttribute(key, value) { this.attributes[key] = String(value); },
      getAttribute(key) { return this.attributes[key] ?? null; },
      contains(other) { return this === other || this.children.some((child) => child.contains(other)); },
      closest(selector) {
        const parts = selector.split(", ");
        assert.ok(parts.every((part) => /^(?:[\w-]+|\.[\w-]+|\[[\w-]+\])$/.test(part)), `unsupported closest selector: ${selector}`);
        const match = (item, part) => {
          if (part === "[hidden]") return item.hidden;
          if (part === "[data-changes-ref]") return Object.hasOwn(item.dataset, "changesRef");
          if (part.startsWith("[")) return item.getAttribute(part.slice(1, -1)) !== null;
          if (part.startsWith(".")) return item.className.split(/\s+/).includes(part.slice(1));
          return item.tagName === part;
        };
        for (let item = this; item; item = item.parent) if (parts.some((part) => match(item, part))) return item;
        return null;
      },
      querySelectorAll(selector) {
        if (selector === ".msg.md") return [];
        const selectors = selector.split(", ");
        assert.ok(selectors.every((part) => ["button", "select", "textarea", '[tabindex="0"]'].includes(part)));
        return this.children.flatMap((child) => [
          ...(selectors.includes(child.tagName) || child.getAttribute("tabindex") === "0" ? [child] : []),
          ...child.querySelectorAll(selector),
        ]);
      },
      focus() { if (!this.closest("[hidden]")) document.activeElement = this; },
      select() { assert.equal(this.tagName, "textarea"); this.focus(); },
      setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; },
      setPointerCapture(id) { captures.set(id, this); },
      hasPointerCapture(id) { return captures.get(id) === this; },
      releasePointerCapture(id) {
        if (!this.hasPointerCapture(id)) return;
        captures.delete(id);
        this.dispatch("lostpointercapture", { pointerId: id });
      },
      addEventListener(type, handler, capture = false) {
        if (!handlers.has(type)) handlers.set(type, []);
        handlers.get(type).push({ handler, capture });
      },
      removeEventListener(type, handler, capture = false) {
        const listeners = handlers.get(type);
        const index = listeners?.findIndex((entry) => entry.handler === handler && entry.capture === capture) ?? -1;
        if (index >= 0) listeners.splice(index, 1);
      },
      dispatch(type, fields = {}) {
        const capture = ["pointermove", "pointerup", "pointercancel"].includes(type) && captures.get(fields.pointerId);
        const target = capture || this;
        const event = { target, ...fields,
          preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; },
        };
        const chain = [];
        for (let item = target; item; item = item.parent) chain.push(item);
        for (const item of [...chain].reverse()) {
          for (const { handler, capture } of item._handlers.get(type) || []) if (capture) handler(event);
          if (event.stopped) return event;
        }
        for (const item of type === "scroll" ? [this] : chain) {
          item["on" + type]?.(event);
          for (const { handler, capture } of item._handlers.get(type) || []) if (!capture) handler(event);
          if (event.stopped) break;
        }
        return event;
      },
      _handlers: handlers,
      get textContent() { return (this.text || "") + this.children.map((child) => child.textContent).join(""); },
      set textContent(text) {
        for (const child of this.children) child.parent = null;
        this.children = [];
        this.text = String(text);
      },
      set innerHTML(_html) { assert.fail("Changes must render dynamic content as text"); },
    };
    if (tagName === "textarea") element.value = "";
    element.classList = { toggle(name, force) {
      const classes = new Set(element.className.split(/\s+/).filter(Boolean));
      if (force) classes.add(name);
      else classes.delete(name);
      element.className = [...classes].join(" ");
    } };
    return element;
  }
  const document = node("#document");
  document.createElement = node;
  document.createDocumentFragment = () => node("#fragment");
  document.createTextNode = (text) => Object.assign(node("#text"), { text, closest: undefined });
  document.body = document.appendChild(node("body"));
  // Read the static trees so hidden ancestry, focus order, and event propagation match the markup.
  function mount(markup) {
    const stack = [document.body];
    for (const [token, tag, attrs, text] of markup.matchAll(/<\/?([\w-]+)([^>]*?)>|([^<]+)/g)) {
      if (!tag) { stack.at(-1).appendChild(document.createTextNode(text)); continue; }
      if (token.startsWith("</")) { assert.equal(stack.pop().tagName, tag); continue; }
      const element = node(tag);
      for (const [, key, value] of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
        if (key === "id") elements.set(value, element);
        else if (key === "hidden") element.hidden = true;
        else if (key === "class") element.className = value;
        else element.setAttribute(key, value ?? "");
      }
      stack.at(-1).appendChild(element);
      if (!token.endsWith("/>")) stack.push(element);
    }
    assert.equal(stack.length, 1);
  }
  mount(between('<div id="sessbar"', '<div id="conn">', html));
  mount(`<div id="composer">${between('<textarea id="prompt"', '<button id="sidenote"', html)}</div>`);
  mount(between('<div id="changes"', '<script>', html));
  for (const id of ["msgs", "info"]) elements.set(id, node("div"));
  return { document, node, $: (id) => elements.get(id) ?? null };
}

const base = "https://example.invalid", token = "mock-token";
const defaultId = "s/α+%";
const enabled = { features: { changes: true } };
const commitsEnabled = { features: { changes: true, changes_commits: true } };
const file = (id, path, overrides = {}) => ({ id, path, old_path: null, status: "modified", additions: 3, deletions: 1, binary: false, ...overrides });
const manifest = (files = [], overrides = {}) => ({ source: "working_tree", git_available: true,
  root_label: "sample repo", revision: "rev/α?= +%", files, truncated: false, reason: null, ...overrides });
const diff = (list, item, overrides = {}) => ({ revision: list.revision, file_id: item.id, path: item.path, status: item.status, format: "unified", diff: "", truncated: false, reason: null, ...overrides });
const scope = (id, label, { current = false, files = 0 } = {}) => ({ id, label, path: "/repo/" + label, current, changed: files > 0, files });
const mainScope = scope("current/α+%", "session repo", { current: true });
const scopeInfo = ({ id, label, path, current }) => ({ id, label, path, current });
const scopedManifest = (scope, files = [], overrides = {}) => manifest(files, { scope: scopeInfo(scope), ...overrides });
const catalog = (...scopes) => ({ source: "git", git_available: true, scopes, truncated: false, reason: null });
const headCommit = { id: "ab123cd".padEnd(40, "e"), subject: "Fix <README> & keep a literal subject", committed_at: 1761940800 };
const olderCommit = { id: "dfe420a".padEnd(40, "b"), subject: "Add sample file", committed_at: 1761854400 };
const commitHistory = (scope, rows, overrides = {}) => ({ source: "commit_history", scope: scopeInfo(scope), branch: "main",
  head: rows[0]?.id ?? null, rows, next_cursor: null, truncated: false, reason: null, ...overrides });
const commitDiff = (scope, item, overrides = {}) => ({ source: "commit", scope: scopeInfo(scope), commit_id: item.id,
  base: "first_parent", base_id: olderCommit.id, format: "unified", diff: "", truncated: false, reason: null, ...overrides });
const listPath = (sid = defaultId) => "/sessions/" + encodeURIComponent(sid) + "/changes";
const scopesPath = (sid = defaultId) => listPath(sid) + "/scopes";
const scopedListPath = (scope, sid) => listPath(sid) + "?scope=" + encodeURIComponent(scope.id);
const detailPath = (list, item) => listPath() + "/" + encodeURIComponent(item.id) + "?revision=" + encodeURIComponent(list.revision)
  + (list.scope ? "&scope=" + encodeURIComponent(list.scope.id) : "");
const commitsPath = (scope, cursor, sid) => listPath(sid) + "/commits?scope=" + encodeURIComponent(scope.id) + "&limit=25"
  + (cursor ? "&before=" + encodeURIComponent(cursor) : "");
const commitPath = (scope, item, sid) => listPath(sid) + "/commits/" + encodeURIComponent(item.id) + "?scope=" + encodeURIComponent(scope.id);
const reply = (body, status = 200) => ({ ok: status >= 200 && status < 300, status, statusText: "HTTP error", json: async () => body });
function deferred() {
  let resolve;
  const promise = new Promise((yes) => { resolve = yes; });
  return { promise, resolve };
}
async function settle() { await new Promise((resolve) => setImmediate(resolve)); }
function harness({ phone = false, finePointer = !phone, visualViewport, storage = new Map(),
  sid = defaultId, list = manifest(), respond, clipboard, execCopy, confirm = () => assert.fail("unexpected confirmation") } = {}) {
  const { document, node, $ } = dom();
  const window = node("#window"), entries = [{ base: true }], travel = [], calls = [], copied = [];
  Object.assign(window, { innerWidth: phone ? 390 : 1440, innerHeight: 1000, confirm });
  // The DOM fixture does not resolve CSS, so reflect inline resizing in measured panel bounds.
  const panel = $("changes-panel");
  panel.getBoundingClientRect = () => {
    const width = parseFloat(panel.style["--changes-width"] ?? "1080px");
    const height = parseFloat(panel.style["--changes-height"] ?? "760px");
    return panel.bounds ?? { left: 16, top: 48, bottom: 48 + height, width, height };
  };
  if (visualViewport) window.visualViewport = Object.assign(node("#visual-viewport"), visualViewport);
  Object.defineProperty($("changes-body"), "clientWidth", { get() {
    const viewport = window.visualViewport ?? { width: window.innerWidth, height: window.innerHeight };
    const gap = parseFloat($("changes").style["--changes-viewport-gap"] ?? "24px");
    const grip = Math.min(20, viewport.height / 8);
    return Math.round(Math.min(panel.getBoundingClientRect().width, viewport.width - 2 * gap) - grip - 2);
  } });
  const timers = new Map();
  const navigator = { clipboard: clipboard === undefined ? { writeText: async (text) => { copied.push(text); } } : clipboard };
  if (execCopy) document.execCommand = (command) => execCopy(command, document.activeElement);
  let index = 0, info = enabled, drawerOpens = 0;
  $("sessbar").onclick = () => drawerOpens++;
  const history = {
    get state() { return entries[index]; },
    pushState(state) { entries.splice(++index, entries.length, state); },
    go(delta) { travel.push(delta); },
  };
  const resizeMedia = node("#resize-media");
  Object.defineProperty(resizeMedia, "matches", { get: () => finePointer && window.innerWidth >= 701 });
  window.matchMedia = (query) => {
    if (query === "(max-width: 700px)") return { matches: window.innerWidth <= 700 };
    assert.equal(query, "(min-width:701px) and (pointer:fine)");
    return resizeMedia;
  };
  const noop = () => {};
  const context = vm.createContext({
    $, document, window, history, navigator, AbortController, sessionId: sid, store: { base, token },
    setTimeout(fn, delay) { const id = Symbol(); timers.set(id, { fn, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, { method = "GET", headers, signal, cache, body }) {
      const request = { url, path: url.slice(context.store.base.length), method, headers, signal, cache, body };
      calls.push(request);
      const fallback = request.path === "/info" ? info : request.path.endsWith("/changes/scopes") ? catalog() : list;
      return Promise.resolve(respond?.(request) ?? reply(fallback));
    },
    fillSelect: noop, syncModelTrigger: noop, syncPickTrigger: noop, syncOvrSummary: noop, updateAnswerReference: noop,
    pauseHistoryRetry: noop, abortEvents: noop, updateExecutionModel: noop, ensureTranscriptOwner: noop,
    syncStopButton: noop, restoreDraft: noop, resetQuestionNavigator: noop, restoreSkillPick: noop, restoreTeamPick: noop,
    localStorage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, String(value)),
      removeItem: (key) => storage.delete(key),
    },
    booted: false, infoRetryTimer: null,
  });
  for (const region of [
    between("function headers(json)", "const esc ="),
    between("function copyCodeText(", "let replyReferenceVersion"),
    between("// BEGIN working-tree changes", "// END working-tree changes"),
    between("async function loadInfo()", "// The drawer only highlights"),
    between("// BEGIN session switch", "// END session switch"),
    script.slice(script.lastIndexOf("function isPhoneLayout()"), script.indexOf("// Dragging across the drawer")),
  ]) vm.runInContext(region, context);
  context.wireChanges();
  function flushHistory() {
    while (travel.length) {
      index += travel.shift();
      assert.ok(index >= 0 && index < entries.length, "viewer must not navigate away from the app");
      window.dispatch("popstate", { state: history.state });
    }
  }
  return {
    $, context, calls, copied, timers, document, window, history, flushHistory,
    rows: () => $("changes-files").children.map((item) => item.children[0]),
    commitRows: () => $("changes-commits").children,
    scopeOptions: () => $("changes-scope-options").children,
    drawerOpens: () => drawerOpens,
    async info(payload = enabled) { info = payload; await context.loadInfo(); },
    click: (id) => $(id).dispatch("click"),
    setScope(scope) {
      if ($("changes-scope-options").hidden) $("changes-scope").dispatch("click");
      const option = this.scopeOptions().find((option) => option.value === scope.id);
      assert.ok(option, "scope is available in the picker");
      option.dispatch("click");
    },
    key: (key, extra) => document.activeElement.dispatch("keydown", { key, ...extra }),
    back() { history.go(-1); flushHistory(); },
    resize(value) { window.innerWidth = value ? 390 : 1440; window.dispatch("resize"); },
  };
}
async function opened(options, info = enabled) {
  const h = harness(options);
  await h.info(info);
  h.click("changesbtn");
  return h;
}
async function openCommitList(h) {
  h.click("changes-mode-committed");
  await settle();
}
function openRefMenu(h, node, point = { clientX: 100, clientY: 120 }) {
  assert.equal(node.dispatch("contextmenu", point).defaultPrevented, true, "the name opens the app's copy menu");
  const menu = h.$("changes-copy-ref");
  assert.equal(menu.hidden, false);
  assert.equal(h.document.activeElement, menu);
  return menu;
}
async function copyRef(h, node, anchor = node) {
  const menu = openRefMenu(h, node);
  menu.dispatch("click");
  assert.equal(menu.hidden, true);
  assert.equal(h.document.activeElement, anchor);
  await settle();
}

const changesSizeKey = "serve.changesSize";
const grip = (h, edge = "corner") => h.$("changes-resize-" + edge);
const panelSize = (h) => ["width", "height"].map((axis) => h.$("changes-panel").style["--changes-" + axis]);
function startChangesDrag(h, edge = "corner") {
  const handle = grip(h, edge);
  assert.ok(handle, `the ${edge} grip exists`);
  const pointer = { pointerId: 7, pointerType: "mouse", button: 0, isPrimary: true };
  const send = (type, dx, dy, target = handle) => target.dispatch(type, { ...pointer, clientX: 1200 + dx, clientY: 820 + dy });
  assert.equal(send("pointerdown", 0, 0).defaultPrevented, true);
  assert.equal(handle.hasPointerCapture(pointer.pointerId), true);
  return send;
}

test("Changes resizing: grips appear only on wide fine-pointer layouts", async () => {
  const storage = new Map([[changesSizeKey, JSON.stringify({ width: 820, height: 700 })]]);
  const desktop = await opened({ storage });
  const handles = [["right", "ew-resize"], ["bottom", "ns-resize"], ["corner", "nwse-resize"]];
  for (const [edge] of handles) {
    const handle = grip(desktop, edge);
    assert.ok(handle, `the desktop ${edge} grip exists`);
    assert.equal(handle.closest("[hidden]"), null);
  }
  const css = between('<style id="changes-style">', '</style>', html);
  assert.match(css, /\.changes-resize-grip\s*\{\s*display:none;/);
  const resizeRules = between("@media (min-width:701px) and (pointer:fine)", "@media (max-width:700px)", css);
  for (const [edge, cursor] of handles) {
    assert.match(resizeRules, new RegExp(`#changes-resize-${edge}\\s*\\{[^}]*cursor:${cursor}`));
  }
  const corner = grip(desktop);
  assert.match(corner.getAttribute("aria-label"), /arrow keys.*double-click/i);
  assert.match(corner.getAttribute("title"), /double-click to reset/i);
  assert.equal(corner.dispatch("pointerdown", { pointerId: 1, pointerType: "touch", button: 0 }).defaultPrevented, undefined);
  assert.equal(corner.hasPointerCapture(1), false);
  desktop.$("changes-refresh").focus();
  desktop.key("Tab");
  assert.equal(desktop.document.activeElement, corner);
  desktop.resize(true);
  assert.equal(desktop.document.activeElement, desktop.$("changes-close"));
  assert.equal(corner.hidden, true);
  assert.deepEqual(panelSize(desktop), [undefined, undefined]);
  for (const options of [{ phone: true }, { phone: true, finePointer: true }, { finePointer: false }]) {
    const h = await opened({ ...options, storage });
    for (const [edge] of handles) {
      const handle = grip(h, edge);
      assert.equal(handle.hidden, true);
      assert.equal(handle.dispatch("pointerdown", { pointerId: 1, pointerType: "mouse", button: 0 }).defaultPrevented, undefined);
      assert.equal(handle.hasPointerCapture(1), false);
    }
    assert.deepEqual(panelSize(h), [undefined, undefined]);
  }
});

test("Changes resizing: a captured corner drag changes width and height live", async () => {
  const storage = new Map(), h = await opened({ storage });
  const send = startChangesDrag(h);
  assert.deepEqual(panelSize(h), [undefined, undefined]);
  assert.equal(send("pointermove", -50, -40).defaultPrevented, true);
  assert.deepEqual(panelSize(h), ["980px", "680px"]);
  assert.equal(storage.has(changesSizeKey), false, "live movement does not repeatedly write to storage");
  send("pointerup", -60, -50);
  assert.deepEqual(panelSize(h), ["960px", "660px"]);
  assert.equal(grip(h).hasPointerCapture(7), false);
  startChangesDrag(h)("pointerup", 1200, 820);
  assert.deepEqual(panelSize(h), ["1392px", "952px"], "dragging out cannot overrun the viewport");
});

test("Changes resizing: inward drags stop at the minimum size", async () => {
  const h = await opened();
  const send = startChangesDrag(h);
  send("pointermove", -1200, -820);
  assert.deepEqual(panelSize(h), ["360px", "320px"]);
  send("pointerup", -1200, -820);
  assert.deepEqual(panelSize(h), ["360px", "320px"]);
});

test("Changes resizing: dimensions restore after reopening and double-click resets them", async () => {
  const storage = new Map(), h = await opened({ storage });
  startChangesDrag(h)("pointerup", -90, -60);
  assert.deepEqual(JSON.parse(storage.get(changesSizeKey)), { width: 900, height: 640 });
  h.click("changes-close");
  h.flushHistory();
  h.click("changesbtn");
  assert.deepEqual(panelSize(h), ["900px", "640px"]);
  const reloaded = await opened({ storage });
  assert.deepEqual(panelSize(reloaded), ["900px", "640px"]);
  assert.equal(grip(reloaded).dispatch("dblclick", { detail: 2 }).defaultPrevented, true);
  assert.deepEqual(panelSize(reloaded), [undefined, undefined]);
  assert.equal(storage.has(changesSizeKey), false);
});

test("Changes resizing: releasing over the scrim does not close the dialog", async () => {
  const h = await opened();
  await settle();
  const send = startChangesDrag(h), scrim = h.$("changes-scrim");
  send("pointermove", -70, -80, scrim);
  const release = send("pointerup", -70, -80, scrim);
  assert.equal(release.target, grip(h), "capture retargets an outside release to the grip");
  release.target.dispatch("click", { detail: 1 });
  assert.deepEqual(panelSize(h), ["940px", "600px"]);
  assert.equal(h.$("changes").hidden, false);
  for (const id of ["changes-list-scroll", "changes-detail-scroll"]) {
    assert.equal(h.$(id).dispatch("wheel").defaultPrevented, undefined);
  }
  await copyRef(h, h.$("changes-root"));
  assert.deepEqual(h.copied, ["sample repo"]);
  scrim.dispatch("click", { detail: 1 });
  assert.equal(h.$("changes").hidden, true, "an ordinary scrim click still dismisses the dialog");
});

test("Changes resizing: saved dimensions clamp to the visual viewport before and after resize", async () => {
  const saved = JSON.stringify({ width: 1400, height: 1000 }), storage = new Map([[changesSizeKey, saved]]);
  const h = await opened({ storage, visualViewport: { width: 1000, height: 620, offsetLeft: 200, offsetTop: 130 } });
  const { visualViewport } = h.window, { style } = h.$("changes");
  assert.deepEqual(panelSize(h), ["952px", "572px"]);
  assert.equal(style["--changes-viewport-left"], "200px");
  assert.equal(style["--changes-viewport-top"], "130px");
  Object.assign(visualViewport, { width: 800, height: 440 });
  h.window.dispatch("resize");
  assert.deepEqual(panelSize(h), ["752px", "392px"]);
  Object.assign(visualViewport, { width: 300, height: 280 });
  visualViewport.dispatch("resize");
  assert.deepEqual(panelSize(h), ["252px", "232px"], "available space takes priority over the minimum size");
  visualViewport.offsetLeft = 230;
  visualViewport.dispatch("scroll");
  assert.equal(style["--changes-viewport-left"], "230px");
  assert.equal(storage.get(changesSizeKey), saved);
});

test("Changes resizing: edge drags affect one axis and keyboard resizing respects Shift and Escape", async () => {
  const h = await opened();
  const right = startChangesDrag(h, "right");
  right("pointermove", -50, -100);
  right("pointercancel", -50, -100);
  assert.equal(grip(h, "right").hasPointerCapture(7), false);
  assert.deepEqual(panelSize(h), ["980px", "760px"]);
  startChangesDrag(h, "bottom")("pointerup", -100, -50);
  assert.deepEqual(panelSize(h), ["980px", "660px"]);
  assert.equal(h.document.activeElement, grip(h));
  assert.equal(h.key("ArrowLeft").defaultPrevented, true);
  assert.equal(h.key("ArrowUp", { shiftKey: true }).defaultPrevented, true);
  assert.deepEqual(panelSize(h), ["970px", "620px"]);
  grip(h).dispatch("click", { detail: 0 }); // Native keyboard activation of the reset button.
  assert.deepEqual(panelSize(h), [undefined, undefined]);
  startChangesDrag(h);
  assert.equal(h.key("Escape").defaultPrevented, true);
  assert.equal(h.$("changes").hidden, true);
  assert.equal(grip(h).hasPointerCapture(7), false);
});

const changesPositionKey = "serve.changesPosition";
const changesHeading = (h) => h.$("changes-title").parentNode;
const changesHeader = (h) => changesHeading(h).parentNode;
const panelOffset = (h) => h.$("changes-panel").style["--changes-position"];
function changesMovePointer(target) {
  return (type, dx = 0, dy = 0, hit = target, fields) => hit.dispatch(type, {
    pointerId: 7, pointerType: "mouse", button: 0, isPrimary: true, clientX: 720 + dx, clientY: 150 + dy, ...fields,
  });
}
function startChangesMove(h, target = changesHeading(h)) {
  const send = changesMovePointer(target);
  assert.equal(send("pointerdown").defaultPrevented, undefined);
  assert.equal(changesHeader(h).hasPointerCapture(7), false);
  return send;
}

for (const [surface, targets] of [
  ["the title text", (h) => [h.$("changes-title"), h.$("changes-title").children[0]]],
  ["the subtitle/origin text", (h) => [h.$("changes-origin-label").parentNode, h.$("changes-origin-label")]],
  ["blank header padding", (h) => [changesHeader(h)]],
]) {
  test(`Changes moving: dragging ${surface} moves the dialog and double-clicking recenters`, async () => {
    const storage = new Map(), h = await opened({ storage });
    await settle();
    for (const target of targets(h)) {
      const click = startChangesMove(h, target);
      assert.equal(click("pointermove", 3).defaultPrevented, undefined);
      assert.equal(click("mousemove", 3).defaultPrevented, undefined);
      assert.equal(panelOffset(h), undefined);
      assert.equal(changesHeader(h).hasPointerCapture(7), false);
      assert.equal(click("pointerup", 3).defaultPrevented, undefined);
      assert.equal(storage.has(changesPositionKey), false);

      const drag = startChangesMove(h, target);
      assert.equal(drag("pointermove", 4).defaultPrevented, true);
      assert.equal(panelOffset(h), "4px 0px");
      assert.equal(changesHeader(h).hasPointerCapture(7), true);
      assert.equal(drag("mousemove", 4).defaultPrevented, true);
      drag("pointerup", 24, -12);
      assert.equal(panelOffset(h), "24px -12px");
      assert.equal(drag("mousemove", 24, -12).defaultPrevented, undefined);
      assert.deepEqual(JSON.parse(storage.get(changesPositionKey)), { x: 24, y: -12 });

      assert.equal(target.dispatch("dblclick", { detail: 2 }).defaultPrevented, true);
      assert.equal(panelOffset(h), undefined);
      assert.equal(storage.has(changesPositionKey), false);
    }
  });
}

test("Changes moving: the header drags live and a release over the scrim does not dismiss", async () => {
  const storage = new Map(), h = await opened({ storage });
  const send = startChangesMove(h), scrim = h.$("changes-scrim");
  assert.equal(send("pointermove", 60, -40, scrim).defaultPrevented, true);
  assert.equal(panelOffset(h), "60px -40px");
  assert.equal(storage.has(changesPositionKey), false, "live movement does not write to storage");
  const release = send("pointerup", 70, -50, scrim);
  assert.equal(release.target, changesHeader(h));
  assert.equal(changesHeader(h).hasPointerCapture(7), false);
  release.target.dispatch("click", { detail: 1 });
  assert.equal(h.$("changes").hidden, false);
  assert.deepEqual(JSON.parse(storage.get(changesPositionKey)), { x: 70, y: -50 });
  scrim.dispatch("click", { detail: 1 });
  assert.equal(h.$("changes").hidden, true, "an ordinary scrim click still dismisses the dialog");
});

test("Changes moving: transferring implicit pen capture to the header keeps the drag active", async () => {
  const storage = new Map(), h = await opened({ storage });
  const head = changesHeader(h), title = h.$("changes-title"), send = changesMovePointer(title);
  const pen = { pointerType: "pen" };
  title.setPointerCapture(7); // The browser implicitly captures a direct-manipulation pointer on press.
  send("pointerdown", 0, 0, title, pen);
  send("pointermove", 10, 5, title, pen);
  title.dispatch("lostpointercapture", { pointerId: 7, ...pen });
  assert.equal(head.hasPointerCapture(7), true);
  send("pointermove", 20, 10, h.$("changes-scrim"), pen);
  assert.equal(panelOffset(h), "20px 10px");
  head.releasePointerCapture(7);
  assert.deepEqual(JSON.parse(storage.get(changesPositionKey)), { x: 20, y: 10 });
  assert.equal(send("mousemove", 20, 10).defaultPrevented, undefined);
});

test("Changes moving: controls and selectable text keep their pointer and keyboard behavior", async () => {
  const linked = scope("linked", "linked worktree");
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) =>
    path === scopesPath() ? reply(catalog(mainScope, linked)) : undefined });
  await h.info(commitsEnabled);
  await settle();
  const refresh = h.$("changes-refresh"), close = h.$("changes-close"), before = h.calls.length;
  const controls = [refresh.children[0], close, h.$("changes-mode-committed"), h.$("changes-scope-name")];
  for (const control of controls) {
    const drag = startChangesMove(h);
    drag("pointermove", 15, 10, control);
    const release = drag("pointerup", 15, 10, control);
    assert.equal(release.target, changesHeader(h));
    release.target.dispatch("click", { detail: 1 });
  }
  assert.equal(h.calls.length, before);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-mode-working").getAttribute("aria-pressed"), "true");
  assert.equal(h.$("changes-scope").getAttribute("aria-expanded"), "false");
  const link = changesHeading(h).appendChild(h.document.createElement("a"));
  const focusable = changesHeading(h).appendChild(h.document.createElement("span"));
  focusable.tabIndex = -1;
  const focusableChild = focusable.appendChild(h.document.createElement("span"));
  const editable = changesHeading(h).appendChild(h.document.createElement("div"));
  editable.setAttribute("contenteditable", "true");
  editable.append("Editable header text");
  const bodyText = h.$("changes-directory-path"), gripTargets = [grip(h, "right"), grip(h, "bottom")];
  for (const target of [...controls, h.$("changes-root"), link, focusableChild, editable.children[0], bodyText]) {
    const send = changesMovePointer(target);
    assert.equal(send("pointerdown").defaultPrevented, undefined);
    assert.equal(send("pointermove", 15, 10).defaultPrevented, undefined);
    assert.equal(send("pointerup", 15, 10).defaultPrevented, undefined);
    assert.equal(changesHeader(h).hasPointerCapture(7), false);
    assert.equal(target.dispatch("dblclick", { detail: 2 }).defaultPrevented, undefined);
  }
  for (const target of gripTargets) assert.equal(target.dispatch("dblclick", { detail: 2 }).defaultPrevented, undefined);
  assert.equal(panelOffset(h), "60px 40px");
  refresh.focus();
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, grip(h));
  h.key("Tab");
  assert.equal(h.document.activeElement, close);
  assert.equal(h.key(" ").defaultPrevented, undefined);
  refresh.focus();
  assert.equal(h.key("Enter").defaultPrevented, undefined);
  refresh.dispatch("click", { detail: 0 });
  assert.equal(h.calls.at(-1).path, scopedListPath(mainScope));
});

test("Changes moving: Refresh, Back, mode tabs, scope picker, and Close activate without moving", async () => {
  const saved = JSON.stringify({ x: 36, y: 22 });
  const storage = new Map([[changesSizeKey, JSON.stringify({ width: 500, height: 640 })], [changesPositionKey, saved]]);
  const linked = scope("linked", "linked worktree"), item = file("one", "changed.txt");
  const list = scopedManifest(mainScope, [item]);
  const h = await opened({ storage, list, respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, linked));
    if (path === detailPath(list, item)) return reply(diff(list, item));
    if (path === commitsPath(mainScope)) return reply(commitHistory(mainScope, [headCommit]));
  } }, commitsEnabled);
  await settle();
  const unchanged = () => {
    assert.equal(panelOffset(h), "36px 22px");
    assert.equal(storage.get(changesPositionKey), saved);
  };
  const click = (target) => {
    assert.equal(target.closest("[hidden]"), null);
    const send = changesMovePointer(target);
    assert.equal(send("pointerdown").defaultPrevented, undefined);
    assert.equal(send("pointermove", 8, 6).defaultPrevented, undefined);
    const release = send("pointerup", 8, 6);
    assert.equal(release.target, target);
    assert.equal(release.defaultPrevented, undefined);
    unchanged();
    target.dispatch("click", { detail: 1 });
  };

  click(h.$("changes-refresh").children[0]);
  assert.equal(h.calls.at(-1).path, scopedListPath(mainScope));
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.$("changes-listpane").hidden, true);
  click(h.$("changes-back").children[0]);
  assert.equal(h.$("changes-listpane").hidden, false);
  h.flushHistory();

  const picker = h.$("changes-scope"), pickerName = h.$("changes-scope-name");
  click(pickerName);
  assert.equal(picker.getAttribute("aria-expanded"), "true");
  click(pickerName);
  assert.equal(picker.getAttribute("aria-expanded"), "false");
  picker.focus();
  assert.equal(h.key("ArrowDown").defaultPrevented, true);
  assert.equal(picker.getAttribute("aria-expanded"), "true");

  const committed = h.$("changes-mode-committed"), working = h.$("changes-mode-working");
  click(committed);
  assert.equal(h.calls.at(-1).path, commitsPath(mainScope));
  assert.equal(committed.getAttribute("aria-pressed"), "true");
  await settle();
  working.focus();
  assert.equal(h.key("Enter").defaultPrevented, undefined);
  working.dispatch("click", { detail: 0 }); // Native keyboard activation.
  assert.equal(working.getAttribute("aria-pressed"), "true");
  await settle();

  startChangesDrag(h, "right")("pointerup", 10, 0);
  assert.deepEqual(panelSize(h), ["520px", "640px"]);
  unchanged();
  click(h.$("changes-close").children[0]);
  assert.equal(h.$("changes").hidden, true);
  assert.equal(storage.get(changesPositionKey), saved);
});

test("Changes moving: saved position restores independently of size and double-click recenters", async () => {
  const size = JSON.stringify({ width: 820, height: 640 });
  const storage = new Map([[changesSizeKey, size]]), h = await opened({ storage });
  const drag = startChangesMove(h);
  drag("pointermove", -80, 55);
  drag("pointerup", -80, 55);
  h.click("changes-close");
  h.flushHistory();
  h.click("changesbtn");
  assert.equal(panelOffset(h), "-80px 55px");
  const reloaded = await opened({ storage });
  assert.equal(panelOffset(reloaded), "-80px 55px");
  assert.equal(changesHeader(reloaded).dispatch("dblclick", { detail: 2 }).defaultPrevented, true);
  assert.equal(panelOffset(reloaded), undefined);
  assert.deepEqual(panelSize(reloaded), ["820px", "640px"]);
  assert.equal(storage.get(changesSizeKey), size);
  assert.equal(storage.has(changesPositionKey), false);
});

test("Changes moving: restored and dragged positions clamp through viewport and panel resizing", async () => {
  const saved = JSON.stringify({ x: 2000, y: -2000 });
  const storage = new Map([[changesSizeKey, JSON.stringify({ width: 820, height: 600 })], [changesPositionKey, saved]]);
  const h = await opened({ storage, visualViewport: { width: 1440, height: 1000, offsetLeft: 140, offsetTop: 60 } });
  assert.equal(panelOffset(h), "310px -200px");
  assert.equal(storage.get(changesPositionKey), saved);
  const send = startChangesMove(h);
  send("pointermove", -40, 30);
  assert.equal(panelOffset(h), "270px -170px", "movement starts at the displayed position, not the stored preference");
  send("pointerup", 2000, -2000);
  assert.equal(panelOffset(h), "310px -200px");

  const { visualViewport } = h.window;
  Object.assign(visualViewport, { width: 1000, height: 700 });
  h.window.dispatch("resize");
  assert.equal(panelOffset(h), "90px -50px");
  startChangesDrag(h, "right")("pointerup", 40, 0);
  assert.deepEqual(panelSize(h), ["900px", "600px"]);
  assert.equal(panelOffset(h), "50px -50px");
  Object.assign(visualViewport, { width: 960, height: 640 });
  visualViewport.dispatch("resize");
  assert.equal(panelOffset(h), "30px -24px");
  assert.deepEqual(JSON.parse(storage.get(changesPositionKey)), { x: 310, y: -200 });
});

test("Changes moving: narrow or coarse layouts do not attach drag handlers or apply offsets", async () => {
  const storage = new Map([[changesPositionKey, JSON.stringify({ x: 40, y: 35 })]]);
  const desktop = await opened({ storage });
  assert.equal(panelOffset(desktop), "40px 35px");
  assert.equal(typeof changesHeader(desktop).onpointerdown, "function");
  desktop.resize(true);
  const unaffected = (h) => {
    assert.ok(!changesHeader(h).onpointerdown && !changesHeader(h).ondblclick);
    const send = changesMovePointer(h.$("changes-title"));
    assert.equal(send("pointerdown").defaultPrevented, undefined);
    assert.equal(send("pointermove", 40, 30).defaultPrevented, undefined);
    assert.equal(send("pointerup", 40, 30).defaultPrevented, undefined);
    assert.equal(panelOffset(h), undefined);
  };
  unaffected(desktop);
  desktop.resize(false);
  assert.equal(panelOffset(desktop), "40px 35px");
  assert.equal(typeof changesHeader(desktop).onpointerdown, "function");
  for (const options of [{ phone: true }, { phone: true, finePointer: true }, { finePointer: false }]) {
    unaffected(await opened({ ...options, storage }));
  }
  const title = desktop.$("changes-title"), touch = changesMovePointer(title);
  for (const [type, x] of [["pointerdown", 0], ["pointermove", 40], ["pointerup", 40]]) {
    assert.equal(touch(type, x, 0, title, { pointerType: "touch" }).defaultPrevented, undefined);
  }
  assert.equal(panelOffset(desktop), "40px 35px");
});

test("Changes moving: cancelling or closing a drag releases pointer capture", async () => {
  const storage = new Map(), h = await opened({ storage });
  const header = changesHeader(h), send = startChangesMove(h);
  send("pointermove", 40, 20);
  h.window.dispatch("blur");
  assert.equal(header.hasPointerCapture(7), false);
  send("pointerup", 90, 80);
  assert.equal(panelOffset(h), "40px 20px");
  const next = startChangesMove(h);
  next("pointermove", -10, -10);
  next("pointercancel", 90, 80);
  assert.equal(header.hasPointerCapture(7), false);
  assert.deepEqual(JSON.parse(storage.get(changesPositionKey)), { x: 30, y: 10 });
  const pending = startChangesMove(h), scrim = h.$("changes-scrim");
  assert.equal(pending("pointerup", 2, 0, scrim).defaultPrevented, undefined);
  assert.equal(panelOffset(h), "30px 10px");
  startChangesMove(h)("pointermove", 5);
  assert.equal(header.hasPointerCapture(7), true);
  assert.equal(h.key("Escape").defaultPrevented, true);
  assert.equal(h.$("changes").hidden, true);
  assert.equal(header.hasPointerCapture(7), false);
  assert.equal(h.document.activeElement, h.$("changesbtn"));
});

test("branch copy: worktree options and selected text copy exact names without picking a scope", async () => {
  const long = scope("linked-id", "feature/α-" + "long-".repeat(30) + "<end>");
  const other = scope("other-id", "another-worktree");
  const h = await opened({ respond: ({ path }) =>
    path === scopesPath() ? reply(catalog(mainScope, other, long)) : undefined });
  await settle();
  const before = h.calls.length, picker = h.$("changes-scope");
  assert.equal(picker.getAttribute("aria-haspopup"), "listbox", "options must be reachable by page events rather than a native select");
  assert.equal(h.$("changes-scope-name").textContent, mainScope.label + " (current)");
  await copyRef(h, h.$("changes-scope-name"), picker);
  h.click("changes-scope");
  assert.equal(picker.getAttribute("aria-expanded"), "true");
  const [current, middle, last] = h.scopeOptions();
  assert.deepEqual([current.textContent, middle.textContent, last.textContent], [mainScope.label + " (current)", other.label, long.label]);
  assert.equal(current.getAttribute("aria-selected"), "true");
  assert.equal(last.getAttribute("aria-selected"), "false");
  await copyRef(h, current);
  await copyRef(h, middle);
  await copyRef(h, last);
  assert.deepEqual(h.copied, [mainScope.label, mainScope.label, other.label, long.label]);
  assert.equal(picker.getAttribute("aria-expanded"), "true");
  assert.equal(picker.value, mainScope.id);
  assert.equal(h.calls.length, before, "copying must not fetch a new scope");
  current.dispatch("click");
  assert.equal(picker.getAttribute("aria-expanded"), "false");
  assert.equal(h.calls.length, before, "picking the already selected scope must not refetch it");
  const status = h.$("changes-copy-status");
  assert.equal(status.getAttribute("role"), "status");
  assert.equal(status.textContent, "Copied");
  assert.equal(h.timers.size, 1);
  const [{ fn, delay }] = h.timers.values();
  assert.ok(delay > 0 && delay <= 3000);
  fn();
  assert.equal(status.textContent, "");
});

test("branch copy: stacked menus dismiss independently and phone keyboard selection still refreshes the chosen scope", async () => {
  const linked = scope("linked-id", "feature/notes");
  const files = [file("one", "changed.txt")];
  const h = await opened({ phone: true, list: scopedManifest(mainScope, files), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, linked));
    if (path === scopedListPath(linked)) return reply(scopedManifest(linked, files));
  } });
  await settle();
  const picker = h.$("changes-scope"), options = h.$("changes-scope-options"), panel = h.$("changes-panel");
  assert.ok(options, "the picker must expose its options to right-click events");
  Object.assign(panel, { bounds: { left: 8, top: 10, bottom: 310 }, clientWidth: 320, clientHeight: 300 });
  h.$("changes-scope-control").bounds = { top: 230, bottom: 278 };
  options.offsetHeight = 132;
  picker.focus();
  assert.equal(h.key("ArrowDown").defaultPrevented, true);
  assert.match(options.className, /\babove\b/);
  const [current, alternate] = h.scopeOptions();
  assert.equal(h.document.activeElement, current);
  h.key("End");
  assert.equal(h.document.activeElement, alternate);
  const menu = openRefMenu(h, alternate, { clientX: 320, clientY: 304 });
  assert.equal(menu.style.left, "214px");
  assert.equal(menu.style.top, "250px");
  assert.equal(h.key("Escape").defaultPrevented, true);
  assert.equal(menu.hidden, true);
  assert.equal(options.hidden, false);
  assert.equal(h.document.activeElement, alternate);
  openRefMenu(h, alternate);
  options.dispatch("scroll");
  assert.equal(menu.hidden, true);
  assert.equal(options.hidden, false);
  openRefMenu(h, alternate);
  h.click("changes-coverage");
  assert.equal(menu.hidden, true);
  assert.equal(options.hidden, true);
  assert.equal(h.document.activeElement, picker);
  h.key("ArrowUp");
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(options.hidden, true);
  assert.equal(h.document.activeElement, h.rows()[0]);
  picker.focus();
  h.key("ArrowDown");
  h.key("ArrowDown");
  assert.equal(h.document.activeElement, alternate);
  assert.equal(h.key("Enter").defaultPrevented, undefined);
  alternate.dispatch("click"); // Native button activation from Enter.
  await settle();
  assert.equal(options.hidden, true);
  assert.equal(picker.value, linked.id);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.calls.at(-1).path, scopedListPath(linked));
  const before = h.calls.length;
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(before).map((call) => call.path), [scopedListPath(linked), scopesPath()]);
});

test("branch copy: Committed copies its named branch separately from the worktree badge", async () => {
  const branch = "topic/推进-é";
  let detached = false;
  const h = await opened({ list: scopedManifest(mainScope, [], { root_label: "working-tree label" }), respond: ({ path }) =>
    path === commitsPath(mainScope) ? reply(commitHistory(mainScope, [headCommit], { branch: detached ? null : branch })) : undefined,
  }, commitsEnabled);
  await settle();
  await copyRef(h, h.$("changes-root"));
  await openCommitList(h);
  const name = h.$("changes-branch");
  await copyRef(h, h.$("changes-root"));
  await copyRef(h, name);
  assert.equal(h.key("F10", { shiftKey: true }).defaultPrevented, true);
  h.click("changes-copy-ref");
  await settle();
  assert.deepEqual(h.copied, ["working-tree label", mainScope.label, branch, branch]);
  const [id, , meta] = h.commitRows()[0].children;
  assert.equal(id.dispatch("contextmenu").defaultPrevented, undefined);
  assert.equal(meta.children[1].dispatch("contextmenu").defaultPrevented, undefined, "HEAD is a badge, not a branch name");
  detached = true;
  h.click("changes-refresh");
  await settle();
  assert.equal(name.textContent, "detached HEAD");
  assert.equal(name.dispatch("contextmenu").defaultPrevented, undefined);
  assert.equal(h.$("changes-copy-ref").hidden, true);
});

test("branch copy: empty-state choices and scope notices copy each literal worktree label", async () => {
  const linked = scope("linked-id", "<feature> & notes", { files: 1 });
  const other = scope("other-id", "another-worktree", { files: 1 });
  let scopes = [mainScope, linked, other];
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(...scopes));
    if (path === scopedListPath(linked)) return scopes.includes(linked) ? reply(scopedManifest(linked, [file("one", "changed.txt")]))
      : reply({ error: "unknown changes scope", code: "unknown_scope" }, 400);
  } });
  await settle();
  const before = h.calls.length;
  for (const item of h.$("changes-scope-choices").children) await copyRef(h, item.children[0]);
  assert.deepEqual(h.copied, [linked.label, other.label]);
  assert.equal(h.calls.length, before);
  h.click("changes-close");
  h.flushHistory();
  scopes = [mainScope, linked];
  h.click("changesbtn");
  await settle();
  const labels = () => h.$("changes-scope-note").children.filter((node) => node.tagName === "span");
  assert.deepEqual(labels().map((node) => node.textContent), [mainScope.label, linked.label]);
  for (const node of labels()) await copyRef(h, node);
  scopes = [mainScope];
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(labels().map((node) => node.textContent), [linked.label]);
  await copyRef(h, labels()[0]);
  assert.deepEqual(h.copied, [linked.label, other.label, mainScope.label, linked.label, linked.label]);
});

test("branch copy: the selection fallback copies exact text and both failures report an error", async () => {
  const label = "worktree/é-<notes>", attempts = [];
  let supported = true;
  const h = await opened({ list: manifest([], { root_label: label }), clipboard: null, execCopy: (command, area) => {
    attempts.push([command, area.tagName, area.value, area.selectionStart, area.selectionEnd]);
    return supported;
  } });
  await settle();
  const root = h.$("changes-root"), status = h.$("changes-copy-status");
  await copyRef(h, root);
  assert.deepEqual(attempts, [["copy", "textarea", label, 0, label.length]]);
  assert.equal(status.textContent, "Copied");
  supported = false;
  const denied = [];
  h.context.navigator.clipboard = { async writeText(text) { denied.push(text); throw new Error("Permission denied"); } };
  await copyRef(h, root);
  assert.deepEqual(denied, [label]);
  assert.deepEqual(attempts.at(-1), attempts[0]);
  assert.equal(attempts.length, 2);
  assert.equal(h.document.body.children.some((node) => node.tagName === "textarea"), false);
  assert.equal(status.textContent, "Could not copy name to clipboard.");
  assert.match(status.className, /\berror\b/);
});

test("branch copy: a pending clipboard write cannot announce into a reopened viewer", async () => {
  const pending = deferred();
  const h = await opened({ clipboard: { writeText: () => pending.promise } });
  await settle();
  openRefMenu(h, h.$("changes-root")).dispatch("click");
  h.click("changes-close");
  h.flushHistory();
  h.click("changesbtn");
  await settle();
  pending.resolve();
  await settle();
  assert.equal(h.$("changes-copy-status").textContent, "");
  assert.equal(h.timers.size, 0);
});

test("the selected-session button requires the exact /info capability and stops drawer clicks", async () => {
  const h = harness({ sid: "" });
  assert.equal(h.$("changesbtn").hidden, true);
  await h.info();
  assert.equal(h.$("changesbtn").hidden, true);
  h.context.switchSession(defaultId, false);
  assert.equal(h.$("changesbtn").hidden, false);
  for (const capabilities of [{ changes: true }, { features: {} }, { features: { changes: false } }, { features: { changes: "true" } }]) {
    await h.info(capabilities);
    assert.equal(h.$("changesbtn").hidden, true);
  }
  await h.info();
  assert.equal(h.$("changesbtn").hidden, false);
  assert.equal(h.click("changesbtn").stopped, true);
  assert.equal(h.drawerOpens(), 0);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changesbtn").getAttribute("aria-expanded"), "true");
  assert.equal(h.document.activeElement, h.$("changes-close"));
  assert.equal(h.$("changes-list-status").textContent, "Loading changes…");
  h.context.switchSession("", false);
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.$("changesbtn").hidden, true);
  h.flushHistory();
});

for (const [modifier, key, flags] of [
  ["Ctrl", "i", { ctrlKey: true }],
  ["Cmd", "I", { metaKey: true }],
]) {
  test(`Changes shortcut: ${modifier}+I opens the dialog in capture`, async () => {
    const h = harness();
    await h.info();
    const target = h.$("questionbtn");
    target.focus();
    let reachedTarget = false;
    target.onkeydown = () => { reachedTarget = true; };
    const event = h.key(key, flags);
    assert.equal(event.defaultPrevented, true);
    assert.equal(reachedTarget, false);
    assert.equal(h.$("changes").hidden, false);
    assert.equal(h.$("changesbtn").getAttribute("aria-expanded"), "true");
    assert.equal(h.document.activeElement, h.$("changes-close"));
    assert.equal(h.calls.at(-1).path, listPath());
  });
}

test("Changes shortcut: hidden button, Shift, Alt, composition, and repeat leave the key alone", async () => {
  const h = harness();
  h.$("questionbtn").focus();
  const ignored = (key, extra) => {
    const event = h.key(key, { ctrlKey: true, ...extra });
    assert.equal(event.defaultPrevented, undefined);
    assert.equal(event.stopped, undefined);
    assert.equal(h.$("changes").hidden, true);
  };
  assert.equal(h.$("changesbtn").hidden, true);
  ignored("i");
  await h.info();
  ignored("I", { shiftKey: true });
  for (const extra of [{ altKey: true }, { isComposing: true }, { repeat: true }]) ignored("i", extra);
  assert.deepEqual(h.calls.map((call) => call.path), ["/info"]);
});

test("Changes shortcut: the composer and other editable controls keep Ctrl+I", async () => {
  const h = harness();
  await h.info();
  const add = (tag) => h.document.body.appendChild(h.document.createElement(tag));
  const editor = add("div");
  editor.setAttribute("contenteditable", "true");
  const child = editor.appendChild(h.document.createElement("span"));
  child.tabIndex = 0;
  for (const target of [h.$("prompt"), add("input"), add("select"), child]) {
    target.focus();
    const event = h.key("i", { ctrlKey: true });
    assert.equal(event.defaultPrevented, undefined);
    assert.equal(event.stopped, undefined);
    assert.equal(h.$("changes").hidden, true);
  }
  assert.deepEqual(h.calls.map((call) => call.path), ["/info"]);
});

test("Changes shortcut: pressing it again leaves the selected diff and history intact", async () => {
  const item = file("one", "first.txt"), list = manifest([item]), raw = "@@ -1 +1 @@\n-old\n+new\n";
  const h = harness({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? reply(diff(list, item, { diff: raw })) : undefined });
  await h.info();
  h.$("questionbtn").focus();
  assert.equal(h.key("i", { ctrlKey: true }).defaultPrevented, true);
  await settle();
  h.rows()[0].dispatch("click");
  await settle();
  const requests = h.calls.length, state = h.history.state;
  const event = h.key("i", { ctrlKey: true });
  assert.equal(event.defaultPrevented, undefined);
  assert.equal(event.stopped, undefined);
  await settle();
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.$("changes-diff").textContent, raw);
  assert.equal(h.document.activeElement, h.$("changes-back"));
  assert.equal(h.history.state, state);
  assert.equal(h.calls.length, requests);
});

test("the file list and unified diff use literal text and the authenticated encoded routes", async () => {
  const hostile = "<img src=x onerror=alert(1)> & 🦊.js";
  const renamed = file("opaque/a?%+", hostile, { old_path: "old/α.js", status: "renamed" });
  const list = manifest([renamed, file("m", "m"), file("a", "a", { status: "added" }), file("d", "d", { status: "deleted" })], { root_label: "<sample> & repo" });
  const raw = "diff --git a/old b/new\n--- a/old\n+++ b/new\n@@ -1,2 +1,2 @@\n-old\n+++ literal code\n+" + hostile + "\n unchanged\n\\ No newline at end of file\n";
  const h = await opened({ list, respond: ({ path }) => path === detailPath(list, renamed) ? reply(diff(list, renamed, { diff: raw })) : undefined });
  await settle();
  assert.equal(h.$("changes-root").textContent, "<sample> & repo");
  assert.match(h.$("changes").textContent, /Working tree · uncommitted/);
  assert.equal(h.$("changes-count").textContent, "4");
  assert.deepEqual(h.rows().map((row) => row.children[0].textContent), ["R", "M", "A", "D"]);
  const row = h.rows()[0];
  assert.equal(row.children[1].textContent, "old/α.js → " + hostile);
  assert.deepEqual(row.children[1].children.map((line) => line.textContent), ["old/α.js", " → " + hostile]);
  assert.equal(row.title, row.children[1].textContent);
  assert.deepEqual(row.children[2].children.map((count) => count.textContent), ["+3", "−1"]);
  row.dispatch("click");
  assert.equal(row.getAttribute("aria-current"), "true");
  assert.equal(h.$("changes-detail-status").textContent, "Loading diff…");
  await settle();
  assert.equal(h.$("changes-diff").textContent, raw);
  assert.deepEqual(h.$("changes-diff").children.filter((line) => line.tagName === "span").map((line) => line.className),
    ["changes-line meta", "changes-line meta", "changes-line meta", "changes-line hunk", "changes-line del", "changes-line add", "changes-line add", "changes-line", "changes-line meta"]);
  assert.deepEqual(h.calls.map((call) => [call.method, call.url, call.headers.Authorization]),
    ["/info", listPath(), scopesPath(), detailPath(list, renamed)].map((route) => ["GET", base + route, "Bearer " + token]));
  assert.deepEqual(h.calls.slice(1).map((call) => call.cache), ["no-store", "no-store", "no-store"]);
});

test("file row line counts use the API values and leave the path text unchanged", async () => {
  const files = [
    file("changed", "src/bin/ai/serve/<draft> & notes.md", { additions: 14, deletions: 2 }),
    file("empty", "empty.txt", { status: "added", additions: 0, deletions: 0 }),
  ];
  const h = await opened({ phone: true, list: manifest(files) });
  await settle();
  const rows = h.rows();
  assert.deepEqual(rows.map((row) => row.children[1].textContent), files.map((item) => item.path));
  assert.deepEqual(rows.map((row) => row.children[2].children.map((count) => count.textContent)), [["+14", "−2"], ["+0", "−0"]]);
  assert.equal(rows[0].getAttribute("aria-label"), "Modified: " + files[0].path + ". Lines added: 14, lines removed: 2.");
});

test("file row line counts omit missing or partial stats and leave only the binary label for media", async () => {
  const files = [
    file("unknown", "unknown.txt", { additions: null, deletions: null }),
    { id: "omitted", path: "omitted.txt", status: "modified" },
    file("partial", "partial.txt", { deletions: null }),
    file("image", "banner.png", { binary: true, additions: 0, deletions: 0 }),
  ];
  const h = await opened({ list: manifest(files) });
  await settle();
  assert.deepEqual(h.rows().map((row) => row.children.slice(2).map((node) => node.textContent)), [[], [], [], ["Binary"]]);
});

test("rapid picks and refresh abort requests and ignore stale response bodies", async () => {
  const one = file("one", "first.txt"), two = file("two", "second.txt"), list = manifest([one, two]);
  const freshTwo = file("new-id", two.path), freshList = manifest([freshTwo], { revision: "next" });
  const slowBody = deferred(), refreshFlight = deferred();
  let fresh = false, picksOfTwo = 0;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === listPath() && fresh) return reply(freshList);
    if (path === detailPath(list, one)) return { ...reply(null), json: () => slowBody.promise };
    if (path === detailPath(list, two)) return ++picksOfTwo === 1
      ? reply(diff(list, two, { diff: "@@ -1 +1 @@\n+second\n" })) : refreshFlight.promise;
    if (path === detailPath(freshList, freshTwo)) return reply(diff(freshList, freshTwo, { diff: "@@ -1 +1 @@\n+fresh\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  const oneSignal = h.calls.at(-1).signal;
  await settle();
  h.rows()[1].dispatch("click");
  assert.equal(oneSignal.aborted, true);
  await settle();
  slowBody.resolve(diff(list, one, { diff: "@@ -1 +1 @@\n+stale first\n" }));
  await settle();
  assert.match(h.$("changes-diff").textContent, /\+second/);
  assert.equal(h.rows()[0].getAttribute("aria-current"), "false");
  h.rows()[1].dispatch("click");
  const twoSignal = h.calls.at(-1).signal;
  fresh = true;
  h.click("changes-refresh");
  assert.equal(twoSignal.aborted, true);
  await settle();
  refreshFlight.resolve(reply(diff(list, two, { diff: "@@ -1 +1 @@\n+stale second\n" })));
  await settle();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [listPath(), detailPath(freshList, freshTwo), scopesPath()]);
  assert.equal(h.rows()[0].getAttribute("aria-current"), "true");
  assert.equal(h.$("changes-diff").textContent, "@@ -1 +1 @@\n+fresh\n");
});

test("refresh does not match a lossy display path to a different file ID", async () => {
  const old = file("before", "name\uFFFD.txt"), replacement = file("after", old.path);
  const list = manifest([old]), freshList = manifest([replacement], { revision: "new" });
  let fresh = false;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === listPath() && fresh) return reply(freshList);
    if (path === detailPath(list, old)) return reply(diff(list, old));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  fresh = true;
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [listPath(), scopesPath()]);
  assert.equal(h.rows()[0].getAttribute("aria-current"), "false");
  assert.match(h.$("changes-detail-status").textContent, /Select a file/);
});

test("unavailable list errors stay inline and Retry can reveal the empty state", async () => {
  let unavailable = true;
  const h = await opened({ respond: ({ path }) => path === listPath() && unavailable ? reply({ error: "No repository" }, 404) : undefined });
  await settle();
  assert.match(h.$("changes-list-status").textContent, /Changes are unavailable for this session.*No repository/);
  assert.equal(h.$("changes-list-retry").hidden, false);
  h.$("changes-list-retry").focus();
  unavailable = false;
  h.click("changes-list-retry");
  await settle();
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.document.activeElement, h.$("changes-refresh"));
});

test("409 detail retry fetches a fresh revision rather than resending a stale file request", async () => {
  const item = file("before", "changed.txt"), freshItem = file("after", item.path);
  const list = manifest([item]), freshList = manifest([freshItem], { revision: "updated" });
  let changed = false;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === listPath() && changed) return reply(freshList);
    if (path === detailPath(list, item)) return reply({ error: "Working tree changed" }, 409);
    if (path === detailPath(freshList, freshItem)) return reply(diff(freshList, freshItem));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  await settle();
  assert.match(h.$("changes-detail-status").textContent, /Changes are unavailable for this session.*Working tree changed/);
  assert.equal(h.$("changes-detail-retry").hidden, false);
  changed = true;
  h.click("changes-detail-retry");
  await settle();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [listPath(), detailPath(freshList, freshItem), scopesPath()]);
  assert.equal(h.$("changes-detail-status").textContent, "No textual differences.");
  assert.equal(h.$("changes-detail-retry").hidden, true);
});

test("binary, symlink, and truncated responses show the corresponding plain-text notes", async () => {
  const media = file("media", "banner.png", { binary: true, additions: null, deletions: null });
  const link = file("link", "latest", { status: "added", additions: null, deletions: null });
  const text = file("text", "notes.md");
  const list = manifest([media, link, text], { truncated: true, reason: "Only the first matches are shown." });
  const h = await opened({ list, respond: ({ path }) => {
    if (path === detailPath(list, media)) return reply(diff(list, media, { format: "none" }));
    if (path === detailPath(list, link)) return reply(diff(list, link, { format: "none", reason: "Symbolic link." }));
    if (path === detailPath(list, text)) return reply(diff(list, text, { diff: "@@ -1 +1 @@\n+preview", truncated: true, reason: "Size limit reached." }));
  } });
  await settle();
  assert.equal(h.$("changes-count").textContent, "3+");
  assert.match(h.$("changes-list-warning").textContent, /File list truncated.*Only the first/);
  assert.equal(h.rows()[0].children[2].textContent, "Binary");
  h.rows()[0].dispatch("click");
  await settle();
  assert.match(h.$("changes-detail-status").textContent, /Binary file. No text diff/);
  assert.equal(h.$("changes-diff").hidden, true);
  h.rows()[1].dispatch("click");
  await settle();
  assert.match(h.$("changes-detail-status").textContent, /No text diff.*Symbolic link/);
  h.rows()[2].dispatch("click");
  await settle();
  assert.equal(h.$("changes-detail-warning").textContent, "Diff truncated. Size limit reached.");
  assert.match(h.$("changes-diff").textContent, /\+preview/);
});

test("the complete Changes directory is visible in both mobile panes without a scope selector", async () => {
  const current = { ...mainScope, path: "/checkout/" + "x".repeat(180) + "/<worktree> & docs" };
  const item = file("one", "first.txt"), list = scopedManifest(current, [item]);
  const h = await opened({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? reply(diff(list, item)) : undefined });
  await settle();
  const row = h.$("changes-directory"), directory = h.$("changes-directory-path");
  assert.equal(row.parent, h.$("changes-panel"), "directory is outside the header and the switchable panes");
  assert.match(row.textContent, /^Directory:/);
  assert.equal(directory.textContent, current.path);
  assert.equal(directory.closest("[hidden]"), null);
  assert.equal(h.$("changes-scope-control").hidden, true);
  assert.match(between('<style id="changes-style">', '</style>', html), /#changes-directory\s*\{[^}]*overflow-wrap:anywhere/);
  h.rows()[0].dispatch("click");
  assert.equal(h.$("changes-listpane").hidden, true);
  assert.equal(directory.closest("[hidden]"), null);
  assert.equal(directory.textContent, current.path);
  await settle();
});

for (const phone of [true, false]) test("a 3,740-character " + (phone ? "mobile" : "desktop") + " directory scrolls independently of the Changes body", async () => {
  const path = "/" + "x".repeat(3735) + "/end";
  const h = await opened({ phone, list: scopedManifest({ ...mainScope, path }, [file("one", "first.txt")]) });
  await settle();
  const directory = h.$("changes-directory"), body = h.$("changes-body");
  assert.equal(h.$("changes-directory-path").textContent, path);
  assert.equal(directory.parent, body.parent, "the path and body must be independent flex items");
  assert.equal(body.closest("[hidden]"), null);
  const css = between('<style id="changes-style">', '</style>', html);
  const base = css.slice(0, css.indexOf("@media (max-width:700px)"));
  const pathRules = base.match(/#changes-directory\s*\{([^}]+)\}/)?.[1] ?? "";
  assert.match(pathRules, /max-height:\s*4\.5em/);
  assert.match(pathRules, /overflow-y:\s*auto/);
  assert.match(pathRules, /overflow-wrap:\s*anywhere/);
  assert.match(base, /#changes-body\s*\{[^}]*flex:1;[^}]*min-height:0/);
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, directory);
});

test("the directory follows scope selection, the returned path and a reopened viewer", async () => {
  const linked = scope("linked", "linked worktree"), pending = deferred();
  const currentFile = file("current", "changed.txt"), currentList = scopedManifest(mainScope, [currentFile]);
  const listed = { ...linked, path: linked.path + "/project" };
  const h = await opened({ list: currentList, respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, linked));
    if (path === scopedListPath(linked)) return pending.promise;
  } });
  await settle();
  const directory = h.$("changes-directory-path");
  assert.equal(directory.textContent, mainScope.path);
  h.setScope(linked);
  assert.equal(directory.textContent, linked.path, "the directory updates while the new listing loads");
  pending.resolve(reply(scopedManifest(listed, [currentFile])));
  await settle();
  assert.equal(directory.textContent, listed.path);
  h.click("changes-close");
  h.flushHistory();
  h.click("changesbtn");
  assert.equal(directory.textContent, "—", "the new viewer does not show a stale path");
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [scopesPath(), scopedListPath(linked)]);
  assert.equal(directory.textContent, listed.path);
});

const changesScopeKey = (sid = defaultId) => "serve.changesScope." + encodeURIComponent(sid);

test("Changes scope restores each session's pick before loading files and keeps it across tabs and Refresh", async () => {
  const storage = new Map(), other = "other/β?+%";
  const first = scope("linked/α?= +%", "feature"), second = scope("linked-other", "other feature");
  const available = catalog(mainScope, first, second), list = scopedManifest(mainScope, [file("current", "current.txt")]);
  const respond = ({ path }) => {
    if (path === scopesPath() || path === scopesPath(other)) return reply(available);
    if (path === scopedListPath(first)) return reply(scopedManifest(first, [file("first", "first.txt")]));
    if (path === scopedListPath(second, other)) return reply(scopedManifest(second, [file("second", "second.txt")]));
    if (path === commitsPath(first)) return reply(commitHistory(first, [headCommit], { branch: "feature" }));
  };
  const open = (options = {}) => opened({ storage, list, respond, ...options }, commitsEnabled);
  const h = await open();
  await settle();
  h.setScope(first);
  await settle();
  const otherView = await open({ sid: other });
  await settle();
  assert.equal(otherView.$("changes-scope").value, mainScope.id);
  otherView.setScope(second);
  await settle();

  const pending = deferred();
  let scanned = false;
  const restored = await open({ respond: (request) => {
    if (request.path === scopesPath() && !scanned) { scanned = true; return pending.promise; }
    return respond(request);
  } });
  assert.deepEqual(restored.calls.map((call) => call.path), ["/info", scopesPath()]);
  const initial = restored.calls.at(-1);
  restored.click("changes-refresh");
  assert.equal(initial.signal.aborted, true);
  await settle();
  pending.resolve(reply(catalog(mainScope, second)));
  await settle();
  assert.deepEqual(restored.calls.map((call) => call.path), ["/info", scopesPath(), scopesPath(), scopedListPath(first)]);
  assert.equal(restored.$("changes-scope").value, first.id);
  assert.deepEqual(restored.scopeOptions().map((option) => option.textContent), ["session repo (current)", first.label, second.label]);
  assert.equal(restored.rows()[0].title, "first.txt");
  restored.click("changes-refresh");
  await settle();
  assert.deepEqual(restored.calls.slice(-2).map((call) => call.path), [scopedListPath(first), scopesPath()]);
  await openCommitList(restored);
  assert.equal(restored.calls.at(-1).path, commitsPath(first));
  restored.click("changes-refresh");
  await settle();
  assert.deepEqual(restored.calls.slice(-2).map((call) => call.path), [commitsPath(first), scopesPath()]);
  restored.click("changes-mode-working");
  await settle();
  assert.equal(restored.rows()[0].title, "first.txt");

  const otherRestored = await open({ sid: other });
  await settle();
  assert.deepEqual(otherRestored.calls.map((call) => call.path), ["/info", scopesPath(other), scopedListPath(second, other)]);
  assert.equal(otherRestored.rows()[0].title, "second.txt");
  assert.equal(storage.get(changesScopeKey()), JSON.stringify(first.id));
  assert.equal(storage.get(changesScopeKey(other)), JSON.stringify(second.id));
});

test("Changes scope falls back silently and forgets an absent saved pick only after a complete scan", async () => {
  const saved = scope("remembered", "remembered worktree"), dirty = scope("other-dirty", "other dirty worktree", { files: 1 });
  const incomplete = { ...catalog(mainScope, dirty), truncated: true, reason: "Worktree scan timed out." };
  for (const [scan, keep] of [[reply(catalog(mainScope, dirty)), false], [reply(incomplete), true], [reply({}, 503), true]]) {
    const storage = new Map([[changesScopeKey(), JSON.stringify(saved.id)]]);
    let discovery = scan;
    const options = { storage, list: scopedManifest(mainScope), respond: ({ path }) => {
      if (path === scopesPath()) return discovery;
      if (path === scopedListPath(saved)) return reply(scopedManifest(saved, [file("remembered", "remembered.txt")]));
    } };
    const h = await opened(options);
    await settle();
    assert.deepEqual(h.calls.map((call) => call.path), ["/info", scopesPath(), listPath()]);
    assert.equal(h.$("changes-scope").value, mainScope.id);
    assert.equal(h.$("changes-scope-notice").hidden, true);
    assert.equal(h.$("changes-list-retry").hidden, true);
    assert.equal(storage.has(changesScopeKey()), keep);
    if (keep) {
      discovery = reply(catalog(mainScope, saved));
      const restored = await opened(options);
      await settle();
      assert.equal(restored.rows()[0].title, "remembered.txt");
    } else assert.equal(h.$("changes-scopes-status").hidden, true);
  }
});

test("Changes scope forgets a removed selection on Refresh and preserves the notice", async () => {
  const worktree = scope("linked", "feature");
  for (const unknownScope of [false, true]) {
    const storage = new Map(), list = scopedManifest(mainScope, [file("current", "current.txt")]);
    let removed = false;
    const options = { list, storage, respond: ({ path }) => {
      if (path === scopesPath()) return reply(removed ? catalog(mainScope) : catalog(mainScope, worktree));
      if (path === scopedListPath(worktree)) return removed && unknownScope
        ? reply({ code: "unknown_scope", error: "unknown changes scope" }, 400)
        : reply(scopedManifest(worktree, [file("linked", "linked.txt")]));
    } };
    const h = await opened(options);
    await settle();
    h.setScope(worktree);
    await settle();
    assert.equal(storage.get(changesScopeKey()), JSON.stringify(worktree.id));
    removed = true;
    h.click("changes-refresh");
    await settle();
    assert.equal(h.$("changes-scope").value, mainScope.id);
    assert.match(h.$("changes-scope-note").textContent, /feature is no longer available.*session directory/);
    assert.equal(h.rows()[0].title, "current.txt");
    assert.equal(storage.has(changesScopeKey()), false);
    const reopened = await opened(options);
    await settle();
    assert.deepEqual(reopened.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
    assert.equal(reopened.$("changes-scope-notice").hidden, true);
  }
});

test("Changes scope ignores corrupt or empty preferences and replaces them on a pick", async () => {
  const worktree = scope("linked", "feature"), list = scopedManifest(mainScope, [file("current", "current.txt")]);
  for (const value of ["", "broken", '""', '"   "', "null", "17", JSON.stringify({ id: worktree.id }), JSON.stringify([worktree.id])]) {
    const storage = new Map([[changesScopeKey(), value]]);
    const h = await opened({ list, storage, respond: ({ path }) => {
      if (path === scopesPath()) return reply(catalog(mainScope, worktree));
      if (path === scopedListPath(worktree)) return reply(scopedManifest(worktree));
    } });
    await settle();
    assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
    assert.equal(h.$("changes-scope").value, mainScope.id);
    assert.equal(h.$("changes-scope-notice").hidden, true);
    h.setScope(worktree);
    await settle();
    assert.equal(storage.get(changesScopeKey()), JSON.stringify(worktree.id));
  }
});

test("Changes scope remembers explicit current and never saves an automatic pick", async () => {
  const storage = new Map(), worktree = scope("linked", "feature", { files: 1 });
  const options = { storage, list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, worktree));
    if (path === scopedListPath(worktree)) return reply(scopedManifest(worktree, [file("linked", "linked.txt")]));
  } };
  const h = await opened(options);
  await settle();
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.equal(storage.has(changesScopeKey()), false);
  const before = h.calls.length;
  h.setScope(worktree);
  assert.equal(h.calls.length, before, "picking an already active scope does not reload it");
  assert.equal(storage.get(changesScopeKey()), JSON.stringify(worktree.id));
  h.click("changes-show-current");
  await settle();
  assert.equal(storage.get(changesScopeKey()), JSON.stringify(mainScope.id));
  const reopened = await opened(options);
  await settle();
  assert.deepEqual(reopened.calls.map((call) => call.path), ["/info", scopesPath(), listPath()]);
  assert.equal(reopened.$("changes-scope").value, mainScope.id);
  assert.equal(reopened.$("changes-scope-notice").hidden, true);
});

test("mobile browser Back aborts the file request, restores the list, then closes", async () => {
  const item = file("one", "first.txt"), list = manifest([item]), pending = deferred();
  const h = await opened({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? pending.promise : undefined });
  await settle();
  assert.equal(h.$("changes-detail").hidden, true);
  h.rows()[0].dispatch("click");
  const detail = h.calls.at(-1);
  assert.equal(h.$("changes-listpane").hidden, true);
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.document.activeElement, h.$("changes-back"));
  h.back();
  assert.equal(detail.signal.aborted, true);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.document.activeElement, h.rows()[0]);
  pending.resolve(reply(diff(list, item, { diff: "@@ -1 +1 @@\n+stale\n" })));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "");
  h.back();
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.document.activeElement, h.$("changesbtn"));
});

test("mobile in-app Back followed by Close consumes only the viewer's history entries", async () => {
  const item = file("one", "first.txt"), list = manifest([item]);
  const h = await opened({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? reply(diff(list, item)) : undefined });
  await settle();
  h.rows()[0].dispatch("click");
  h.click("changes-back");
  assert.equal(h.$("changes-listpane").hidden, false);
  h.click("changes-close");
  assert.equal(h.$("changes").hidden, true);
  h.flushHistory();
  assert.equal(h.history.state.base, true);
  assert.equal(h.history.state.changes, undefined);
});

test("mobile refresh returns to the list when the selected file disappears", async () => {
  const item = file("one", "first.txt"), list = manifest([item]);
  let refreshed = false;
  const h = await opened({ phone: true, list, respond: ({ path }) => {
    if (path === detailPath(list, item)) return reply(diff(list, item));
    if (path === listPath() && refreshed) return reply(manifest([], { revision: "fresh" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.history.state.changes.depth, 2);
  refreshed = true;
  h.click("changes-refresh");
  await settle();
  assert.equal(h.$("changes-listpane").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  h.flushHistory();
  assert.equal(h.history.state.changes.depth, 1);
  h.back();
  assert.equal(h.$("changes").hidden, true);
});

test("Escape from mobile detail and close from loading desktop abort and restore focus", async () => {
  const item = file("one", "first.txt"), list = manifest([item]), pending = deferred();
  const h = await opened({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? pending.promise : undefined });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.key("Escape").defaultPrevented, true);
  assert.equal(h.calls.at(-1).signal.aborted, true);
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.document.activeElement, h.$("changesbtn"));
  h.flushHistory();
  assert.equal(h.history.state.changes, undefined);
  const laterList = deferred();
  const desktop = await opened({ respond: ({ path }) => path === listPath() ? laterList.promise : undefined });
  assert.equal(desktop.key("Tab").defaultPrevented, true);
  assert.equal(desktop.document.activeElement, desktop.$("changes-directory"));
  assert.equal(desktop.key("Tab").defaultPrevented, true);
  assert.equal(desktop.document.activeElement, desktop.$("changes-detail-scroll"));
  desktop.click("changes-close");
  assert.equal(desktop.calls.at(-1).signal.aborted, true);
  laterList.resolve(reply(list));
  await settle();
  assert.equal(desktop.rows().length, 0);
  assert.equal(desktop.document.activeElement, desktop.$("changesbtn"));
});

test("session and connection switches abort and prevent stale data or capability", async () => {
  const pending = deferred();
  const h = await opened({ respond: ({ path }) => path === listPath() ? pending.promise : undefined });
  const previousRequest = h.calls.at(-1);
  h.context.switchSession("other", false);
  assert.equal(previousRequest.signal.aborted, true);
  assert.equal(h.$("changes").hidden, true);
  h.flushHistory();
  h.click("changesbtn");
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [listPath("other"), scopesPath("other")]);
  pending.resolve(reply(manifest([file("stale", "stale.txt")])));
  await settle();
  assert.equal(h.rows().length, 0);
  h.context.store.base = "https://older.example.invalid";
  h.window.dispatch("storage", { key: "serve.base" });
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.$("changesbtn").hidden, true);
  await h.info({});
  assert.equal(h.$("changesbtn").hidden, true);
});

test("older /info responses cannot re-enable a capability removed by the latest response", async () => {
  const pending = deferred();
  let calls = 0;
  const h = harness({ respond: ({ path }) => path === "/info" && calls++ === 0 ? pending.promise : undefined });
  const oldInfo = h.info();
  await h.info({ features: { changes: false } });
  pending.resolve(reply(enabled));
  await oldInfo;
  assert.equal(h.$("changesbtn").hidden, true);
});

test("an empty current scope automatically opens the only dirty worktree and can show current again", async () => {
  const worktree = scope("worktree/α?= +%", "<feature> & worktree", { files: 1 });
  const item = file("one", "first.txt"), list = scopedManifest(mainScope);
  const h = await opened({ list, respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, worktree));
    if (path === scopedListPath(worktree)) return reply(scopedManifest(worktree, [item]));
  } });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath(), scopedListPath(worktree)]);
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.equal(h.rows().length, 1);
  assert.equal(h.$("changes-scope-notice").hidden, false);
  assert.equal(h.$("changes-scope-note").textContent, "No changes in session repo when opened; showing worktree <feature> & worktree.");
  const showCurrent = h.$("changes-show-current");
  assert.equal(showCurrent.hidden, false);
  showCurrent.focus();
  h.click("changes-show-current");
  await settle();
  assert.equal(h.calls.at(-1).path, scopedListPath(mainScope));
  assert.equal(h.$("changes-scope").value, mainScope.id);
  assert.equal(h.$("changes-scope-notice").hidden, true);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.document.activeElement, h.$("changes-scope"));
  const beforeRefresh = h.calls.length;
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(beforeRefresh).map((call) => call.path), [scopedListPath(mainScope), scopesPath()]);
  assert.equal(h.$("changes-scope").value, mainScope.id);
});

test("multiple dirty worktrees are offered as keyboard-reachable empty-state choices", async () => {
  const first = scope("first", "first worktree", { files: 1 }), second = scope("second", "second worktree", { files: 2 });
  const clean = scope("clean", "clean worktree");
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, first, clean, second));
    if (path === scopedListPath(second)) return reply(scopedManifest(second, [file("chosen", "picked.txt")]));
  } });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
  const choices = h.$("changes-scope-choices");
  assert.equal(choices.hidden, false);
  assert.match(h.$("changes-list-status").textContent, /Choose a worktree with changes/);
  const buttons = choices.children.map((row) => row.children[0]);
  assert.deepEqual(buttons.map((button) => button.textContent), [first.label, second.label]);
  h.key("Tab");
  h.key("Tab");
  h.key("Tab");
  assert.equal(h.document.activeElement, buttons[0]);
  h.key("Tab");
  assert.equal(h.document.activeElement, buttons[1]);
  h.document.activeElement.dispatch("click");
  await settle();
  assert.equal(h.calls.at(-1).path, scopedListPath(second));
  assert.equal(h.$("changes-scope").value, second.id);
  assert.equal(h.rows()[0].title, "picked.txt");
  assert.equal(choices.hidden, true);
  assert.equal(h.document.activeElement, h.$("changes-scope"));
});

test("scope changes abort list and diff requests and cannot retain a shared file ID across scopes", async () => {
  const worktree = scope("feature/α?= +%", "feature", { files: 1 });
  const item = file("same/id", "shared.txt");
  const list = scopedManifest(mainScope, [item]), worktreeList = scopedManifest(worktree, [item]);
  const staleDiff = deferred(), staleList = deferred();
  let currentDiffs = 0;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog({ ...mainScope, changed: true, files: 1 }, worktree));
    if (path === scopedListPath(worktree)) return reply(worktreeList);
    if (path === scopedListPath(mainScope)) return staleList.promise;
    if (path === detailPath(list, item)) return ++currentDiffs === 1
      ? reply(diff(list, item, { diff: "+current\n" })) : staleDiff.promise;
    if (path === detailPath(worktreeList, item)) return reply(diff(worktreeList, item, { diff: "+worktree\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.calls.at(-1).path, detailPath(list, item));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "+current\n");
  h.rows()[0].dispatch("click");
  const previousDiff = h.calls.at(-1);
  h.setScope(worktree);
  assert.equal(previousDiff.signal.aborted, true);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.equal(h.$("changes-path").textContent, "File diff");
  assert.equal(h.calls.at(-1).path, scopedListPath(worktree));
  await settle();
  assert.equal(h.rows()[0].getAttribute("aria-current"), "false");
  h.rows()[0].dispatch("click");
  assert.equal(h.calls.at(-1).path, detailPath(worktreeList, item));
  staleDiff.resolve(reply(diff(list, item, { diff: "+stale\n" })));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "+worktree\n");
  h.setScope(mainScope);
  const previousList = h.calls.at(-1);
  assert.equal(previousList.path, scopedListPath(mainScope));
  h.setScope(worktree);
  assert.equal(previousList.signal.aborted, true);
  staleList.resolve(reply(scopedManifest(mainScope, [file("stale", "stale.txt")])));
  await settle();
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.deepEqual(h.rows().map((row) => row.title), [item.path]);
});

test("refresh preserves the worktree and mobile detail until the scope vanishes, then returns to current", async () => {
  const worktree = scope("alternate", "feature", { files: 1 });
  const item = file("first-id", "worktree.txt"), nextItem = file("next-id", item.path);
  const worktreeList = scopedManifest(worktree, [item]);
  const refreshed = scopedManifest(worktree, [nextItem], { revision: "next" });
  const currentItem = file("current", "current.txt");
  let vanish = false, worktreeLoads = 0;
  const h = await opened({ phone: true, list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === listPath() && vanish) return reply(scopedManifest(mainScope, [currentItem]));
    if (path === scopesPath()) return reply(vanish ? catalog(mainScope) : catalog(mainScope, worktree));
    if (path === scopedListPath(worktree)) return vanish ? reply({ error: "Unknown scope" }, 400)
      : reply(++worktreeLoads === 1 ? worktreeList : refreshed);
    if (path === detailPath(worktreeList, item)) return reply(diff(worktreeList, item));
    if (path === detailPath(refreshed, nextItem)) return reply(diff(refreshed, nextItem, { diff: "+refreshed\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.history.state.changes.depth, 2);
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [scopedListPath(worktree), detailPath(refreshed, nextItem), scopesPath()]);
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.$("changes-diff").textContent, "+refreshed\n");
  vanish = true;
  h.click("changes-refresh");
  await settle();
  h.flushHistory();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [scopedListPath(worktree), listPath(), scopesPath()]);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.$("changes-scope-control").hidden, true);
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.rows()[0].title, currentItem.path);
  assert.match(h.$("changes-scope-note").textContent, /feature is no longer available.*session directory/);
  assert.equal(h.history.state.changes.depth, 1);
  h.back();
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.document.activeElement, h.$("changesbtn"));
});

test("the labeled scope picker is in the desktop Tab loop", async () => {
  const worktree = scope("alternate", "another worktree");
  const list = scopedManifest(mainScope, [file("one", "first.txt")]);
  const h = await opened({ list, respond: ({ path }) => path === scopesPath() ? reply(catalog(mainScope, worktree)) : undefined });
  await settle();
  const picker = h.$("changes-scope");
  assert.equal(h.$("changes-scope-label").getAttribute("for"), "changes-scope");
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, h.$("changes-directory"));
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, picker);
  h.key("Tab");
  assert.equal(h.document.activeElement, h.rows()[0]);
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, picker);
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, h.$("changes-directory"));
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, h.$("changes-close"));
});

test("slow or failed scope discovery does not block a nonempty list or its diff", async () => {
  const item = file("one", "first.txt"), list = scopedManifest(mainScope, [item]);
  const pendingScopes = deferred();
  const h = await opened({ list, respond: ({ path }) => {
    if (path === scopesPath()) return pendingScopes.promise;
    if (path === detailPath(list, item)) return reply(diff(list, item, { diff: "+visible\n" }));
  } });
  await settle();
  const lookup = h.calls.at(-1);
  assert.equal(lookup.path, scopesPath());
  assert.equal(h.rows().length, 1);
  h.rows()[0].dispatch("click");
  await settle();
  assert.equal(lookup.signal.aborted, false);
  pendingScopes.resolve(reply({ error: "Scope discovery failed" }, 500));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "+visible\n");
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.$("changes-scope-control").hidden, true);
});

test("an empty truncated file list does not auto-select a dirty worktree", async () => {
  const worktree = scope("alternate", "another worktree", { files: 1 });
  const list = scopedManifest(mainScope, [], { truncated: true, reason: "File list limit reached." });
  const h = await opened({ list, respond: ({ path }) => path === scopesPath() ? reply(catalog(mainScope, worktree)) : undefined });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
  assert.equal(h.$("changes-scope").value, mainScope.id);
  assert.equal(h.$("changes-list-status").textContent, "No files could be listed.");
  assert.equal(h.$("changes-list-warning").textContent, "File list truncated. File list limit reached.");
  assert.equal(h.$("changes-scope-notice").hidden, true);
});

test("a truncated worktree catalog never auto-selects but keeps known worktrees selectable", async () => {
  const worktree = scope("alternate", "another worktree", { files: 1 });
  const item = file("one", "worktree.txt");
  const partial = { ...catalog(mainScope, worktree), truncated: true, reason: "Worktree scan timed out." };
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === scopesPath()) return reply(partial);
    if (path === scopedListPath(worktree)) return reply(scopedManifest(worktree, [item]));
  } });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
  assert.equal(h.$("changes-scope").value, mainScope.id);
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(h.$("changes-scope-notice").hidden, true);
  assert.match(h.$("changes-scopes-status").textContent, /Worktree scan incomplete.*timed out/);
  h.setScope(worktree);
  await settle();
  assert.equal(h.calls.at(-1).path, scopedListPath(worktree));
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.equal(h.rows()[0].title, item.path);
});

test("only a complete refreshed catalog missing the selected worktree returns to the session directory", async () => {
  const worktree = scope("alternate", "feature", { files: 1 });
  const oldItem = file("alternate-file", "worktree.txt"), currentItem = file("current-file", "current.txt");
  const worktreeList = scopedManifest(worktree, [oldItem]);
  let currentList = scopedManifest(mainScope), nextCatalog = reply(catalog(mainScope, worktree));
  const h = await opened({ phone: true, respond: ({ path }) => {
    if (path === listPath()) return reply(currentList);
    if (path === scopesPath()) return nextCatalog;
    if (path === scopedListPath(worktree)) return reply(worktreeList);
    if (path === detailPath(worktreeList, oldItem)) return reply(diff(worktreeList, oldItem, { diff: "+old\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.history.state.changes.depth, 2);
  const picker = h.$("changes-scope");
  nextCatalog = reply({ ...catalog(mainScope), truncated: true });
  h.click("changes-refresh");
  await settle();
  assert.equal(picker.value, worktree.id);
  assert.ok(h.scopeOptions().some((option) => option.value === worktree.id));
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.$("changes-diff").textContent, "+old\n");
  assert.doesNotMatch(h.$("changes-scope-note").textContent, /no longer available/);

  const pendingCatalog = deferred(), beforeRefresh = h.calls.length;
  nextCatalog = pendingCatalog.promise;
  currentList = scopedManifest(mainScope, [currentItem]);
  h.click("changes-refresh");
  await settle();
  assert.equal(h.rows()[0].title, oldItem.path);
  assert.equal(h.$("changes-diff").textContent, "+old\n");
  pendingCatalog.resolve(reply(catalog(mainScope)));
  await settle();
  h.flushHistory();
  assert.deepEqual(h.calls.slice(beforeRefresh).map((call) => call.path),
    [scopedListPath(worktree), detailPath(worktreeList, oldItem), scopesPath(), listPath()]);
  assert.equal(picker.value, mainScope.id);
  assert.equal(h.$("changes-scope-control").hidden, true);
  assert.equal(h.$("changes-listpane").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.deepEqual(h.rows().map((row) => row.title), [currentItem.path]);
  assert.match(h.$("changes-scope-note").textContent, /feature is no longer available.*session directory/);
  assert.equal(h.history.state.changes.depth, 1);
});

test("a 503 on refresh invalidates cached choices and Retry only rediscovers worktrees", async () => {
  const first = scope("first", "first worktree", { files: 1 }), second = scope("second", "second worktree", { files: 2 });
  const fresh = scope("fresh", "fresh worktree", { files: 1 });
  let nextCatalog = reply(catalog(mainScope, first, second));
  const h = await opened({ phone: true, list: scopedManifest(mainScope), respond: ({ path }) => path === scopesPath() ? nextCatalog : undefined });
  await settle();
  const picker = h.$("changes-scope"), choices = h.$("changes-scope-choices");
  assert.equal(choices.hidden, false);
  picker.focus();
  nextCatalog = reply({ error: "Worktree discovery temporarily unavailable." }, 503);
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [scopedListPath(mainScope), scopesPath()]);
  assert.deepEqual(h.scopeOptions().map((option) => option.value), [mainScope.id]);
  assert.equal(h.$("changes-scope-control").hidden, true);
  assert.equal(choices.hidden, true);
  assert.equal(choices.children.length, 0);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-count").textContent, "0");
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.document.activeElement, h.$("changes-refresh"));

  const status = h.$("changes-scopes-status"), retry = h.$("changes-scopes-retry");
  assert.equal(status.getAttribute("role"), "status");
  assert.match(status.textContent, /Could not discover worktrees.*503.*temporarily unavailable/);
  assert.equal(retry.getAttribute("aria-describedby"), "changes-scopes-status");
  h.$("changes-close").focus();
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, h.$("changes-directory"));
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, retry);
  const pendingRetry = deferred(), beforeRetry = h.calls.length;
  nextCatalog = pendingRetry.promise;
  h.click("changes-scopes-retry");
  assert.equal(h.document.activeElement, h.$("changes-refresh"));
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-count").textContent, "0");
  pendingRetry.resolve(reply(catalog(mainScope, fresh)));
  await settle();
  assert.deepEqual(h.calls.slice(beforeRetry).map((call) => call.path), [scopesPath()]);
  assert.equal(status.hidden, true);
  assert.equal(retry.hidden, true);
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(picker.value, mainScope.id);
  assert.deepEqual(h.scopeOptions().map((option) => option.value), [mainScope.id, fresh.id]);
});

test("a stale scope catalog is aborted on session switch and cannot auto-select in the new session", async () => {
  const pendingScopes = deferred();
  const worktree = scope("stale", "old worktree", { files: 1 });
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => path === scopesPath() ? pendingScopes.promise : undefined });
  await settle();
  const lookup = h.calls.at(-1);
  assert.equal(lookup.path, scopesPath());
  h.context.switchSession("other", false);
  assert.equal(lookup.signal.aborted, true);
  assert.equal(h.$("changes").hidden, true);
  h.flushHistory();
  h.click("changesbtn");
  pendingScopes.resolve(reply(catalog(mainScope, worktree)));
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [listPath("other"), scopesPath("other")]);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-scope-control").hidden, true);
});

test("Committed lists the selected worktree's commits with local times, subject, branch, and HEAD", async () => {
  const worktree = scope("linked/α?+%", "linked worktree", { files: 1 });
  const item = file("one", "working.txt");
  const h = await opened({ list: scopedManifest(mainScope, [item]), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, worktree));
    if (path === scopedListPath(worktree)) return reply(scopedManifest(worktree, [item]));
    if (path === commitsPath(worktree)) return reply(commitHistory(worktree, [headCommit, olderCommit], { branch: "feature/notes" }));
    if (path === commitsPath(mainScope)) return reply(commitHistory(mainScope, [olderCommit], { branch: null }));
  } }, commitsEnabled);
  await settle();
  assert.equal(h.$("changes-origin-label").textContent, "Working tree · uncommitted");
  assert.equal(h.$("changes-modes").hidden, false);
  h.setScope(worktree);
  await settle();
  h.$("changes-close").focus();
  h.key("Tab");
  assert.equal(h.document.activeElement, h.$("changes-directory"));
  h.key("Tab");
  assert.equal(h.document.activeElement, h.$("changes-mode-working"));
  h.key("Tab");
  const committed = h.$("changes-mode-committed");
  assert.equal(h.document.activeElement, committed);
  assert.equal(committed.getAttribute("type"), "button");
  assert.equal(h.key("Enter").defaultPrevented, undefined);
  await openCommitList(h);
  const request = h.calls.at(-1);
  assert.deepEqual([request.method, request.path, request.headers.Authorization, request.cache],
    ["GET", commitsPath(worktree), "Bearer " + token, "no-store"]);
  assert.equal(committed.getAttribute("aria-pressed"), "true");
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.equal(h.$("changes-branch").textContent, "feature/notes");
  assert.equal(h.$("changes-root").textContent, worktree.label);
  assert.equal(h.$("changes-files").hidden, true);
  assert.equal(h.rows().length, 0);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.commitRows().length, 2);
  const [id, subject, meta] = h.commitRows()[0].children, [time, head] = meta.children;
  assert.deepEqual([id.tagName, id.type, id.textContent, id.title], ["button", "button", "ab123cd", headCommit.id]);
  assert.equal(subject.textContent, headCommit.subject);
  const date = new Date(headCommit.committed_at * 1000);
  assert.deepEqual([time.tagName, time.textContent, time.getAttribute("datetime")], ["time", date.toLocaleString(), date.toISOString()]);
  assert.deepEqual([head.textContent, head.hidden, h.commitRows()[1].children[2].children[1].hidden], ["HEAD", false, true]);
  h.setScope(mainScope);
  await settle();
  assert.equal(h.calls.at(-1).path, commitsPath(mainScope));
  assert.equal(h.$("changes-branch").textContent, "detached HEAD");
  assert.equal(h.commitRows()[0].children[0].title, olderCommit.id);
});

test("a commit ID opens the full-OID unified diff and Back restores the existing list", async () => {
  const raw = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+<new>\n"
    + "diff --git a/b b/b\n--- a/b\n+++ b/b\n@@ -0,0 +1 @@\n+second\n";
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === commitsPath(mainScope)) return reply(commitHistory(mainScope, [headCommit, olderCommit]));
    if (path === commitPath(mainScope, headCommit)) return reply(commitDiff(mainScope, headCommit, { diff: raw }));
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  const before = [...h.commitRows()], id = before[0].children[0];
  assert.equal(h.calls.some((call) => call.path === commitPath(mainScope, headCommit)), false);
  h.key("Tab");
  assert.equal(h.document.activeElement, h.$("changes-branch"));
  h.key("Tab");
  assert.equal(h.document.activeElement, id);
  // Native buttons activate with Space; the fixture dispatches the resulting click.
  assert.equal(h.key(" ").defaultPrevented, undefined);
  id.dispatch("click");
  assert.equal(h.calls.at(-1).path, commitPath(mainScope, headCommit));
  assert.equal(h.$("changes-listpane").hidden, true);
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.$("changes-commits-back").textContent, "Back to commits");
  assert.equal(h.document.activeElement, h.$("changes-commits-back"));
  await settle();
  assert.equal(h.$("changes-path").textContent, "ab123cd · " + headCommit.subject);
  assert.equal(h.$("changes-commit-base").textContent, "first parent");
  assert.equal(h.$("changes-diff").textContent, raw);
  assert.deepEqual(h.$("changes-diff").children.filter((line) => line.tagName === "span").map((line) => line.className),
    ["changes-line meta", "changes-line meta", "changes-line meta", "changes-line hunk", "changes-line del", "changes-line add",
      "changes-line meta", "changes-line meta", "changes-line meta", "changes-line hunk", "changes-line add"]);
  h.key("Tab");
  assert.equal(h.document.activeElement, h.$("changes-detail-scroll"));
  h.key("Tab");
  assert.equal(h.document.activeElement, h.$("changes-root"));
  h.key("Tab");
  assert.equal(h.document.activeElement, h.$("changes-refresh"));
  const requests = h.calls.length;
  h.click("changes-commits-back");
  h.flushHistory();
  assert.equal(h.calls.length, requests);
  assert.equal(h.$("changes-listpane").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.deepEqual(h.commitRows(), before);
  assert.equal(h.document.activeElement, id);
});

test("Load more uses the full cursor, appends and keeps the existing commit rows", async () => {
  const next = deferred();
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === commitsPath(mainScope)) return reply(commitHistory(mainScope, [headCommit], { next_cursor: headCommit.id }));
    if (path === commitsPath(mainScope, headCommit.id)) return next.promise;
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  const first = h.commitRows()[0], more = h.$("changes-load-more");
  assert.equal(more.hidden, false);
  more.focus();
  h.click("changes-load-more");
  assert.equal(h.calls.at(-1).path, commitsPath(mainScope, headCommit.id));
  assert.equal(more.getAttribute("aria-disabled"), "true");
  const requests = h.calls.length;
  h.click("changes-load-more");
  assert.equal(h.calls.length, requests);
  next.resolve(reply(commitHistory(mainScope, [olderCommit], { head: headCommit.id })));
  await settle();
  assert.equal(h.commitRows()[0], first);
  assert.deepEqual(h.commitRows().map((row) => row.children[0].title), [headCommit.id, olderCommit.id]);
  assert.equal(h.$("changes-count").textContent, "2");
  assert.equal(more.hidden, true);
  assert.equal(h.document.activeElement, h.commitRows()[1].children[0]);
});

test("changing the worktree drops the list and ignores a late page body", async () => {
  const worktree = scope("linked", "linked worktree"), next = deferred();
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, worktree));
    if (path === commitsPath(mainScope)) return reply(commitHistory(mainScope, [headCommit], { next_cursor: headCommit.id }));
    if (path === commitsPath(mainScope, headCommit.id)) return { ...reply(null), json: () => next.promise };
    if (path === commitsPath(worktree)) return reply(commitHistory(worktree, [olderCommit], { branch: "linked" }));
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  h.click("changes-load-more");
  const previous = h.calls.at(-1);
  await settle();
  h.setScope(worktree);
  assert.equal(previous.signal.aborted, true);
  assert.equal(h.commitRows().length, 0);
  assert.equal(h.$("changes-load-more").hidden, true);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.equal(h.calls.at(-1).path, commitsPath(worktree));
  await settle();
  next.resolve(commitHistory(mainScope, [headCommit]));
  await settle();
  assert.deepEqual(h.commitRows().map((row) => row.children[0].title), [olderCommit.id]);
  assert.equal(h.$("changes-branch").textContent, "linked");
  assert.equal(h.$("changes-scope").value, worktree.id);
});

test("mode and session switches reset commit state and reject old list and diff bodies", async () => {
  const lateDiff = deferred(), lateList = deferred();
  const nextSession = "other/β";
  let lists = 0;
  const h = await opened({ list: scopedManifest(mainScope, [file("work", "working.txt")]), respond: ({ path }) => {
    if (path === commitsPath(mainScope)) return ++lists === 1 ? reply(commitHistory(mainScope, [headCommit]))
      : { ...reply(null), json: () => lateList.promise };
    if (path === commitsPath(mainScope, null, nextSession)) return reply(commitHistory(mainScope, [olderCommit]));
    if (path === commitPath(mainScope, headCommit)) return { ...reply(null), json: () => lateDiff.promise };
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  h.commitRows()[0].children[0].dispatch("click");
  const previousDiff = h.calls.at(-1);
  await settle();
  h.click("changes-mode-working");
  assert.equal(previousDiff.signal.aborted, true);
  assert.equal(h.commitRows().length, 0);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.equal(h.$("changes-path").textContent, "File diff");
  h.flushHistory();
  await settle();
  assert.equal(h.rows()[0].title, "working.txt");
  await openCommitList(h);
  const previousList = h.calls.at(-1);
  assert.equal(previousList.path, commitsPath(mainScope));
  assert.equal(h.commitRows().length, 0);
  h.context.switchSession(nextSession, false);
  assert.equal(previousList.signal.aborted, true);
  assert.equal(h.$("changes").hidden, true);
  h.flushHistory();
  h.click("changesbtn");
  assert.equal(h.$("changes-mode-working").getAttribute("aria-pressed"), "true");
  await settle();
  await openCommitList(h);
  assert.ok(h.calls.some((call) => call.path === commitsPath(mainScope, null, nextSession)));
  lateList.resolve(commitHistory(mainScope, [headCommit]));
  lateDiff.resolve(commitDiff(mainScope, headCommit, { diff: "+stale detail\n" }));
  await settle();
  assert.deepEqual(h.commitRows().map((row) => row.children[0].title), [olderCommit.id]);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.equal(h.$("changes-detail").hidden, true);
});

test("older daemons hide Committed and a revoked capability aborts the active history", async () => {
  const pending = deferred();
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => path === commitsPath(mainScope) ? pending.promise : undefined });
  await settle();
  assert.equal(h.$("changes-mode-committed").closest("[hidden]"), h.$("changes-modes"));
  const before = h.calls.length;
  await openCommitList(h);
  assert.equal(h.calls.length, before);
  await h.info(commitsEnabled);
  h.click("changes-mode-committed");
  const request = h.calls.at(-1);
  assert.equal(request.path, commitsPath(mainScope));
  await h.info({ features: { changes: true, changes_commits: "true" } });
  assert.equal(request.signal.aborted, true);
  assert.equal(h.$("changes-modes").hidden, true);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-origin-label").textContent, "Working tree · uncommitted");
  pending.resolve(reply(commitHistory(mainScope, [headCommit])));
  await settle();
  assert.equal(h.commitRows().length, 0);
});

test("Committed waits for Git availability but does not wait for the worktree scan", async () => {
  const current = scope("current", "session repo", { current: true });
  const worktree = scope("linked", "linked worktree", { files: 1 }), working = deferred(), lateCatalog = deferred();
  let scans = 0;
  const h = await opened({ respond: ({ path }) => {
    if (path === listPath()) return working.promise;
    if (path === commitsPath(current)) return reply(commitHistory(current, []));
    if (path === scopesPath()) return ++scans === 1 ? lateCatalog.promise : reply(catalog(current, worktree));
  } }, commitsEnabled);
  const first = h.calls.at(-1);
  assert.equal(h.$("changes-modes").hidden, true);
  await openCommitList(h);
  assert.equal(first.signal.aborted, false);
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath()]);
  working.resolve(reply(scopedManifest(current, [file("one", "working.txt")])));
  await settle();
  const scan = h.calls.at(-1);
  assert.equal(scan.path, scopesPath());
  assert.equal(h.$("changes-modes").hidden, false);
  await openCommitList(h);
  assert.equal(scan.signal.aborted, true);
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath(), commitsPath(current), scopesPath()]);
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(h.$("changes-scope").value, current.id);
  assert.equal(h.$("changes-commits-status").textContent, "No commits yet");
  lateCatalog.resolve(reply(catalog(current)));
  await settle();
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(h.rows().length, 0);
});

test("history explains Git errors, retries, and distinguishes the unborn branch", async () => {
  let requests = 0;
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path !== commitsPath(mainScope)) return;
    if (++requests === 1) return reply({ error: "not a Git worktree" }, 409);
    if (requests === 2) return reply({ error: "Git busy" }, 503);
    return reply(commitHistory(mainScope, [], { branch: "unborn" }));
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  const status = h.$("changes-commits-status"), retry = h.$("changes-commits-retry");
  assert.equal(status.textContent, "Git history is unavailable for this worktree.");
  assert.equal(retry.hidden, false);
  h.click("changes-commits-retry");
  await settle();
  assert.equal(status.textContent, "Git is unavailable right now.");
  assert.equal(retry.hidden, false);
  h.click("changes-commits-retry");
  await settle();
  assert.equal(requests, 3);
  assert.equal(status.textContent, "No commits yet");
  assert.equal(h.$("changes-branch").textContent, "unborn");
  assert.equal(h.commitRows().length, 0);
  assert.equal(h.$("changes-load-more").hidden, true);
  assert.equal(retry.hidden, true);
});

test("a missing commit can retry its detail; empty diffs show the reason and base", async () => {
  let requests = 0;
  const reason = "Commit has no changes in the selected directory.";
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === commitsPath(mainScope)) return reply(commitHistory(mainScope, [headCommit, olderCommit]));
    if (path === commitPath(mainScope, headCommit)) return ++requests === 1 ? reply({ error: "off chain" }, 404)
      : reply(commitDiff(mainScope, headCommit, { base: "shallow_boundary", base_id: null, format: "none", reason }));
    if (path === commitPath(mainScope, olderCommit)) return reply(commitDiff(mainScope, olderCommit, { base: "root", base_id: null, format: "none" }));
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  h.commitRows()[0].children[0].dispatch("click");
  await settle();
  const status = h.$("changes-detail-status"), retry = h.$("changes-detail-retry");
  assert.equal(status.textContent, "Commit not found.");
  assert.equal(retry.hidden, false);
  h.click("changes-detail-retry");
  assert.equal(h.calls.at(-1).path, commitPath(mainScope, headCommit));
  await settle();
  assert.equal(status.textContent, reason);
  assert.equal(h.$("changes-commit-base").textContent, "shallow boundary");
  assert.equal(h.$("changes-diff").hidden, true);
  assert.equal(retry.hidden, true);
  h.click("changes-commits-back");
  h.flushHistory();
  h.commitRows()[1].children[0].dispatch("click");
  await settle();
  assert.equal(h.$("changes-commit-base").textContent, "root");
  assert.equal(status.textContent, "No changes in this worktree for this commit.");
});

test("phone Back restores commit paging and a failed next page retries the same cursor", async () => {
  const page = deferred(), detail = deferred(), cursorPath = commitsPath(mainScope, headCommit.id);
  let pages = 0;
  const last = commitHistory(mainScope, [olderCommit], { head: headCommit.id });
  const h = await opened({ phone: true, list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === commitsPath(mainScope)) return reply(commitHistory(mainScope, [headCommit], { next_cursor: headCommit.id }));
    if (path === cursorPath) {
      if (++pages === 1) return page.promise;
      if (pages === 2) return reply({ error: "Git busy" }, 503);
      return reply(last);
    }
    if (path === commitPath(mainScope, headCommit)) return detail.promise;
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  const modes = h.$("changes-modes"), more = h.$("changes-load-more"), scroll = h.$("changes-list-scroll");
  assert.notEqual(modes.parent, h.$("changes-refresh").parent);
  assert.match(between('<style id="changes-style">', '</style>', html), /#changes-modes\s*\{[^}]*flex-wrap:wrap/);
  scroll.scrollTop = 80;
  h.click("changes-load-more");
  const previousPage = h.calls.at(-1), id = h.commitRows()[0].children[0];
  id.dispatch("click");
  const previousDetail = h.calls.at(-1);
  assert.equal(previousPage.signal.aborted, true);
  assert.equal(h.$("changes-commits-back").hidden, false);
  assert.equal(modes.hidden, false);
  scroll.scrollTop = 0;
  h.back();
  assert.equal(previousDetail.signal.aborted, true);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(scroll.scrollTop, 80);
  assert.equal(h.document.activeElement, id);
  assert.equal(more.getAttribute("aria-disabled"), "false");
  page.resolve(reply(last));
  detail.resolve(reply(commitDiff(mainScope, headCommit, { diff: "+stale\n" })));
  await settle();
  assert.equal(h.commitRows().length, 1);
  assert.equal(h.$("changes-diff").textContent, "");
  more.focus();
  h.click("changes-load-more");
  assert.equal(h.calls.at(-1).path, cursorPath);
  await settle();
  assert.equal(h.$("changes-commits-status").textContent, "Git is unavailable right now.");
  assert.equal(h.document.activeElement, h.$("changes-commits-retry"));
  h.click("changes-commits-retry");
  assert.equal(h.calls.at(-1).path, cursorPath);
  await settle();
  assert.deepEqual(h.commitRows().map((row) => row.children[0].title), [headCommit.id, olderCommit.id]);
  assert.equal(more.hidden, true);
});

for (const [variant, label, phone] of [
  [{ source: "plain", reason: "not_a_repository" }, "Not a Git repository", true],
  [{ git_available: false, reason: "git_unavailable" }, "Git unavailable", false],
]) {
  test(`${label} renders a read-only file inventory and an empty refresh`, async () => {
    const paths = ["docs/<draft> & notes.md", "assets/banner.png"];
    const files = paths.map((name, i) => file("f" + i, name, { status: "file", additions: 0, deletions: 0, binary: i > 0, size: 29 }));
    const currentPath = "/workspace/plain directory";
    let empty = false;
    const h = await opened({ phone, respond: ({ path }) => {
      if (path === listPath()) return reply({ ...variant, revision: "plain:" + (empty ? "empty" : "files"), files: empty ? [] : files, truncated: false });
      if (path === scopesPath()) return reply({ source: "plain", git_available: false, current_path: currentPath,
        scopes: [{ id: "current", path: currentPath, label: "plain directory", is_current: true, dirty: false, file_count: empty ? 0 : files.length }], truncated: false });
    } }, commitsEnabled);
    await settle();
    assert.equal(h.$("changes-origin-label").textContent, label);
    assert.equal(h.$("changes-directory-path").textContent, currentPath);
    assert.equal(h.$("changes-list-title").textContent, "Files");
    assert.equal(h.$("changes-count").textContent, "2");
    assert.equal(h.$("changes-list-status").hidden, true);
    for (const id of ["changes-detail", "changes-modes", "changes-scope-control", "changes-commits", "changes-back"]) {
      assert.equal(h.$(id).hidden, true);
    }
    const rows = h.$("changes-files").children;
    assert.deepEqual(rows.map((row) => row.textContent), paths);
    for (const row of rows) {
      assert.equal(row.tagName, "li");
      assert.equal(row.getAttribute("aria-current"), null);
      assert.deepEqual(row.children, [], "inventory rows have no buttons, badges, or diff actions");
      row.dispatch("click");
    }
    await settle();
    assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
    empty = true;
    h.click("changes-refresh");
    await settle();
    assert.equal(h.$("changes-files").children.length, 0);
    assert.equal(h.$("changes-list-status").textContent, "No files in this directory.");
    assert.equal(h.$("changes-list-retry").hidden, true);
  });
}

for (const gitAvailable of [false, true]) {
  test(`a ${gitAvailable ? "successful empty" : "Git-unavailable"} catalog updates the current path without refresh`, async () => {
    const pending = deferred(), currentPath = "/workspace/current <directory>";
    const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => path === scopesPath() ? pending.promise : undefined }, commitsEnabled);
    await settle();
    const directory = h.$("changes-directory-path");
    assert.equal(directory.textContent, mainScope.path);
    pending.resolve(reply({ ...catalog(), git_available: gitAvailable, current_path: currentPath, reason: gitAvailable ? null : "not_a_repository" }));
    await settle();
    assert.equal(directory.textContent, currentPath);
    assert.equal(h.$("changes-scope-control").hidden, true);
    assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
  });
}

test("a plain catalog replaces earlier Git rows with the current directory inventory", async () => {
  const pending = deferred(), currentPath = "/workspace/plain";
  const first = scopedManifest(mainScope, [file("changed", "changed.txt")]);
  const inventory = { source: "plain", git_available: false, current_path: currentPath, revision: "plain:files",
    files: [file("f", "notes.txt", { status: "file", additions: 0, deletions: 0, size: 29 })], truncated: false };
  let gone = false;
  const h = await opened({ list: first, respond: ({ path }) => {
    if (path === scopesPath()) return pending.promise;
    if (path === listPath() && gone) return reply(inventory);
  } }, commitsEnabled);
  await settle();
  assert.equal(h.rows()[0].tagName, "button");
  gone = true;
  pending.resolve(reply({ source: "plain", git_available: false, current_path: currentPath,
    scopes: [{ id: "current", path: currentPath, label: "plain", is_current: true, dirty: false, file_count: 1 }], truncated: false }));
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath(), listPath()]);
  assert.equal(h.$("changes-directory-path").textContent, currentPath);
  assert.equal(h.$("changes-modes").hidden, true);
  assert.deepEqual(h.$("changes-files").children.map((row) => row.textContent), ["notes.txt"]);
});

test("non-Git inventory and a git_unavailable commit error both gate Committed; Retry reads inventory", async () => {
  let available = false;
  const h = await opened({ respond: ({ path }) => {
    if (path === listPath()) return reply(available ? scopedManifest(mainScope)
      : scopedManifest(mainScope, [], { git_available: false, reason: "not_a_repository" }));
    if (path === commitsPath(mainScope)) return reply({ code: "git_unavailable", reason: "git_unavailable" }, 409);
  } }, commitsEnabled);
  await settle();
  assert.equal(h.$("changes-modes").hidden, true);
  assert.match(h.$("changes-list-status").textContent, /Git history is unavailable/);
  assert.equal(h.$("changes-directory-path").textContent, mainScope.path, "the inspected directory is known even without Git");
  assert.equal(h.$("changes-list-retry").hidden, false);
  const before = h.calls.length;
  await openCommitList(h);
  assert.equal(h.calls.length, before);

  available = true;
  h.click("changes-list-retry");
  assert.equal(h.calls.at(-1).path, listPath());
  await settle();
  assert.equal(h.$("changes-modes").hidden, false);
  await openCommitList(h);
  assert.equal(h.calls.at(-1).path, commitsPath(mainScope));
  assert.equal(h.$("changes-modes").hidden, true);
  assert.equal(h.$("changes-mode-working").getAttribute("aria-pressed"), "true");
  assert.match(h.$("changes-list-status").textContent, /Git is unavailable/);
  assert.equal(h.$("changes-list-retry").hidden, false);
  assert.equal(h.commitRows().length, 0);

  available = false;
  h.click("changes-list-retry");
  assert.equal(h.calls.at(-1).path, listPath());
  await settle();
  assert.equal(h.calls.filter((call) => call.path === commitsPath(mainScope)).length, 1);
  assert.equal(h.$("changes-modes").hidden, true);
});

test("a non-Git worktree catalog revokes an earlier inventory confirmation", async () => {
  const response = deferred();
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => path === scopesPath() ? response.promise : undefined }, commitsEnabled);
  await settle();
  assert.equal(h.$("changes-modes").hidden, false);
  response.resolve(reply({ ...catalog(), git_available: false, reason: "not_a_repository" }));
  await settle();
  assert.equal(h.$("changes-modes").hidden, true);
  assert.match(h.$("changes-list-status").textContent, /Git history is unavailable/);
  assert.equal(h.$("changes-list-retry").hidden, false);
});

for (const changed of ["stale cursor", "new head", "moved scope"]) {
  test(`a ${changed} resets pagination to page one without appending the invalid page`, async () => {
    const cursor = headCommit.id, fresh = deferred();
    let paged = false;
    const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
      if (path === commitsPath(mainScope)) return paged ? fresh.promise
        : reply(commitHistory(mainScope, [headCommit], { next_cursor: cursor }));
      if (path !== commitsPath(mainScope, cursor)) return;
      paged = true;
      if (changed === "stale cursor") return reply({ code: "stale_cursor", head: olderCommit.id, reason: "HEAD advanced" }, 409);
      if (changed === "new head") return reply(commitHistory(mainScope, [olderCommit], { branch: "unexpected page" }));
      return reply(commitHistory({ ...mainScope, path: "/repo/moved" }, [olderCommit], { head: headCommit.id, branch: "unexpected page" }));
    } }, commitsEnabled);
    await settle();
    await openCommitList(h);
    assert.equal(h.commitRows()[0].children[0].title, headCommit.id);
    h.click("changes-load-more");
    await settle();
    assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [commitsPath(mainScope, cursor), commitsPath(mainScope)]);
    assert.equal(h.commitRows().length, 0);
    assert.match(h.$("changes-scope-note").textContent, /history changed/i);
    fresh.resolve(reply(commitHistory(mainScope, [olderCommit], { branch: "refreshed" })));
    await settle();
    assert.deepEqual(h.commitRows().map((row) => row.children[0].title), [olderCommit.id]);
    assert.equal(h.$("changes-branch").textContent, "refreshed");
    assert.equal(h.$("changes-load-more").hidden, true);
    assert.equal(h.$("changes-commits-retry").hidden, true);
  });
}

for (const [mode, shape] of ["working", "committed"].flatMap((mode) => [[mode, "coded"], [mode, "legacy"]])) {
  test(`${shape} unknown_scope from a ${mode} diff rediscovers worktrees and makes refresh target current`, async () => {
    const current = scope("current", "session repo", { current: true }), worktree = scope("linked", "linked worktree");
    const item = file("one", "working.txt"), linked = scopedManifest(worktree, [item]);
    const invalid = mode === "working" ? detailPath(linked, item) : commitPath(worktree, headCommit);
    const fallback = mode === "working" ? listPath() : commitsPath(current);
    let removed = false;
    const h = await opened({ phone: true, list: scopedManifest(current, [item]), respond: ({ path }) => {
      if (path === scopesPath()) return reply(removed ? catalog(current) : catalog(current, worktree));
      if (path === scopedListPath(worktree)) return reply(linked);
      if (path === commitsPath(worktree)) return reply(commitHistory(worktree, [headCommit]));
      if (path === commitsPath(current)) return reply(commitHistory(current, [olderCommit]));
      if (path === invalid) {
        const error = { error: "unknown changes scope" };
        return reply(shape === "coded" ? { ...error, code: "unknown_scope" } : error, 400);
      }
    } }, commitsEnabled);
    await settle();
    h.setScope(worktree);
    await settle();
    if (mode === "committed") await openCommitList(h);
    const row = mode === "working" ? h.rows()[0] : h.commitRows()[0].children[0];
    removed = true;
    row.dispatch("click");
    await settle();
    h.flushHistory();
    assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [invalid, fallback, scopesPath()]);
    assert.equal(h.$("changes-listpane").hidden, false);
    assert.equal(h.$("changes-detail").hidden, true);
    assert.equal(h.$("changes-scope").value, current.id);
    assert.deepEqual(h.scopeOptions().map((option) => option.value), [current.id]);
    assert.match(h.$("changes-scope-note").textContent, /linked worktree is no longer available.*session directory/);
    const before = h.calls.length;
    h.click("changes-refresh");
    await settle();
    assert.deepEqual(h.calls.slice(before).map((call) => call.path),
      [mode === "working" ? scopedListPath(current) : commitsPath(current), scopesPath()]);
  });
}

test("commit identity checks reject malformed rows, cross-scope pages and colliding short IDs", async () => {
  const wrongScope = scope("different", "wrong worktree");
  const sha256 = { ...headCommit, id: headCommit.id.padEnd(64, "a") }, cursor = sha256.id;
  const collision = sha256.id.slice(0, 7).padEnd(64, "f");
  let lists = 0, details = 0;
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === commitsPath(mainScope)) {
      if (++lists === 1) return reply(commitHistory(mainScope, [{ ...headCommit, id: headCommit.id.slice(0, 7) }], { head: headCommit.id }));
      return reply(commitHistory(mainScope, [sha256], { next_cursor: lists === 2 ? cursor : null }));
    }
    if (path === commitsPath(mainScope, cursor)) return reply(commitHistory(wrongScope, [sha256]));
    if (path === commitPath(mainScope, sha256)) {
      const valid = commitDiff(mainScope, sha256, { base: "root", base_id: null, diff: "@@ -0,0 +1 @@\n+expected\n" });
      if (++details === 1) return reply({ ...valid, commit_id: collision });
      if (details === 2) return reply({ ...valid, scope: scopeInfo(wrongScope) });
      return reply(valid);
    }
  } }, commitsEnabled);
  await settle();
  await openCommitList(h);
  const listStatus = h.$("changes-commits-status");
  assert.equal(listStatus.textContent, "Could not load commits.");
  assert.equal(h.commitRows().length, 0);
  h.click("changes-commits-retry");
  await settle();
  assert.equal(h.commitRows()[0].children[0].title, sha256.id);
  h.click("changes-load-more");
  await settle();
  assert.equal(h.$("changes-root").textContent, "");
  assert.equal(h.$("changes-scope").value, mainScope.id);
  assert.equal(h.commitRows().length, 0);
  assert.equal(listStatus.textContent, "Could not load commits.");
  h.click("changes-commits-retry");
  assert.equal(h.calls.at(-1).path, commitsPath(mainScope));
  await settle();

  h.commitRows()[0].children[0].dispatch("click");
  assert.equal(h.calls.at(-1).path, commitPath(mainScope, sha256));
  const detailStatus = h.$("changes-detail-status");
  for (let i = 0; i < 2; i++) {
    await settle();
    assert.equal(detailStatus.textContent, "Could not load this commit diff.");
    assert.equal(h.$("changes-diff").hidden, true);
    assert.equal(h.$("changes-detail-retry").hidden, false);
    h.click("changes-detail-retry");
  }
  await settle();
  assert.equal(h.$("changes-diff").textContent, "@@ -0,0 +1 @@\n+expected\n");
});

const changesColumnKey = "serve.changesColumnWidth";
const columnHandle = (h) => h.$("changes-column-resize");
const columnWidth = (h) => h.$("changes-body").style["--changes-file-width"];

test("Changes file column: captured pointer drag saves only the column and double-click restores the default", async () => {
  const modalSize = JSON.stringify({ width: 1080, height: 760 }), modalPosition = JSON.stringify({ x: 20, y: 30 });
  const storage = new Map([[changesSizeKey, modalSize], [changesPositionKey, modalPosition]]);
  const h = await opened({ storage });
  await settle();
  const separator = columnHandle(h), pointer = { pointerId: 5, pointerType: "mouse", button: 0, isPrimary: true };
  const send = (type, x, target = separator) => target.dispatch(type, { ...pointer, clientX: x });
  assert.equal(separator.hidden, false);
  assert.equal(columnWidth(h), "310px");
  assert.equal(send("pointerdown", 490).defaultPrevented, true);
  assert.equal(separator.hasPointerCapture(5), true);
  assert.equal(h.document.activeElement, separator);
  send("pointermove", 580);
  assert.equal(columnWidth(h), "400px");
  assert.equal(storage.has(changesColumnKey), false);
  const release = send("pointerup", 610, h.$("changes-scrim"));
  assert.equal(release.target, separator);
  assert.equal(separator.hasPointerCapture(5), false);
  assert.equal(columnWidth(h), "430px");
  assert.equal(storage.get(changesColumnKey), "430");
  assert.equal(storage.get(changesSizeKey), modalSize);
  assert.equal(storage.get(changesPositionKey), modalPosition);
  const reopened = await opened({ storage });
  await settle();
  assert.equal(columnWidth(reopened), "430px");
  assert.equal(columnHandle(reopened).dispatch("dblclick", { detail: 2 }).defaultPrevented, true);
  assert.equal(columnWidth(reopened), "310px");
  assert.equal(storage.has(changesColumnKey), false);
});

test("Changes file column: keyboard resizing clamps to both limits without losing a wider saved preference", async () => {
  const storage = new Map();
  const h = await opened({ storage, visualViewport: { width: 1440, height: 1000 } });
  await settle();
  const separator = columnHandle(h);
  assert.equal(separator.getAttribute("role"), "separator");
  assert.equal(separator.getAttribute("aria-orientation"), "vertical");
  assert.equal(separator.getAttribute("aria-controls"), "changes-listpane");
  assert.equal(separator.getAttribute("aria-label"), "Resize changed files column");
  h.$("changes-detail-scroll").focus();
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, separator);
  assert.equal(h.key("ArrowRight").defaultPrevented, true);
  assert.equal(columnWidth(h), "326px");
  h.key("ArrowLeft", { shiftKey: true });
  assert.equal(columnWidth(h), "262px");
  h.key("Home");
  h.key("ArrowLeft");
  assert.equal(columnWidth(h), "240px");
  assert.equal(separator.getAttribute("aria-valuemin"), "240");
  h.key("End");
  assert.equal(columnWidth(h), "634px", "at full width, 60 percent is the tighter limit");
  assert.equal(separator.getAttribute("aria-valuenow"), "634");
  assert.equal(storage.get(changesColumnKey), "634");
  const { visualViewport } = h.window;
  visualViewport.width = 580;
  visualViewport.dispatch("resize");
  assert.equal(columnWidth(h), "260px", "at reduced width, the diff still has 240px plus the separator");
  assert.equal(separator.getAttribute("aria-valuemax"), "260");
  assert.equal(storage.get(changesColumnKey), "634");
  visualViewport.width = 1440;
  visualViewport.dispatch("resize");
  assert.equal(columnWidth(h), "634px");
  const reopened = await opened({ storage });
  await settle();
  assert.equal(columnWidth(reopened), "634px");
});

test("Changes file column: phones and coarse pointers have no separator, and very narrow modals use Back", async () => {
  for (const layout of [{ phone: true }, { phone: true, finePointer: true }, { finePointer: false }]) {
    const h = await opened(layout);
    const separator = columnHandle(h);
    assert.equal(separator.hidden, true);
    assert.equal(separator.dispatch("pointerdown", { button: 0, pointerId: 5, clientX: 100 }).defaultPrevented, undefined);
    assert.equal(separator.hasPointerCapture(5), false);
    assert.equal(columnWidth(h), undefined);
  }
  const item = file("one", "changed.txt"), list = manifest([item]);
  const h = await opened({ list, respond: ({ path }) => path === detailPath(list, item) ? reply(diff(list, item)) : undefined });
  await settle();
  startChangesDrag(h, "right")("pointerup", -500, 0);
  assert.equal(columnHandle(h).hidden, true);
  assert.equal(h.$("changes-body").dataset.columns, "single");
  assert.equal(h.$("changes-detail").hidden, true);
  h.rows()[0].dispatch("click");
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.$("changes-back").hidden, false);
  h.click("changes-back");
  assert.equal(h.$("changes-listpane").hidden, false);
});

const sideRows = (h, side) => h.$(`changes-${side}-files`).children.map((item) => item.children[0]);
const sideDetailPath = (list, item, side) => listPath() + "/" + encodeURIComponent(item.id) + "?revision="
  + encodeURIComponent(list.revision) + "&side=" + side + (list.scope ? "&scope=" + encodeURIComponent(list.scope.id) : "");
const stagedFile = (id = "both/α", name = "new/name.txt", overrides = {}) => file(id, name, {
  status: "renamed", old_path: "previous/name.txt", staged_status: "R", worktree_status: "M",
  staged_additions: 7, staged_deletions: 2, additions: 3, deletions: 1, ...overrides,
});

test("Changes file sides: grouped rows use their own badges, counts, and diff URLs", async () => {
  const both = stagedFile();
  const stagedOnly = stagedFile("single", "staged.txt", { status: "modified", staged_status: "M", worktree_status: null });
  const untracked = file("loose", "untracked.txt", { status: "added", untracked: true, staged_status: null, worktree_status: "A" });
  const list = scopedManifest(mainScope, [both, stagedOnly, untracked]);
  const h = await opened({ list, respond: ({ path }) => {
    if (path === sideDetailPath(list, both, "staged")) return reply(diff(list, both, { side: "staged", diff: "staged content\n" }));
    if (path === sideDetailPath(list, both, "worktree")) return reply(diff(list, both, { side: "worktree", diff: "unstaged content\n" }));
  } });
  await settle();
  assert.equal(h.$("changes-file-groups").hidden, false);
  assert.equal(h.$("changes-count").textContent, "3", "the header counts unique files, not rows");
  assert.equal(h.$("changes-staged-count").textContent, "2");
  assert.equal(h.$("changes-unstaged-count").textContent, "2");
  assert.equal(h.rows().length, 0);
  const [staged, single] = sideRows(h, "staged"), [worktree, loose] = sideRows(h, "unstaged");
  assert.deepEqual(staged.children.map((node) => node.textContent), ["R", "previous/name.txt → new/name.txt", "+7−2"]);
  assert.deepEqual(worktree.children.map((node) => node.textContent), ["M", "new/name.txt", "+3−1"]);
  assert.equal(single.children[0].textContent, "M");
  assert.equal(single.children[0].className, "changes-badge");
  assert.match(loose.getAttribute("aria-label"), /Unstaged — Untracked: untracked.txt/);
  assert.equal(h.$("changes-commit-toggle").hidden, true, "commit requires its own daemon capability");
  staged.dispatch("click");
  assert.equal(h.calls.at(-1).path, sideDetailPath(list, both, "staged"));
  assert.equal(h.$("changes-diff-side").textContent, "Staged");
  await settle();
  assert.equal(h.$("changes-diff").textContent, "staged content\n");
  worktree.dispatch("click");
  assert.equal(h.calls.at(-1).path, sideDetailPath(list, both, "worktree"));
  assert.equal(h.$("changes-diff-side").textContent, "Unstaged");
  assert.equal(staged.getAttribute("aria-current"), "false");
  assert.equal(worktree.getAttribute("aria-current"), "true");
  await settle();
  assert.equal(h.$("changes-diff").textContent, "unstaged content\n");
});

test("Changes file sides: Refresh retains only the selected side across new file IDs", async () => {
  const both = stagedFile();
  const list = manifest([both]), fresh = stagedFile("new-id"), worktreeOnly = { ...fresh, staged_status: null };
  const refreshed = manifest([fresh], { revision: "fresh" });
  const final = manifest([worktreeOnly], { revision: "worktree-only" });
  let load = 0;
  const h = await opened({ respond: ({ path }) => {
    if (path === listPath()) return reply([list, refreshed, final][load++]);
    if (path === sideDetailPath(list, both, "staged")) return reply(diff(list, both));
    if (path === sideDetailPath(refreshed, fresh, "staged")) return reply(diff(refreshed, fresh));
  } });
  await settle();
  sideRows(h, "staged")[0].dispatch("click");
  await settle();
  h.click("changes-refresh");
  await settle();
  assert.ok(h.calls.some(({ path }) => path === sideDetailPath(refreshed, fresh, "staged")));
  assert.equal(sideRows(h, "staged")[0].getAttribute("aria-current"), "true");
  h.click("changes-refresh");
  await settle();
  assert.equal(sideRows(h, "staged").length, 0);
  assert.equal(sideRows(h, "unstaged")[0].getAttribute("aria-current"), "false");
  assert.equal(h.$("changes-diff-side").hidden, true);
  assert.ok(!h.calls.some(({ path }) => path === sideDetailPath(final, worktreeOnly, "worktree")));
});

test("Changes file sides: older manifests stay flat and retain the old diff contract", async () => {
  const item = file("old", "legacy.txt"), partial = file("partial", "legacy-partial.txt", { staged_status: "A" });
  const list = scopedManifest(mainScope, [item, partial]);
  const h = await opened({ list, respond: ({ path }) => path === detailPath(list, item) ? reply(diff(list, item)) : undefined });
  await settle();
  assert.equal(h.$("changes-file-groups").hidden, true);
  assert.equal(h.$("changes-commit-toggle").hidden, true);
  assert.deepEqual(h.rows().map((row) => row.title), [item.path, partial.path]);
  h.rows()[0].dispatch("click");
  assert.equal(h.calls.at(-1).path, detailPath(list, item));
  assert.equal(h.$("changes-diff-side").hidden, true);
});

const commitWriteEnabled = { features: { changes: true, changes_commits: true, changes_commit: true } };
const commitSubmitPath = "/sessions/" + encodeURIComponent(defaultId) + "/commit";
const commitMessagePath = (scope) => commitSubmitPath + "-message?scope=" + encodeURIComponent(scope.id);
async function openCommitPanel(h) {
  await settle();
  const toggle = h.$("changes-commit-toggle"), message = h.$("changes-commit-message");
  assert.equal(toggle.hidden, false);
  h.click("changes-commit-toggle");
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.equal(h.document.activeElement, message);
  return message;
}
function setCommitDraft(message, text) {
  message.value = text;
  message.dispatch("input");
}

test("Changes moving: the commit editor and selectable body text do not move the dialog", async () => {
  const h = await opened({ list: scopedManifest(mainScope, [stagedFile()]) }, commitWriteEnabled);
  const message = await openCommitPanel(h), commitPanel = h.$("changes-commit-panel");
  const input = commitPanel.appendChild(h.document.createElement("input"));
  const editable = commitPanel.appendChild(h.document.createElement("p"));
  editable.setAttribute("contenteditable", "true");
  editable.append("Editable body text");
  for (const target of [message, input, editable.children[0], h.$("changes-commit-scope-note")]) {
    const send = changesMovePointer(target);
    assert.equal(send("pointerdown").defaultPrevented, undefined);
    assert.equal(send("pointermove", 24, 16).defaultPrevented, undefined);
    assert.equal(send("mousemove", 24, 16).defaultPrevented, undefined);
    assert.equal(send("pointerup", 24, 16).defaultPrevented, undefined);
    assert.equal(changesHeader(h).hasPointerCapture(7), false);
    assert.equal(panelOffset(h), undefined);
  }
  setCommitDraft(message, "A draft remains editable");
  assert.equal(message.value, "A draft remains editable");
});

test("Changes conflicts: one group, no diff request, and commits blocked on desktop and phone", async () => {
  const conflict = stagedFile("conflict", "conflict.txt", {
    unmerged: true, status: "unknown", staged_status: "U", worktree_status: "U", old_path: null,
  });
  const staged = stagedFile("staged", "staged.txt", { status: "modified", staged_status: "M", worktree_status: null });
  const unstaged = stagedFile("worktree", "unstaged.txt", { staged_status: null });
  const list = scopedManifest(mainScope, [staged, unstaged, conflict]);
  for (const phone of [false, true]) {
    const h = await opened({ phone, list }, commitWriteEnabled);
    const message = await openCommitPanel(h);
    assert.deepEqual(sideRows(h, "staged").map((row) => row.title), [staged.path]);
    assert.deepEqual(sideRows(h, "unstaged").map((row) => row.title), [unstaged.path]);
    const [conflictRow] = sideRows(h, "conflicts");
    assert.equal(conflictRow.title, conflict.path);
    assert.equal(conflictRow.children[0].textContent, "U");
    assert.equal(sideRows(h, "conflicts").length, 1);
    assert.equal(h.$("changes-conflicts-count").textContent, "1");
    assert.equal(h.$("changes-file-groups").firstElementChild, h.$("changes-conflicts"));
    assert.equal(h.$("changes-count").textContent, "3");
    assert.equal(h.$("changes-staged-summary").textContent, "1 file staged, +7 −2");

    setCommitDraft(message, "Do not commit yet");
    assert.equal(h.$("changes-commit-submit").disabled, true);
    const notice = h.$("changes-commit-conflict-note");
    assert.equal(notice.hidden, false);
    assert.match(notice.textContent, /resolve.*conflicts.*before committing/i);
    h.click("changes-commit-submit");
    h.click("changes-commit-submit");
    assert.equal(h.calls.some(({ method }) => method === "POST"), false);

    const calls = h.calls.length;
    conflictRow.dispatch("click");
    await settle();
    assert.equal(h.calls.length, calls, "a conflict must not request either diff side");
    assert.equal(h.$("changes-diff").hidden, true);
    assert.match(h.$("changes-detail-status").textContent, /unresolved merge conflicts/i);
    assert.equal(h.$("changes-listpane").hidden, phone);
    assert.equal(h.$("changes-detail").hidden, false);
    assert.equal(columnHandle(h).hidden, phone);
    if (phone) {
      h.click("changes-back");
      h.flushHistory();
      assert.equal(h.$("changes-listpane").hidden, false);
      assert.equal(h.document.activeElement, conflictRow);
    }
  }
});

test("Changes conflicts: a clipped list blocks commit until Refresh shows resolution", async () => {
  const staged = stagedFile();
  const list = scopedManifest(mainScope, [staged], { conflict_count: 3, truncated: true });
  const h = await opened({ list, respond: ({ path }) => path === scopedListPath(mainScope)
    ? reply(scopedManifest(mainScope, [staged], { revision: "resolved" })) : undefined }, commitWriteEnabled);
  const message = await openCommitPanel(h), submit = h.$("changes-commit-submit");
  setCommitDraft(message, "Keep the draft while resolving");
  assert.equal(h.$("changes-conflicts-count").textContent, "3");
  assert.equal(sideRows(h, "conflicts").length, 0);
  assert.equal(submit.disabled, true);

  h.click("changes-refresh");
  assert.equal(submit.disabled, true, "submission remains disabled while the list reloads");
  await settle();
  assert.equal(h.$("changes-conflicts").hidden, true);
  assert.equal(h.$("changes-commit-conflict-note").hidden, true);
  assert.equal(submit.disabled, false);
  assert.equal(message.value, "Keep the draft while resolving");
  assert.equal(h.calls.some(({ method }) => method === "POST"), false);
});

test("Changes commit: two-step confirmation posts the exact scoped draft once and refreshes on success", async () => {
  const item = stagedFile(), list = scopedManifest(mainScope, [item]);
  const updated = scopedManifest(mainScope, [{ ...item, staged_status: null }], { revision: "after" });
  const result = deferred(), refresh = deferred();
  const h = await opened({ list, respond: ({ path }) => {
    if (path === commitSubmitPath) return result.promise;
    if (path === scopedListPath(mainScope)) return refresh.promise;
  } }, commitWriteEnabled);
  const message = await openCommitPanel(h), submit = h.$("changes-commit-submit"), generate = h.$("changes-commit-generate");
  const summary = h.$("changes-staged-summary"), postCalls = () => h.calls.filter(({ method }) => method === "POST");
  assert.equal(h.$("changes-commit-label").getAttribute("for"), "changes-commit-message");
  assert.equal(summary.textContent, "1 file staged, +7 −2");
  assert.equal(submit.disabled, true);
  h.key("Tab");
  assert.equal(h.document.activeElement, generate);
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, message);
  h.key("Escape");
  assert.equal(h.$("changes").hidden, false);
  setCommitDraft(message, "Rename sample\n\nDescribe why.");
  h.click("changes-commit-submit");
  assert.equal(postCalls().length, 0);
  assert.equal(submit.textContent, "确认提交？");
  setCommitDraft(message, "Rename sample\n\nUpdated body.");
  assert.equal(submit.textContent, "提交", "edits require a fresh confirmation");
  h.click("changes-commit-submit");
  assert.equal(postCalls().length, 0);
  submit.focus();
  h.click("changes-commit-submit");
  assert.equal(postCalls().length, 1);
  assert.equal(h.document.activeElement, message, "disabling the focused action keeps focus inside the modal");
  const [request] = postCalls();
  assert.equal(request.path, commitSubmitPath);
  assert.equal(request.headers.Authorization, "Bearer " + token);
  assert.equal(request.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(request.body), { scope: mainScope.id, message: "Rename sample\n\nUpdated body." });
  assert.equal(submit.disabled, true);
  assert.equal(generate.disabled, true);
  assert.equal(message.readOnly, true);
  h.click("changes-commit-submit");
  assert.equal(postCalls().length, 1);
  result.resolve(reply({ commit: { id: headCommit.id, short_id: "ab123cd", branch: "main" } }));
  await settle();
  assert.equal(h.calls.at(-1).path, scopedListPath(mainScope));
  assert.equal(sideRows(h, "staged").length, 0);
  assert.equal(message.value, "");
  assert.equal(h.$("changes-commit-panel").hidden, false);
  assert.equal(h.$("changes-commit-status").textContent, "Committed ab123cd");
  refresh.resolve(reply(updated));
  await settle();
  assert.equal(summary.textContent, "0 files staged, +0 −0");
  assert.equal(h.$("changes-staged-count").textContent, "0");
  assert.equal(sideRows(h, "unstaged").length, 1);
  assert.equal(h.$("changes-commit-panel").hidden, false);
  assert.equal(submit.disabled, true);
  assert.equal(postCalls().length, 1);
});

test("Changes commit: nothing_staged is inline, leaves the draft, and never resends the commit", async () => {
  const list = scopedManifest(mainScope, [stagedFile()]);
  const h = await opened({ list, respond: ({ path }) => {
    if (path === commitSubmitPath) return reply({ code: "nothing_staged", error: "no changes" }, 409);
    if (path === scopedListPath(mainScope)) return reply(scopedManifest(mainScope));
  } }, commitWriteEnabled);
  const message = await openCommitPanel(h);
  setCommitDraft(message, "This draft stays");
  h.click("changes-commit-submit");
  h.click("changes-commit-submit");
  await settle();
  assert.equal(h.calls.filter(({ method }) => method === "POST").length, 1);
  assert.match(h.$("changes-commit-status").textContent, /^Nothing staged.*nothing_staged/);
  assert.match(h.$("changes-commit-status").className, /\berror\b/);
  assert.equal(h.$("changes-staged-summary").textContent, "0 files staged, +0 −0");
  assert.equal(h.$("changes-commit-panel").hidden, false);
  assert.equal(message.value, "This draft stays");
  assert.equal(h.$("changes-commit-submit").textContent, "提交");
});

test("Changes commit: errors prefer message, then legacy error, and never display undefined", async () => {
  const list = scopedManifest(mainScope, [stagedFile()]);
  for (const [body, text] of [
    [{ code: "cwd_unresolved", message: "Use /cd to set the session directory.", error: "legacy text" }, "Use /cd to set the session directory."],
    [{ code: "cwd_unresolved", error: "legacy text" }, "legacy text"],
    [{ code: "cwd_unresolved" }, "Request failed."],
  ]) {
    const h = await opened({ list, respond: ({ path }) => path === commitSubmitPath
      ? { ...reply(body, 409), statusText: "" } : undefined }, commitWriteEnabled);
    const message = await openCommitPanel(h);
    setCommitDraft(message, "Keep draft after error");
    h.click("changes-commit-submit");
    h.click("changes-commit-submit");
    await settle();
    assert.equal(h.$("changes-commit-status").textContent, "Could not commit. (cwd_unresolved) 409 " + text);
    assert.equal(message.value, "Keep draft after error");
    assert.equal(h.calls.filter(({ method }) => method === "POST").length, 1);
  }
});

test("Changes commit AI: generated text and model are literal and replacing a draft requires confirmation", async () => {
  const list = scopedManifest(mainScope, [stagedFile()]);
  const confirm = [], replies = ["Generated <first>", "Do not replace", "Accepted generated body"];
  let allow = false;
  const h = await opened({ list, confirm: (question) => { confirm.push(question); return allow; }, respond: ({ path }) => {
    if (path === commitMessagePath(mainScope)) return reply({ message: replies.shift(), model: "source/model<name>" });
  } }, commitWriteEnabled);
  const message = await openCommitPanel(h);
  setCommitDraft(message, "  \n ");
  h.click("changes-commit-generate");
  const request = h.calls.at(-1);
  assert.equal(request.path, commitMessagePath(mainScope));
  assert.equal(request.method, "GET");
  assert.equal(request.cache, "no-store");
  assert.equal(request.body, undefined);
  await settle();
  assert.equal(confirm.length, 0);
  assert.equal(message.value, "Generated <first>");
  assert.equal(h.$("changes-commit-model").textContent, "Generated by source/model<name>");
  setCommitDraft(message, "Keep my own draft unless I agree");
  h.click("changes-commit-generate");
  await settle();
  assert.equal(confirm.length, 1);
  assert.match(confirm[0], /replace the current commit message/i);
  assert.equal(message.value, "Keep my own draft unless I agree");
  assert.equal(h.$("changes-commit-model").hidden, true);
  allow = true;
  h.click("changes-commit-generate");
  await settle();
  assert.equal(confirm.length, 2);
  assert.equal(message.value, "Accepted generated body");
  assert.equal(h.$("changes-commit-model").hidden, false);
  assert.ok(h.calls.every(({ body }) => body === undefined), "AI reads must not send the draft");
});

test("Changes commit AI: a typed failure keeps existing text without asking to replace it", async () => {
  const pending = deferred(), list = scopedManifest(mainScope, [stagedFile()]);
  const h = await opened({ list, respond: ({ path }) => path === commitMessagePath(mainScope) ? pending.promise : undefined }, commitWriteEnabled);
  const message = await openCommitPanel(h);
  setCommitDraft(message, "A retained handwritten message");
  h.click("changes-commit-generate");
  assert.equal(h.$("changes-commit-generate").disabled, true);
  assert.equal(h.$("changes-commit-submit").disabled, true);
  pending.resolve(reply({ code: "model_unavailable", error: "no model" }, 503));
  await settle();
  assert.equal(message.value, "A retained handwritten message");
  assert.equal(h.$("changes-commit-status").textContent, "Could not generate a commit message. (model_unavailable)");
  assert.match(h.$("changes-commit-status").className, /\berror\b/);
  assert.equal(h.$("changes-commit-model").hidden, true);
  assert.equal(h.$("changes-commit-generate").disabled, false);
});

test("Changes commit AI: a scope switch aborts an old generation and cannot replace another worktree's draft", async () => {
  const other = scope("linked/β", "linked tree"), pending = deferred(), item = stagedFile();
  const h = await opened({ list: scopedManifest(mainScope, [item]), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, other));
    if (path === scopedListPath(other)) return reply(scopedManifest(other, [item]));
    if (path === commitMessagePath(mainScope)) return pending.promise;
  } }, commitWriteEnabled);
  await openCommitPanel(h);
  h.click("changes-commit-generate");
  const { signal } = h.calls.at(-1);
  h.setScope(other);
  assert.equal(signal.aborted, true);
  const message = await openCommitPanel(h);
  setCommitDraft(message, "Message in the other worktree");
  pending.resolve(reply({ message: "Wrong worktree", model: "ignored" }));
  await settle();
  assert.equal(message.value, "Message in the other worktree");
  assert.equal(h.$("changes-commit-model").hidden, true);
});
