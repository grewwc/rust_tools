// Run with: node --test src/bin/ai/serve/app_skills.test.cjs
// Execute the embedded selector and HTTP acceptance path without dependencies.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const between = (start, end) => {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, `${start} .. ${end}`);
  return script.slice(a, b);
};
function nodeList(nodes = []) {
  const list = {
    ...nodes, length: nodes.length,
    item(index) { return this[index] ?? null; },
    forEach(callback, thisArg) { nodes.forEach((node, index) => callback.call(thisArg, node, index, list)); },
    [Symbol.iterator]() { return nodes[Symbol.iterator](); },
  };
  return list;
}
function select() {
  let value = "";
  return {
    options: [], disabled: false,
    get value() { return value; },
    set value(v) { value = this.options.some((o) => o.value === v) ? v : ""; },
    set innerHTML(_) { this.options = []; value = ""; },
    appendChild(o) { this.options.push(o); },
    get selectedIndex() { return this.options.findIndex((o) => o.value === value); },
  };
}
function harness({ storage = new Map(), base = "https://a.test", sid = "alpha", catalog = [
  { name: "review", description: "Review changes" },
  { name: "build", description: "Build software" },
] } = {}) {
  const document = {
    activeElement: null, listeners: {}, createElement: (tag) => node(tag),
    addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); },
  };
  function focusOut(from, to) {
    for (let parent = from; parent; parent = parent.parentElement) parent.onfocusout?.({ relatedTarget: to });
  }
  function node(tag = "div") {
    return {
      tagName: tag, value: "", hidden: true, disabled: false, children: [], attrs: {},
      set textContent(value) { this.text = String(value); this.children = []; },
      get textContent() { return (this.text || "") + this.children.map((c) => c.textContent).join(""); },
      appendChild(child) { child.parentElement = this; this.children.push(child); },
      setAttribute(key, value) { this.attrs[key] = String(value); },
      getAttribute(key) { return this.attrs[key] ?? null; },
      removeAttribute(key) { delete this.attrs[key]; },
      contains(target) { return target === this || this.children.some((child) => child.contains?.(target)); },
      focus(options) {
        if (document.activeElement === this) return;
        const previous = document.activeElement;
        document.activeElement = this; this.focusOptions = options;
        focusOut(previous, this); this.onfocus?.();
      },
      blur() { document.activeElement = null; focusOut(this, null); this.onblur?.(); },
      click() { if (!this.disabled) this.onclick?.(); },
      scrollIntoView() {},
    };
  }
  const elements = {
    skillpicker: node(), skillfield: node(), oskill: node("button"), skillvalue: node(),
    skillmenu: node(), skillsearchwrap: node(), skillsearch: node("input"), skillsearchclear: node("button"),
    skilloptions: node(), skillempty: node(), skillclear: node("button"), skillhint: node(), ovrwrap: {}, prompt: node(),
    omodel: select(), oagent: select(), oeffort: select(), ovrsummary: {},
  };
  for (const [parent, children] of [
    ["skillpicker", ["skillfield", "skillmenu", "skillhint"]],
    ["skillfield", ["oskill"]], ["oskill", ["skillvalue"]],
    ["skillmenu", ["skillsearchwrap", "skilloptions", "skillempty", "skillclear"]],
    ["skillsearchwrap", ["skillsearch", "skillsearchclear"]],
  ]) for (const child of children) elements[parent].appendChild(elements[child]);
  const requests = [];
  const context = vm.createContext({
    $: (id) => elements[id], document,
    store: { base }, sessionId: sid,
    sessionStorage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, String(value)), removeItem: (key) => storage.delete(key),
    },
    sendBusy: false, pendingImages: [],
    headers: () => ({ Authorization: "Bearer test" }),
    api: async (url) => {
      assert.equal(url, "/skills");
      return { json: async () => catalog };
    },
    fetch: async (url, opts) => { requests.push({ url, ...opts, body: JSON.parse(opts.body) }); return { ok: true }; },
  });
  vm.runInContext(between("// BEGIN execution model", "// END execution model"), context);
  vm.runInContext(between("function canSend()", "// The one send/stop button"), context);
  vm.runInContext(between("function syncOvrSummary()", "// The composer grows"), context);
  vm.runInContext(between("function fillSelect(", "async function loadInfo()"), context);
  vm.runInContext(between("// BEGIN skill picker events", "// END skill picker events"), context);
  vm.runInContext(between("async function send()", "  const rewindFloor =") + "\n}", context);
  const body = between("  const body = { prompt, confirm: true };", '  setPromptValue("", true);');
  const accept = between("// BEGIN turn HTTP acceptance", "// END turn HTTP acceptance")
    .replace("// BEGIN turn HTTP acceptance", "").replace(/^\s*try \{\s*/, "");
  vm.runInContext(`async function submit() { if (!validateSkillInput()) return; const prompt = "hello", images = [], sendBase = store.base, sid = sessionId; ${body}\n${accept}\nreturn body; }`, context);
  return {
    context, elements, requests, storage,
    open() { if (elements.skillmenu.hidden) elements.oskill.click(); },
    choose(name) { context.chooseSkill(name); },
    type(text, composing = false) {
      this.open(); elements.skillsearch.focus(); elements.skillsearch.value = text;
      elements.skillsearch.oninput({ isComposing: composing });
    },
    options() { return elements.skilloptions.children.filter((n) => n.attrs.role === "option"); },
    names() { return this.options().map((option) => option.children[0].children[0].textContent); },
    selected() { return Array.from(context.captureSkillPick().skills); },
    key(key, extra = {}) { const event = { key, preventDefault() { this.prevented = true; }, ...extra }; elements.skillmenu.onkeydown(event); return event; },
    outside(target = elements.prompt) { for (const listener of document.listeners.pointerdown || []) listener({ target }); },
    switchTo(sid) { context.sessionId = sid; context.restoreSkillPick(); },
  };
}

test("skill picker loads server names and describes next-message scope", async () => {
  const h = harness();
  await h.context.loadSkills();
  assert.match(html, /<button id="oskill" type="button"[^>]*aria-haspopup="listbox"/);
  assert.match(html, /<input id="skillsearch" type="text" role="combobox"/);
  assert.match(html, /id="skilloptions" role="listbox"[^>]*aria-multiselectable="true"/);
  h.open();
  assert.deepEqual(h.names(), ["review", "build"]);
  assert.match(h.elements.skillhint.textContent, /Next message only.*Session defaults still apply/);
  assert.equal(h.options()[0].children[0].children[1].textContent, "Review changes");
  assert.equal(h.elements.ovrsummary.textContent, "model default · agent default · effort default · skill default");
  h.choose("review");
  assert.equal(h.elements.skillvalue.textContent, "skill: review");
  assert.equal(h.elements.ovrsummary.textContent, "model default · agent default · effort default · skill review (next)");
});

test("tapping skills toggles an ordered set and sends every selected name", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.open(); h.options()[0].click(); h.options()[1].click();
  assert.deepEqual(h.selected(), ["review", "build"]);
  assert.equal(h.elements.skillvalue.textContent, "skill: review + build");
  assert.equal(h.elements.ovrsummary.textContent, "model default · agent default · effort default · skill review + build (next)");
  assert.deepEqual(h.options().map((o) => o.attrs["aria-selected"]), ["true", "true"]);
  assert.equal(h.elements.skillmenu.hidden, false);
  h.options()[0].click();
  assert.deepEqual(h.selected(), ["build"]);
  assert.equal(h.options()[0].attrs["aria-selected"], "false");
  h.options()[0].click();
  assert.deepEqual(h.selected(), ["build", "review"], "a reselected skill goes to the end");
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["build", "review"]);
  assert.deepEqual(h.selected(), []);
});

test("override controls and summary keep model, agent, effort, skill order", async () => {
  const controls = ["omodel", "oagent", "oeffort", "oskill"];
  const positions = controls.map((id) => html.indexOf(`id="${id}"`));
  assert.ok(positions.every((pos, i) => pos >= 0 && (!i || pos > positions[i - 1])));
  // The summary hosts the team trigger beside the text, so the text lives in
  // its own span inside the summary element.
  assert.match(html, /<summary id="ovrsummaryline">\s*<span id="ovrsummary">model · agent · effort · skill<\/span>/);
  const h = harness();
  await h.context.loadSkills();
  for (const [id, value] of [["omodel", "test-model"], ["oagent", "build"], ["oeffort", "high"]]) {
    h.elements[id].appendChild({ value, textContent: value });
    h.elements[id].value = value;
  }
  h.context.syncOvrSummary(); // Select change events refresh the summary independently of the search.
  h.type("unknown");
  assert.equal(h.elements.ovrsummary.textContent, "model test-model · agent build · effort high · skill default");
  h.choose("review");
  assert.equal(h.elements.ovrsummary.textContent, "model test-model · agent build · effort high · skill review (next)");
});

test("typing filters names and descriptions, and touch selection keeps the picker open", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.type("SOFTware");
  assert.deepEqual(h.names(), ["build"]);
  assert.equal(h.elements.oskill.attrs["aria-expanded"], "true");
  const touch = { preventDefault() { this.prevented = true; } };
  h.options()[0].onpointerdown(touch);
  assert.equal(touch.prevented, true);
  h.options()[0].onclick();
  assert.equal(h.elements.skillsearch.value, "SOFTware");
  assert.equal(h.elements.skillmenu.hidden, false);
  assert.equal(h.options()[0].attrs["aria-selected"], "true");
  h.type(" ReVieW ");
  assert.deepEqual(h.names(), ["review"]);
  h.options()[0].click();
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["build", "review"]);
});

test("clearing the filter retains selections and shows the complete catalog", async () => {
  for (const empty of ["", " \t "]) {
    const h = harness();
    await h.context.loadSkills();
    h.choose("review"); h.type("bui");
    h.key("ArrowDown");
    assert.equal(h.elements.skillmenu.hidden, false);
    assert.ok(h.elements.skillsearch.attrs["aria-activedescendant"]);
    h.type(empty);
    assert.equal(h.context.document.activeElement, h.elements.skillsearch);
    assert.equal(h.elements.skillmenu.hidden, false);
    assert.equal(h.elements.oskill.attrs["aria-expanded"], "true");
    assert.equal(h.elements.skillsearch.attrs["aria-activedescendant"], undefined);
    assert.deepEqual(h.names(), ["review", "build"]);
    assert.deepEqual(h.selected(), ["review"]);
    await h.context.submit();
    assert.deepEqual(h.requests[0].body.skills, ["review"]);
  }
});

test("an empty committed composition keeps the filter usable and preserves picks", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("review"); h.type("rev");
  h.elements.skillsearch.oncompositionstart();
  h.type("", true);
  assert.equal(h.elements.skillmenu.hidden, false, "the IME field must remain visible");
  h.elements.skillsearch.oncompositionend();
  h.type(""); // WebView may emit a final input event after compositionend.
  assert.equal(h.elements.skillmenu.hidden, false);
  assert.deepEqual(h.names(), ["review", "build"]);
  assert.deepEqual(h.selected(), ["review"]);
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["review"]);
});

test("keyboard navigation toggles choices without sending and Escape or focus outside closes", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.open();
  assert.equal(h.key("ArrowDown").prevented, true);
  assert.equal(h.elements.skillsearch.attrs["aria-activedescendant"], h.options()[0].id);
  assert.equal(h.options()[0].attrs["data-active"], "true");
  assert.equal(h.options()[0].attrs["aria-selected"], "false", "navigation is separate from selection");
  assert.equal(h.key("Enter").prevented, true);
  assert.deepEqual(h.selected(), ["review"]);
  h.key("ArrowDown"); h.key("Enter");
  assert.deepEqual(h.selected(), ["review", "build"]);
  h.key("Home"); h.key("Enter");
  assert.deepEqual(h.selected(), ["build"]);
  h.key("End"); h.key("Enter");
  assert.deepEqual(h.selected(), []);
  assert.equal(h.requests.length, 0);
  h.key("Escape");
  assert.equal(h.elements.skillmenu.hidden, true);
  assert.equal(h.context.document.activeElement, h.elements.oskill);
  assert.equal(h.elements.oskill.focusOptions.preventScroll, true);
  h.open();
  h.key("ArrowUp");
  assert.equal(h.options().at(-1).attrs["data-active"], "true");
  h.key("Enter");
  assert.deepEqual(h.selected(), ["build"]);
  h.elements.prompt.focus();
  assert.equal(h.elements.skillmenu.hidden, true);
  h.open();
  h.outside(h.options()[0].children[0]);
  assert.equal(h.elements.skillmenu.hidden, false, "tapping inside the list keeps it open");
  h.outside();
  assert.equal(h.elements.skillmenu.hidden, true);
  assert.deepEqual(h.selected(), ["build"]);
  assert.equal(h.elements.skillsearch.attrs["aria-activedescendant"], undefined);
});

test("an unmatched filter has an empty state and never becomes a forced skill", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.type("not-a-skill");
  assert.equal(h.elements.skillempty.hidden, false);
  assert.deepEqual(h.names(), []);
  assert.equal(h.context.validateSkillInput(), true);
  h.switchTo("beta"); h.type("bui"); h.switchTo("alpha");
  assert.equal(h.elements.skillsearch.value, "not-a-skill");
  await h.context.submit();
  assert.equal("skills" in h.requests[0].body, false);
  h.choose("build"); h.type("not-a-skill");
  await h.context.submit();
  assert.deepEqual(h.requests[1].body.skills, ["build"]);
});

test("actual turn request sends skills only when selected, clearing on HTTP success", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("review");
  await h.context.submit();
  assert.deepEqual(h.requests[0].body, { prompt: "hello", confirm: true, skills: ["review"] });
  assert.equal(h.requests[0].url, "https://a.test/sessions/alpha/turns/stream");
  assert.deepEqual(h.selected(), []);
  assert.equal(h.elements.skillvalue.textContent, "skill: session default");
  await h.context.submit();
  assert.match(h.elements.ovrsummary.textContent, /^model default · agent default · effort default · skill default$/);
  assert.equal("skills" in h.requests[1].body, false);
});

test("request payload trims padded selected names without changing their order", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.context.setSkillPick(h.context.skillPickKey(), ["  review  ", "\tbuild\r\n"], "");
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["review", "build"]);
  assert.deepEqual(h.selected(), []);
});

test("eight long chosen skill names remain inspectable in the constrained trigger", async () => {
  const catalog = Array.from({ length: 8 }, (_, i) => ({ name: "long-mobile-skill-selection-" + i }));
  const h = harness({ catalog });
  await h.context.loadSkills();
  for (const skill of catalog) h.choose(skill.name);
  const chosen = catalog.map((skill) => skill.name).join(" + ");
  assert.equal(h.elements.skillvalue.textContent, "skill: " + chosen);
  assert.ok(h.elements.oskill.title.includes(chosen));
  assert.ok(h.elements.oskill.attrs["aria-label"].includes(chosen));
  h.open();
  assert.deepEqual(h.names(), catalog.map((skill) => skill.name));
  assert.ok(h.options().every((option) => option.attrs["aria-selected"] === "true"));
  const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
  assert.match(css, /#skillpicker \{[^}]*flex:1; min-width:0;/);
  assert.match(css, /#skillvalue \{[^}]*flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;/);
});

test("HTTP rejection and transport failure retain the pending selection", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("build"); h.choose("review");
  h.context.fetch = async () => ({ ok: false, status: 409, json: async () => ({ error: "running" }) });
  await assert.rejects(h.context.submit(), /409 running/);
  assert.deepEqual(h.selected(), ["build", "review"]);
  h.context.fetch = async () => { throw new Error("offline"); };
  await assert.rejects(h.context.submit(), /offline/);
  assert.deepEqual(h.selected(), ["build", "review"]);
  const reloaded = harness({ storage: h.storage });
  await reloaded.context.loadSkills();
  assert.deepEqual(reloaded.selected(), ["build", "review"]);
});

test("late acceptance cannot clear another session or a reselected skill", async () => {
  const h = harness();
  await h.context.loadSkills();
  let resolve;
  h.context.fetch = () => new Promise((r) => { resolve = r; });
  h.choose("review");
  const pending = h.context.submit();
  h.switchTo("beta");
  assert.deepEqual(h.selected(), []);
  h.choose("build");
  resolve({ ok: true });
  await pending;
  assert.deepEqual(h.selected(), ["build"]);
  h.switchTo("alpha");
  assert.deepEqual(h.selected(), []);
  h.choose("review");
  const again = h.context.submit();
  h.choose(""); h.choose("review");
  resolve({ ok: true });
  await again;
  assert.deepEqual(h.selected(), ["review"]);
});

test("late HTTP acceptance consumes selected skills and preserves a newer independent filter", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("review");
  let resolve;
  h.context.fetch = () => new Promise((r) => { resolve = r; });
  const pending = h.context.submit();
  h.type("bui");
  resolve({ ok: true }); await pending;
  assert.equal(h.elements.skillsearch.value, "bui");
  assert.deepEqual(h.selected(), []);
  assert.equal(h.context.validateSkillInput(), true);
});

test("Chinese composition filters skills, preserves IME keys, and permits explicit clicks", async () => {
  const h = harness();
  h.context.api = async () => ({ json: async () => [
    { name: "mahjong", description: "麻将分析" },
    { name: "review", description: "Review changes" },
  ] });
  h.context.isTurnRunning = () => false;
  await h.context.loadSkills();
  h.open();
  assert.deepEqual(h.names(), ["mahjong", "review"]);
  h.elements.skillsearch.oncompositionstart();
  h.type("ma", true);
  assert.equal(h.elements.skillmenu.hidden, false);
  assert.deepEqual(h.names(), ["mahjong"], "composition must still filter the catalog");
  for (const key of ["ArrowDown", "ArrowUp", "Home", "End", "Enter"]) {
    for (const extra of [{}, { isComposing: true }, { keyCode: 229 }]) {
      assert.equal(h.key(key, extra).prevented, undefined);
    }
  }
  assert.equal(h.elements.skillsearch.value, "ma");
  assert.deepEqual(h.selected(), []);
  assert.equal(h.context.validateSkillInput(), false);
  h.elements.prompt.value = "hello";
  h.context.pendingImages = [{ filename: "photo.png" }];
  await h.context.send();
  assert.equal(h.elements.ovrwrap.open, true);
  assert.equal(h.elements.prompt.value, "hello");
  assert.equal(h.context.pendingImages.length, 1);
  assert.equal(h.requests.length, 0);
  assert.equal(h.context.sendBusy, false);

  h.options()[0].click();
  assert.equal(h.elements.skillsearch.value, "ma");
  assert.deepEqual(h.selected(), ["mahjong"], "an explicit click settles composition and selects");
  assert.equal(h.context.validateSkillInput(), true);
  assert.equal(h.elements.skillmenu.hidden, false);

  h.elements.skillsearch.oncompositionstart(); h.type("ma", true);
  assert.equal(h.context.validateSkillInput(), false);
  h.elements.skillsearchclear.click();
  assert.equal(h.elements.skillsearch.value, "");
  assert.deepEqual(h.names(), ["mahjong", "review"]);
  assert.deepEqual(h.selected(), ["mahjong"], "search clear preserves the pending choice");
  assert.equal(h.context.validateSkillInput(), true);
  assert.equal(h.elements.skillmenu.hidden, false);

  h.elements.skillsearch.oncompositionstart(); h.type("ma", true);
  assert.equal(h.context.validateSkillInput(), false);
  assert.equal(h.elements.skillclear.disabled, false);
  h.elements.skillclear.click();
  assert.equal(h.elements.skillsearch.value, "");
  assert.deepEqual(h.selected(), [], "session default clears pending choices during composition");
  assert.equal(h.context.validateSkillInput(), true);
  assert.equal(h.elements.skillmenu.hidden, true);

  h.open(); h.elements.skillsearch.oncompositionstart(); h.type("ma", true);
  h.elements.skillsearch.value = "麻将";
  h.elements.skillsearch.oncompositionend();
  assert.equal(h.elements.skillsearch.value, "麻将");
  assert.deepEqual(h.names(), ["mahjong"]);
  assert.equal(h.context.validateSkillInput(), true);
  for (const extra of [{ isComposing: true }, { keyCode: 229 }]) {
    assert.equal(h.key("Enter", extra).prevented, undefined);
  }
  assert.deepEqual(h.selected(), []);
  assert.equal(h.key("Enter").prevented, true);
  assert.deepEqual(h.selected(), ["mahjong"]);
  await h.context.submit();
  assert.equal(h.requests.length, 1);
  assert.deepEqual(h.requests[0].body.skills, ["mahjong"]);
});

test("skill names and descriptions render as text, never catalog-supplied markup", async () => {
  const h = harness();
  const name = '<img src=x onerror="alert(1)">';
  h.context.api = async () => ({ json: async () => [{ name, description: "<script>alert(2)</script>" }] });
  await h.context.loadSkills(); h.open();
  const option = h.options()[0];
  assert.equal(option.children[0].children[0].textContent, name);
  assert.equal(option.children[0].children[1].textContent, "<script>alert(2)</script>");
  assert.equal(option.children[0].children[0].children.length, 0);
  assert.doesNotMatch(between("function renderSkillMenu()", "function skillKeydown("), /innerHTML/);
});

test("pending choices survive session switching and move into the first created session", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("review"); h.choose("build"); h.switchTo("beta"); h.choose("build"); h.switchTo("alpha");
  assert.deepEqual(h.selected(), ["review", "build"]);
  h.switchTo(""); h.choose("build"); h.choose("review");
  h.context.localStorage = { setItem() {} };
  h.context.pendingNew = null;
  h.context.carryTeamPick = () => {};
  vm.runInContext(between("// BEGIN composer session adoption", "// END composer session adoption"), h.context);
  assert.equal(h.context.adoptComposerSession("new", "", h.context.store.base), true);
  assert.equal(h.context.pendingNew, "new");
  assert.deepEqual(h.selected(), ["build", "review"]);
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["build", "review"]);
  h.switchTo("");
  assert.deepEqual(h.selected(), []);
  assert.equal(h.storage.has('serve.skill-draft:["https://a.test",""]'), false);
});

test("catalog removes obsolete choices and empty or failed catalogs preserve normal send", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("review"); h.choose("build");
  h.context.api = async () => ({ json: async () => [{ name: "build" }] });
  await h.context.loadSkills();
  assert.deepEqual(h.selected(), ["build"]);
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["build"]);
  h.context.api = async () => ({ json: async () => [] });
  await h.context.loadSkills();
  assert.equal(h.elements.oskill.disabled, true);
  assert.match(h.elements.skillvalue.textContent, /none available/);
  h.context.api = async () => { throw new Error("offline"); };
  await h.context.loadSkills();
  assert.match(h.elements.skillvalue.textContent, /unavailable/);
  await h.context.submit();
  assert.equal("skills" in h.requests[1].body, false);
});

test("an empty skill discovery is retried on opening overrides and a nonempty catalog is reused", async () => {
  const h = harness();
  let calls = 0;
  h.context.api = async () => ({ json: async () => ++calls === 1 ? [] : [{ name: "review" }] });
  await h.context.loadSkills();
  assert.equal(h.elements.oskill.disabled, true);
  h.elements.ovrwrap.open = true;
  await h.elements.ovrwrap.ontoggle();
  assert.equal(calls, 2);
  assert.equal(h.elements.oskill.disabled, false);
  h.open();
  assert.deepEqual(h.names(), ["review"]);
  h.elements.ovrwrap.open = false; await h.elements.ovrwrap.ontoggle();
  h.elements.ovrwrap.open = true; await h.elements.ovrwrap.ontoggle();
  assert.equal(calls, 2, "a nonempty discovery is reused by later openings");
});

test("concurrent cold skill loads share one request until discovery finishes", async () => {
  const h = harness();
  let calls = 0, resolve;
  h.context.api = () => { calls++; return new Promise((finish) => { resolve = finish; }); };
  const first = h.context.loadSkills(), second = h.context.loadSkills();
  assert.equal(calls, 1);
  assert.equal(first, second);
  h.elements.ovrwrap.open = true;
  const opening = h.elements.ovrwrap.ontoggle();
  assert.equal(opening, first);
  resolve({ json: async () => [{ name: "review" }] });
  await Promise.all([first, second, opening]);
  h.open();
  assert.deepEqual(h.names(), ["review"]);
});

test("overlapping catalog fetches cannot restore options from an old server", async () => {
  const h = harness();
  let resolve;
  h.context.api = () => new Promise((r) => { resolve = r; });
  const old = h.context.loadSkills();
  h.context.store.base = "https://b.test";
  h.context.api = async () => ({ json: async () => [{ name: "new-skill" }] });
  await h.context.loadSkills();
  resolve({ json: async () => [{ name: "old-skill" }] });
  await old;
  h.open();
  assert.deepEqual(h.names(), ["new-skill"]);
});

test("completion matches case-insensitive substrings and boundary fuzzy names", async () => {
  const h = harness();
  h.context.api = async () => ({ json: async () => [
    { name: "play_mahjong_discard", description: "Analyze mahjong discards" },
    { name: "review", description: "Review changes" },
    { name: "build", description: "Build software" },
  ] });
  await h.context.loadSkills();
  h.type("MAHJONG_DISCARD");
  assert.deepEqual(h.names(), ["play_mahjong_discard"]);
  h.type("Analyze mahjong");
  assert.deepEqual(h.names(), ["play_mahjong_discard"]);
  h.type("mhd");
  assert.deepEqual(h.names(), ["play_mahjong_discard"],
    "the boundary-anchored acronym matches like /skills completion");
  h.type("build");
  assert.deepEqual(h.names(), ["build"]);
  assert.deepEqual(h.selected(), [], "an exact search still needs an explicit choice");
  assert.equal(h.key("Enter").prevented, true);
  assert.deepEqual(h.selected(), ["build"], "Enter completes the first matching skill");
  assert.equal(h.requests.length, 0);
  h.type("SOFTwar");
  assert.deepEqual(h.names(), ["build"]);
  h.type(" ");
  assert.deepEqual(h.names(), ["play_mahjong_discard", "review", "build"]);
});

test("boundary-anchored fuzzy names rank audit_own_changes above auditor suffixes", async () => {
  const h = harness();
  h.context.api = async () => ({ json: async () => [
    { name: "agent-sandbox-auditor", description: "Python Agent 沙箱与安全隔离审查" },
    { name: "async-agent-auditor", description: "Python Agent 异步并发模式审查" },
    { name: "llm-call-auditor", description: "LLM 调用安全与成本优化审查" },
    { name: "audit_own_changes", description: "Audit your own changes" },
    { name: "unrelated-skill", description: "audit own changes" },
  ] });
  await h.context.loadSkills();
  h.type("audito");
  assert.deepEqual(h.names(), [
    "audit_own_changes",
    "agent-sandbox-auditor",
    "async-agent-auditor",
    "llm-call-auditor",
  ], "the boundary-anchored fuzzy hit outranks substrings inside auditor");
  assert.equal(h.key("Enter").prevented, true);
  assert.deepEqual(h.selected(), ["audit_own_changes"]);
  h.type("own");
  assert.deepEqual(h.names(), ["audit_own_changes", "unrelated-skill"],
    "description-only hits trail every name hit");
});

test("skill search ignores composing Enter flags after composition has ended", async () => {
  const h = harness();
  await h.context.loadSkills(); h.type("rev");
  for (const extra of [{ isComposing: true }, { keyCode: 229 }]) {
    assert.equal(h.key("Enter", extra).prevented, undefined);
    assert.deepEqual(h.selected(), []);
    assert.equal(h.elements.skillsearch.value, "rev");
    assert.equal(h.elements.skillmenu.hidden, false);
  }
  assert.equal(h.key("Enter").prevented, true);
  assert.deepEqual(h.selected(), ["review"]);
  assert.equal(h.requests.length, 0);
});

test("main model search cannot commit Enter during IME composition", () => {
  const search = {}, option = { click() { picks++; } };
  let picks = 0, closes = 0;
  const context = vm.createContext({
    document: { activeElement: search },
    $: (id) => id === "modelpicksearch" ? search : { querySelectorAll: () => nodeList([option]) },
    closeModelSheet() { closes++; },
  });
  vm.runInContext(between("function modelSheetKeydown(", "// Skills are pending next-turn choices"), context);
  for (const extra of [{ isComposing: true }, { keyCode: 229 }]) {
    context.modelSheetKeydown({ key: "Enter", ...extra, preventDefault() { assert.fail("composing Enter must not select a model"); } });
    assert.equal(picks, 0);
  }
  context.modelSheetKeydown({ key: "Enter", preventDefault() {} });
  assert.equal(picks, 1);
  context.modelSheetKeydown({ key: "Escape", preventDefault() {} });
  assert.equal(closes, 1);
});

test("search clear preserves the ordered set and reset returns to session default", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("build"); h.choose("review"); h.type("bui");
  const touch = { preventDefault() { this.prevented = true; } };
  h.elements.skillsearchclear.onpointerdown(touch);
  assert.equal(touch.prevented, true);
  h.elements.skillsearchclear.click();
  assert.equal(h.elements.skillsearch.value, "");
  assert.equal(h.elements.skillsearchclear.hidden, true);
  assert.deepEqual(h.names(), ["review", "build"]);
  assert.deepEqual(h.selected(), ["build", "review"]);
  assert.equal(h.elements.skillclear.attrs["aria-pressed"], "false");
  h.elements.skillclear.click();
  assert.deepEqual(h.selected(), []);
  assert.equal(h.elements.skillclear.attrs["aria-pressed"], "true");
  assert.equal(h.elements.skillvalue.textContent, "skill: session default");
  assert.equal(h.elements.skillmenu.hidden, true);
  assert.equal(h.context.document.activeElement, h.elements.oskill);
  assert.equal(h.storage.size, 0);
  await h.context.submit();
  assert.equal("skills" in h.requests[0].body, false);
});

test("reload restores the same ordered pending choices and independent search", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("build"); h.choose("review"); h.type("bui");
  const reloaded = harness({ storage: h.storage });
  await reloaded.context.loadSkills();
  assert.deepEqual(reloaded.selected(), ["build", "review"]);
  assert.equal(reloaded.elements.skillsearch.value, "bui");
  assert.equal(reloaded.elements.skillvalue.textContent, "skill: build + review");
  assert.match(reloaded.elements.oskill.attrs["aria-label"], /build \+ review.*next message/);
  await reloaded.context.submit();
  assert.deepEqual(reloaded.requests[0].body.skills, ["build", "review"]);
  const acceptedReload = harness({ storage: h.storage });
  await acceptedReload.context.loadSkills();
  assert.deepEqual(acceptedReload.selected(), [], "accepted selections stay consumed after reload");
  assert.equal(acceptedReload.elements.skillsearch.value, "bui");
});

test("stored skill drafts are isolated by server and session", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("review"); h.choose("build");
  h.switchTo("beta"); h.choose("build");
  h.context.store.base = "https://b.test"; h.switchTo("alpha");
  await h.context.loadSkills(); h.choose("review");
  for (const [base, sid, names] of [
    ["https://a.test", "alpha", ["review", "build"]],
    ["https://a.test", "beta", ["build"]], ["https://b.test", "alpha", ["review"]],
  ]) {
    const reloaded = harness({ storage: h.storage, base, sid });
    await reloaded.context.loadSkills();
    assert.deepEqual(reloaded.selected(), names);
  }
});

test("unavailable tab storage preserves normal in-memory picking and sending", async () => {
  const h = harness();
  const denied = () => { throw new Error("storage unavailable"); };
  h.context.sessionStorage = { getItem: denied, setItem: denied, removeItem: denied };
  await h.context.loadSkills(); h.choose("review"); h.choose("build");
  h.switchTo("beta"); h.switchTo("alpha");
  assert.deepEqual(h.selected(), ["review", "build"]);
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["review", "build"]);
  assert.deepEqual(h.selected(), []);
  const cleanupFailure = harness();
  await cleanupFailure.context.loadSkills(); cleanupFailure.choose("review");
  cleanupFailure.context.sessionStorage.removeItem = denied;
  await cleanupFailure.context.submit();
  assert.deepEqual(cleanupFailure.selected(), [], "a failed storage cleanup must not restore a consumed pick");
  cleanupFailure.switchTo("beta"); cleanupFailure.switchTo("alpha");
  assert.deepEqual(cleanupFailure.selected(), []);
});

test("the eight-skill wire limit disables extra picks without truncating selection", async () => {
  const catalog = Array.from({ length: 9 }, (_, i) => ({ name: "skill-" + (i + 1) }));
  const h = harness({ catalog });
  await h.context.loadSkills();
  for (const skill of catalog.slice(0, 8)) h.choose(skill.name);
  h.open();
  assert.equal(h.options()[8].disabled, true);
  assert.equal(h.options()[8].attrs["aria-disabled"], "true");
  assert.equal(h.options()[0].disabled, false, "selected skills can still be removed at the limit");
  h.choose("skill-9");
  assert.deepEqual(h.selected(), catalog.slice(0, 8).map((s) => s.name));
  assert.match(h.elements.skillhint.textContent, /Selection limit reached/);
  h.options()[0].click();
  assert.equal(h.options()[8].disabled, false);
  h.options()[8].click();
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, catalog.slice(1).map((s) => s.name));
});

test("catalog and restored names use canonical case and deduplicate in order", async () => {
  const storage = new Map([['serve.skill-draft:["https://a.test","alpha"]', JSON.stringify({
    names: ["review", "BUILD", "Review", null, "missing"], query: "",
  })]]);
  const h = harness({ storage, catalog: [{ name: " Review " }, { name: "review" }, { name: "BUILD" }] });
  await h.context.loadSkills(); h.open();
  assert.deepEqual(h.names(), ["Review", "BUILD"]);
  assert.deepEqual(h.selected(), ["Review", "BUILD"]);
  assert.deepEqual(h.options().map((o) => o.attrs["aria-selected"]), ["true", "true"]);
  h.choose("missing");
  assert.deepEqual(h.selected(), ["Review", "BUILD"]);
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["Review", "BUILD"]);
});

test("late acceptance from a previous server clears only that server's sent picks", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("review");
  let resolve;
  h.context.fetch = () => new Promise((r) => { resolve = r; });
  const pending = h.context.submit();
  h.context.store.base = "https://b.test";
  await h.context.loadSkills(); h.choose("build");
  resolve({ ok: true }); await pending;
  assert.deepEqual(h.selected(), ["build"]);
  assert.equal(h.elements.skillvalue.textContent, "skill: build");
  const oldServer = harness({ storage: h.storage });
  await oldServer.context.loadSkills();
  assert.deepEqual(oldServer.selected(), []);
  const newServer = harness({ storage: h.storage, base: "https://b.test" });
  await newServer.context.loadSkills();
  assert.deepEqual(newServer.selected(), ["build"]);
});

test("config hydration leaves pending picks ordered and defaults owned by the session", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("build"); h.choose("review");
  h.context.syncModelTrigger = () => {};
  vm.runInContext(between("function applySessionConfig(", "function resetPicks()"), h.context);
  h.context.applySessionConfig({ model: "", agent: "", reasoning_effort: "", default_skills: ["agent-team", "review"] }, { team: false });
  assert.deepEqual(h.selected(), ["build", "review"]);
  assert.match(h.elements.skillhint.textContent, /Session defaults still apply/);
  await h.context.submit();
  assert.deepEqual(h.requests[0].body, { prompt: "hello", confirm: true, skills: ["build", "review"] });
});

test("skill picker uses themed floating rows, 44px targets, and reduced motion", () => {
  const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
  const picker = html.slice(html.indexOf('<div id="skillpicker">'), html.indexOf('<div id="modelpick"'));
  assert.doesNotMatch(picker, /<datalist|\blist=/);
  for (const id of ["oskill", "skillsearch", "skillclear"]) {
    assert.match(css, new RegExp("#" + id + " \\{[^}]*min-height:44px;"));
  }
  assert.match(css, /#skillsearchclear \{[^}]*width:44px; height:44px;/);
  assert.match(css, /#skilloptions \.modelrow \{[^}]*min-height:44px;/);
  assert.match(css, /#skillvalue \{[^}]*text-overflow:ellipsis; white-space:nowrap;/);
  assert.match(css, /#skillmenu \{[^}]*position:absolute;[^}]*background:var\(--panel\);/);
  assert.match(css, /#skilloptions \{[^}]*min-height:0; overflow-y:auto; overscroll-behavior:contain; touch-action:pan-y;/);
  assert.match(css, /prefers-reduced-motion:reduce\) \{ #skillmenu \{ animation:none; \}/);
  assert.match(picker, /aria-label="reset skills to session default"/);
  assert.doesNotThrow(() => new vm.Script(script));
});
