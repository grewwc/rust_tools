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
function harness() {
  const document = { activeElement: null, createElement: () => node() };
  function node() {
    return {
      value: "", hidden: true, disabled: false, children: [], attrs: {},
      set textContent(value) { this.text = value; this.children = []; },
      get textContent() { return (this.text || "") + this.children.map((c) => c.textContent).join(""); },
      appendChild(child) { this.children.push(child); },
      setAttribute(key, value) { this.attrs[key] = value; },
      removeAttribute(key) { delete this.attrs[key]; },
      focus() { if (document.activeElement === this) return; document.activeElement = this; this.onfocus?.(); },
      blur() { document.activeElement = null; this.onblur?.(); },
      scrollIntoView() {},
    };
  }
  const elements = {
    oskill: node(), skillmenu: node(), skillclear: node(), skillhint: node(), ovrwrap: {}, prompt: node(),
    omodel: select(), oagent: select(), oeffort: select(), ovrsummary: {},
  };
  const requests = [];
  const context = vm.createContext({
    $: (id) => elements[id], document,
    store: { base: "https://a.test" }, sessionId: "alpha",
    sendBusy: false, pendingImages: [],
    headers: () => ({ Authorization: "Bearer test" }),
    api: async (url) => {
      assert.equal(url, "/skills");
      return { json: async () => [
        { name: "review", description: "Review changes" },
        { name: "build", description: "Build software" },
      ] };
    },
    fetch: async (url, opts) => { requests.push({ url, ...opts, body: JSON.parse(opts.body) }); return { ok: true }; },
  });
  vm.runInContext(between("function syncOvrSummary()", "// The composer grows"), context);
  vm.runInContext(between("function fillSelect(", "async function loadInfo()"), context);
  vm.runInContext(between('  $("oskill").onfocus', '  $("prompt").addEventListener("compositionstart"'), context);
  vm.runInContext(between("async function send()", "  const rewindFloor =") + "\n}", context);
  const body = between("  const body = { prompt, confirm: true };", '  setPromptValue("", true);');
  const accept = between('    const res = await fetch(store.base + "/sessions/"', "    turnStarted = true;");
  vm.runInContext(`async function submit() { if (!validateSkillInput()) return; const prompt = "hello", images = []; ${body}\n${accept}\nreturn body; }`, context);
  return {
    context, elements, requests,
    choose(name) { context.chooseSkill(name); },
    type(text, composing = false) { elements.oskill.focus(); elements.oskill.value = text; elements.oskill.oninput({ isComposing: composing }); },
    options() { return elements.skillmenu.children.filter((n) => n.attrs.role === "option"); },
    key(key, extra = {}) { const event = { key, preventDefault() { this.prevented = true; }, ...extra }; elements.oskill.onkeydown(event); return event; },
    switchTo(sid) { context.sessionId = sid; context.restoreSkillPick(); },
  };
}

test("skill combobox loads server names and describes next-message scope", async () => {
  const h = harness();
  await h.context.loadSkills();
  assert.match(html, /<input id="oskill" type="text" role="combobox"/);
  h.elements.oskill.focus();
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["review", "build", "Default"]);
  assert.match(h.elements.oskill.placeholder, /next message/);
  assert.equal(h.options()[0].children[1].textContent, "Review changes");
  assert.equal(h.elements.ovrsummary.textContent, "model default · agent default · effort default · skill default");
  h.choose("review");
  assert.equal(h.elements.ovrsummary.textContent, "model default · agent default · effort default · skill review (next)");
});

test("comma-separated skill names send every selected skill", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("review");
  h.choose("build");
  assert.equal(h.elements.oskill.value, "review, build");
  assert.equal(h.elements.ovrsummary.textContent, "model default · agent default · effort default · skill review, build (next)");
  // Already picked names drop out of the suggestion list; the last segment
  // still drives the query.
  h.elements.oskill.focus();
  h.type("review, bui");
  assert.equal(h.elements.oskill.attrs["aria-invalid"], "true");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["build", "Default"]);
  // Picking from the menu replaces only the partial tail, not the whole field.
  h.choose("build");
  assert.equal(h.elements.oskill.value, "review, build");
  h.type("review, build");
  assert.equal(h.elements.oskill.attrs["aria-invalid"], "false");
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["review", "build"]);
  assert.equal(h.elements.oskill.value, "");
});

test("override controls and summary keep model, agent, effort, skill order", async () => {
  const controls = ["omodel", "oagent", "oeffort", "oskill"];
  const positions = controls.map((id) => html.indexOf(`id="${id}"`));
  assert.ok(positions.every((pos, i) => pos >= 0 && (!i || pos > positions[i - 1])));
  assert.match(html, /<summary id="ovrsummary">model · agent · effort · skill<\/summary>/);
  const h = harness();
  await h.context.loadSkills();
  for (const [id, value] of [["omodel", "test-model"], ["oagent", "build"], ["oeffort", "high"]]) {
    h.elements[id].appendChild({ value, textContent: value });
    h.elements[id].value = value;
  }
  h.type("unknown");
  assert.equal(h.elements.ovrsummary.textContent, "model test-model · agent build · effort high · skill choose a match");
  h.choose("review");
  assert.equal(h.elements.ovrsummary.textContent, "model test-model · agent build · effort high · skill review (next)");
});

test("typing filters names and descriptions, touch selection and clear keep next-turn semantics", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.type("SOFTware");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["build", "Default"]);
  assert.equal(h.elements.oskill.attrs["aria-expanded"], "true");
  const touch = { preventDefault() { this.prevented = true; } };
  h.options()[0].onpointerdown(touch);
  assert.equal(touch.prevented, true);
  h.options()[0].onclick();
  assert.equal(h.elements.oskill.value, "build");
  assert.equal(h.elements.skillmenu.hidden, true);
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["build"]);
  h.type(" ReVieW ");
  assert.equal(h.elements.oskill.attrs["aria-invalid"], "false");
  await h.context.submit();
  assert.deepEqual(h.requests[1].body.skills, ["review"]);
  h.type("build"); h.elements.skillclear.onclick();
  assert.equal(h.elements.oskill.value, "");
  await h.context.submit();
  assert.equal("skills" in h.requests[2].body, false);
});

test("deleting skill text to empty closes suggestions and clears the pending choice", async () => {
  for (const empty of ["", " \t "]) {
    const h = harness();
    await h.context.loadSkills();
    h.type("review");
    h.key("ArrowDown");
    assert.equal(h.elements.skillmenu.hidden, false);
    assert.ok(h.elements.oskill.attrs["aria-activedescendant"]);
    h.type(empty);
    assert.equal(h.context.document.activeElement, h.elements.oskill);
    assert.equal(h.elements.skillmenu.hidden, true);
    assert.equal(h.elements.oskill.attrs["aria-expanded"], "false");
    assert.equal(h.elements.oskill.attrs["aria-activedescendant"], undefined);
    assert.equal(h.elements.oskill.attrs["aria-invalid"], "false");
    assert.equal(h.elements.skillhint.hidden, true);
    assert.match(h.elements.ovrsummary.textContent, / · skill default$/);
    await h.context.submit();
    assert.equal("skills" in h.requests[0].body, false);
    h.switchTo("beta"); h.switchTo("alpha");
    assert.equal(h.elements.oskill.value, "");
    h.type("bui");
    assert.equal(h.elements.skillmenu.hidden, false);
    assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["build", "Default"]);
    h.key("ArrowDown"); h.key("Enter");
    assert.equal(h.elements.oskill.value, "build");
  }
});

test("composition ending with empty skill text does not reopen suggestions", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.type("review");
  h.elements.oskill.oncompositionstart();
  h.type("", true);
  assert.equal(h.elements.skillmenu.hidden, true);
  h.elements.oskill.oncompositionend();
  assert.equal(h.elements.skillmenu.hidden, true);
  h.type(""); // WebView may emit a final input event after compositionend.
  assert.equal(h.elements.skillmenu.hidden, true);
  assert.equal(h.elements.oskill.attrs["aria-expanded"], "false");
  await h.context.submit();
  assert.equal("skills" in h.requests[0].body, false);
  h.elements.oskill.onclick(); // Explicitly browsing the catalog still works.
  assert.equal(h.elements.skillmenu.hidden, false);
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["review", "build", "Default"]);
});

test("keyboard navigation selects a suggestion without sending and Escape/blur close the menu", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.type("rev");
  assert.equal(h.key("ArrowDown").prevented, true);
  assert.equal(h.elements.oskill.attrs["aria-activedescendant"], h.options()[0].id);
  assert.equal(h.options()[0].attrs["aria-selected"], "true");
  assert.equal(h.key("Enter").prevented, true);
  assert.equal(h.elements.oskill.value, "review");
  assert.equal(h.requests.length, 0);
  h.elements.oskill.onclick();
  h.key("Escape");
  assert.equal(h.elements.skillmenu.hidden, true);
  assert.equal(h.elements.oskill.value, "review");
  h.key("ArrowUp");
  assert.equal(h.options().at(-1).attrs["aria-selected"], "true");
  h.key("Enter");
  assert.equal(h.elements.oskill.value, "");
  h.elements.oskill.onclick(); h.elements.oskill.blur();
  assert.equal(h.elements.skillmenu.hidden, true);
});

test("unmatched input is a session draft and cannot silently send the default or lose images", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.type("not-a-skill");
  assert.match(h.elements.skillmenu.textContent, /No matching skills/);
  assert.equal(h.elements.oskill.attrs["aria-invalid"], "true");
  h.switchTo("beta"); h.type("bui"); h.switchTo("alpha");
  assert.equal(h.elements.oskill.value, "not-a-skill");
  h.elements.prompt.value = "hello";
  h.context.pendingImages = [{ filename: "photo.png" }];
  await h.context.send();
  assert.equal(h.elements.ovrwrap.open, true);
  assert.equal(h.elements.prompt.value, "hello");
  assert.equal(h.context.pendingImages.length, 1);
  assert.equal(h.requests.length, 0);
  assert.equal(h.context.sendBusy, false);
});

test("actual turn request sends skills only when selected, clearing on HTTP success", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("review");
  await h.context.submit();
  assert.deepEqual(h.requests[0].body, { prompt: "hello", confirm: true, skills: ["review"] });
  assert.equal(h.requests[0].url, "https://a.test/sessions/alpha/turns/stream");
  assert.equal(h.elements.oskill.value, "");
  await h.context.submit();
  assert.match(h.elements.ovrsummary.textContent, /^model default · agent default · effort default · skill default$/);
  assert.equal("skills" in h.requests[1].body, false);
});

test("HTTP rejection and transport failure retain the pending selection", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("build");
  h.context.fetch = async () => ({ ok: false, status: 409, json: async () => ({ error: "running" }) });
  await assert.rejects(h.context.submit(), /409 running/);
  assert.equal(h.elements.oskill.value, "build");
  h.context.fetch = async () => { throw new Error("offline"); };
  await assert.rejects(h.context.submit(), /offline/);
  assert.equal(h.elements.oskill.value, "build");
});

test("late acceptance cannot clear another session or a reselected skill", async () => {
  const h = harness();
  await h.context.loadSkills();
  let resolve;
  h.context.fetch = () => new Promise((r) => { resolve = r; });
  h.choose("review");
  const pending = h.context.submit();
  h.switchTo("beta");
  assert.equal(h.elements.oskill.value, "");
  h.choose("build");
  resolve({ ok: true });
  await pending;
  assert.equal(h.elements.oskill.value, "build");
  h.switchTo("alpha");
  assert.equal(h.elements.oskill.value, "");
  h.choose("review");
  const again = h.context.submit();
  h.choose(""); h.choose("review");
  resolve({ ok: true });
  await again;
  assert.equal(h.elements.oskill.value, "review");
});

test("late HTTP acceptance preserves a newer unselected search draft", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("review");
  let resolve;
  h.context.fetch = () => new Promise((r) => { resolve = r; });
  const pending = h.context.submit();
  h.type("bui");
  resolve({ ok: true }); await pending;
  assert.equal(h.elements.oskill.value, "bui");
  assert.equal(h.context.validateSkillInput(), false);
});

test("Chinese composition cannot pick, clear, or submit until committed", async () => {
  const h = harness();
  h.context.api = async () => ({ json: async () => [{ name: "mahjong", description: "麻将分析" }] });
  await h.context.loadSkills();
  h.elements.oskill.focus(); h.elements.oskill.oncompositionstart();
  h.type("ma", true);
  assert.equal(h.elements.skillmenu.hidden, true);
  for (const extra of [{}, { isComposing: true }, { keyCode: 229 }]) {
    assert.equal(h.key("Enter", extra).prevented, undefined);
  }
  h.elements.skillclear.onclick();
  assert.equal(h.elements.oskill.value, "ma");
  assert.equal(h.context.validateSkillInput(), false);
  h.elements.oskill.value = "麻将";
  h.elements.oskill.oncompositionend();
  assert.equal(h.elements.oskill.value, "麻将");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["mahjong", "Default"]);
  assert.equal(h.key("Enter", { keyCode: 229 }).prevented, undefined);
  h.options()[0].onclick(); await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["mahjong"]);
});

test("skill names and descriptions render as text, never catalog-supplied markup", async () => {
  const h = harness();
  const name = '<img src=x onerror="alert(1)">';
  h.context.api = async () => ({ json: async () => [{ name, description: "<script>alert(2)</script>" }] });
  await h.context.loadSkills(); h.elements.oskill.focus();
  const option = h.options()[0];
  assert.equal(option.children[0].textContent, name);
  assert.equal(option.children[1].textContent, "<script>alert(2)</script>");
  assert.equal(option.children[0].children.length, 0);
  assert.doesNotMatch(between("function renderSkillMenu()", "function skillKeydown("), /innerHTML/);
});

test("pending choices survive session switching and move into the first created session", async () => {
  const h = harness();
  await h.context.loadSkills();
  h.choose("review"); h.switchTo("beta"); h.choose("build"); h.switchTo("alpha");
  assert.equal(h.elements.oskill.value, "review");
  h.switchTo(""); h.choose("build");
  h.context.sessionId = "new";
  h.context.carrySkillPick("", "new");
  assert.equal(h.elements.oskill.value, "build");
  await h.context.submit();
  assert.deepEqual(h.requests[0].body.skills, ["build"]);
  h.switchTo("");
  assert.equal(h.elements.oskill.value, "");
  assert.match(script, /carrySkillPick\(draftSession, sessionId\)/);
});

test("catalog removes obsolete choices and empty or failed catalogs preserve normal send", async () => {
  const h = harness();
  await h.context.loadSkills(); h.choose("review");
  h.context.api = async () => ({ json: async () => [{ name: "build" }] });
  await h.context.loadSkills();
  assert.equal(h.elements.oskill.value, "");
  await h.context.submit();
  assert.equal("skills" in h.requests[0].body, false);
  h.context.api = async () => ({ json: async () => [] });
  await h.context.loadSkills();
  assert.equal(h.elements.oskill.disabled, true);
  assert.match(h.elements.oskill.placeholder, /none available/);
  h.context.api = async () => { throw new Error("offline"); };
  await h.context.loadSkills();
  assert.match(h.elements.oskill.placeholder, /unavailable/);
  await h.context.submit();
  assert.equal("skills" in h.requests[1].body, false);
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
  h.elements.oskill.focus();
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["new-skill", "Default"]);
});

test("fuzzy search matches multi-token, reordered, and subsequence queries", async () => {
  const h = harness();
  h.context.api = async () => ({ json: async () => [
    { name: "play_mahjong_discard", description: "Analyze mahjong discards" },
    { name: "review", description: "Review changes" },
    { name: "build", description: "Build software" },
  ] });
  await h.context.loadSkills();
  // Multi-token query spans name and description, order-independent.
  h.type("mahjong discard");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["play_mahjong_discard", "Default"]);
  h.type("discard mahjong");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["play_mahjong_discard", "Default"]);
  // Subsequence tolerance: "mhd" appears in order inside "mahjong_discard".
  h.type("mhd");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["play_mahjong_discard", "Default"]);
  // Name substring outranks description substring for the same query.
  h.type("build");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["build", "Default"]);
  h.type("softwar");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["build", "Default"]);
  // Every token must match somewhere: one bad token kills the candidate.
  h.type("review mahjong");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["Default"]);
  h.type("zzz");
  assert.match(h.elements.skillmenu.textContent, /No matching skills/);
  // A query that only fuzzy-matches still shows the intended skill.
  h.type("pl_mj_dscrd");
  assert.deepEqual(h.options().map((o) => o.children[0].textContent), ["play_mahjong_discard", "Default"]);
});
