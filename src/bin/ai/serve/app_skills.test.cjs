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
  const elements = { oskill: select(), omodel: select(), oagent: select(), oeffort: select(), ovrsummary: {} };
  const requests = [];
  const context = vm.createContext({
    $: (id) => elements[id], document: { createElement: () => ({}) },
    store: { base: "https://a.test" }, sessionId: "alpha",
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
  const body = between("  const body = { prompt, confirm: true };", '  $("prompt").value = "";');
  const accept = between('    const res = await fetch(store.base + "/sessions/"', "    turnStarted = true;");
  vm.runInContext(`async function submit() { const prompt = "hello", images = []; ${body}\n${accept}\nreturn body; }`, context);
  return {
    context, elements, requests,
    choose(name) { elements.oskill.value = name; context.saveSkillPick(); },
    switchTo(sid) { context.sessionId = sid; context.restoreSkillPick(); },
  };
}

test("native skill selector loads server names and describes next-message scope", async () => {
  const h = harness();
  await h.context.loadSkills();
  assert.match(html, /<select id="oskill" aria-label="skill for next message"/);
  assert.match(script, /\$\("oskill"\)\.onchange = \(\) => saveSkillPick\(\)/);
  assert.deepEqual(h.elements.oskill.options.map((o) => o.value), ["", "review", "build"]);
  assert.match(h.elements.oskill.options[0].textContent, /next message/);
  assert.equal(h.elements.oskill.options[1].title, "Review changes");
  h.choose("review");
  assert.match(h.elements.ovrsummary.textContent, /^skill review \(next\)/);
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
  assert.match(h.elements.oskill.options[0].textContent, /none available/);
  h.context.api = async () => { throw new Error("offline"); };
  await h.context.loadSkills();
  assert.match(h.elements.oskill.options[0].textContent, /unavailable/);
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
  assert.deepEqual(h.elements.oskill.options.map((o) => o.value), ["", "new-skill"]);
});
