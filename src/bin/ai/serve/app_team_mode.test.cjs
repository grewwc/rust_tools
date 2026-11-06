// Run with: node --test src/bin/ai/serve/app_team_mode.test.cjs
// Execute the marked page functions, including the actual send path and API,
// against a small DOM and controllable session-config responses.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const plain = (value) => JSON.parse(JSON.stringify(value));
function marked(name) {
  const start = "// BEGIN " + name, end = "// END " + name;
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing marked page functions: " + name);
  return script.slice(a + start.length, b);
}
function existingFunction(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing page function: " + start);
  return script.slice(a, b);
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const NORMAL = { model: "", agent: "", reasoning_effort: "", default_skills: ["other-skill"], subagent_backend: "", subagent_model: "", output: "Sticky agent-team: active" };
const KERNEL = { ...NORMAL, default_skills: ["other-skill", "AGENT-TEAM"], output: "Sticky agent-team: off" };
const TRAECLI = { ...KERNEL, subagent_backend: "traecli", subagent_model: "worker-model" };
const response = (cfg) => ({ ok: true, json: async () => plain(cfg) });
const turnResponse = (read = async () => ({ done: false, value: Buffer.from("event: done\n\n") })) => ({ ok: true, body: { getReader: () => ({ read }) } });
const settle = () => new Promise((resolve) => setImmediate(resolve));

function harness({ sid = "alpha", phone = true, cfg = NORMAL, configPost, configGet, createSession, skills = [], turnPost } = {}) {
  const document = { activeElement: null, querySelectorAll: () => [] };
  function node(tag = "div") {
    let text = "";
    const classes = new Set();
    const el = {
      tagName: tag, children: [], parentNode: null, attrs: {}, dataset: {}, style: {}, value: "", title: "",
      hidden: false, disabled: false, tabIndex: tag === "button" || tag === "input" ? 0 : -1,
      classList: { add: (...names) => names.forEach((n) => classes.add(n)), remove: (...names) => names.forEach((n) => classes.delete(n)), toggle: (name, on) => on ? classes.add(name) : classes.delete(name) },
      set textContent(value) { text = String(value); el.children = []; },
      get textContent() { return text; },
      setAttribute(key, value) { el.attrs[key] = String(value); },
      getAttribute(key) { return el.attrs[key] ?? null; },
      appendChild(child) { child.parentNode = el; el.children.push(child); return child; },
      remove() { if (el.parentNode) el.parentNode.children = el.parentNode.children.filter((n) => n !== el); },
      focus() { if (!el.disabled) document.activeElement = el; },
      click() { if (!el.disabled) return el.onclick?.(); },
      closest(selector) {
        for (let cur = el; cur; cur = cur.parentNode) {
          if (selector === "[hidden]" && cur.hidden) return cur;
          if (selector === '[role="radiogroup"]' && cur.attrs.role === "radiogroup") return cur;
        }
        return null;
      },
      querySelectorAll(selector) {
        const found = [];
        const walk = (parent) => parent.children.forEach((child) => {
          if ((selector === "button, input" && ["button", "input"].includes(child.tagName)) || (selector === '[role="radio"]' && child.attrs.role === "radio")) found.push(child);
          walk(child);
        });
        walk(el);
        return found;
      },
    };
    return el;
  }
  document.createElement = node;
  const elements = {};
  for (const match of html.matchAll(/<(div|section|button|span|input|p|label)[^>]*id="(team[^"]*)"[^>]*>/g)) {
    const el = elements[match[2]] = node(match[1]);
    el.hidden = /\shidden(?:\s|>)/.test(match[0]);
    const role = /role="([^"]+)"/.exec(match[0]);
    if (role) el.attrs.role = role[1];
  }
  const attach = (id, ...children) => children.forEach((child) => elements[id].appendChild(elements[child]));
  attach("teampick", "teampickscrim", "teampickpanel");
  attach("teampickpanel", "teamclose", "teambody", "teamactions");
  attach("teambody", "teammodes", "teamadvanced", "teamerror");
  attach("teammodes", "teamnormal", "teamcollab");
  attach("teamadvanced", "teambackends", "teamworkerwrap");
  attach("teambackends", "teamkernel", "teamtraecli");
  attach("teamworkerwrap", "teamworker", "teamworkerclear");
  attach("teamactions", "teamcancel", "teamapply");
  attach("teamcontrols", "teamtrigger");
  attach("teamtrigger", "teamlabel", "teamstatus");
  for (const [id, value] of [["teamnormal", "normal"], ["teamcollab", "team"]]) elements[id].dataset.mode = value;
  for (const [id, value] of [["teamkernel", "kernel"], ["teamtraecli", "traecli"]]) elements[id].dataset.backend = value;
  for (const [id, values] of [["omodel", ["", "main-model"]], ["oagent", ["", "build"]], ["oeffort", ["", "high"]]]) {
    const el = elements[id] = node("select");
    el.options = values.map((value) => ({ value }));
  }
  elements.prompt = node("textarea");
  elements.prompt.value = "一起完成这个任务";
  elements.msgs = node();
  const requests = [], notes = [], messages = [], consumed = [], stored = new Map(), listeners = new Map();
  const configs = new Map([[sid || "fresh", plain(cfg)]]);
  const media = { matches: phone, addEventListener: (_, fn) => { media.change = fn; } };
  const history = { pushes: 0, backs: 0, pushState() { this.pushes++; }, back() { this.backs++; queueMicrotask(() => listeners.get("popstate")?.()); } };
  const context = vm.createContext({
    $: (id) => elements[id], document, history, TextDecoder,
    window: { matchMedia: () => media, addEventListener: (name, fn) => listeners.set(name, fn) },
    isPhoneLayout: () => media.matches, PHONE_LAYOUT_QUERY: "(max-width:700px)",
    store: { base: "https://a.test" }, sessionId: sid, pendingNew: null,
    localStorage: { getItem: (key) => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key) },
    headers: () => ({ "Content-Type": "application/json" }),
    note: (text) => notes.push(text), holdLayout: (_, apply) => apply(),
    syncModelTrigger() {}, syncOvrSummary() {},
    persistDraft() { stored.set("draft." + context.sessionId, elements.prompt.value); },
    restoreDraft() { elements.prompt.value = stored.get("draft." + context.sessionId) || ""; },
    restoreSkillPick() {}, carrySkillPick() {}, resetQuestionNavigator() {},
    sendBusy: false, promptComposing: false, pendingImages: [],
    turns: new Map(), MAX_LIVE_TURNS: 4, historySession: sid, canonicalHistoryTotal: 0,
    pendingUserInputs: new Map(), questionCache: null, runEpoch: 0, turnRunning: false,
    confirmPollSeq: 0, lastHistoryEndsUnanswered: false, LIVE_STALL_MS: 60000,
    validateSkillInput: () => true,
    captureSkillPick: () => ({ skills }), consumeSkillPick: (pick) => consumed.push(pick),
    addMsg: (role, text) => { messages.push({ role, text }); return node(); },
    addLiveUserBubble: () => { const parent = node(), bubble = node(); parent.appendChild(bubble); return bubble; },
    addMsgMd: (_, parent) => parent.appendChild(node()),
    createLiveActivity: () => ({ label: () => "Connecting", setPhase() {}, finish() {} }),
    setPromptValue: (value) => { elements.prompt.value = value; }, clearDraft() {},
    syncStopButton() {}, scrollToLatest() {}, stopStaleFollow() {},
    shouldFollowOutput: () => false, followOutput() {}, decorateAndLoad() {},
    formatElapsed: () => "0s", replyPaintDelay: () => 0, renderMarkdown: (s) => s,
    parseBlock: () => ({ kind: "done", payload: "" }),
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    loadInfo: async () => {}, loadSessions: async () => {},
    fetch: async (url, opts = {}) => {
      const request = { url, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : undefined };
      requests.push(request);
      const pathname = new URL(url).pathname;
      if (pathname === "/sessions") return createSession ? createSession(request) : response({ id: "fresh" });
      if (pathname.endsWith("/config")) {
        const id = pathname.split("/")[2], current = configs.get(id) || plain(NORMAL);
        if (request.method === "GET") return configGet ? configGet(request) : response(current);
        if (request.body.subagents && configPost) return configPost(request);
        let next = { ...current, ...request.body };
        if (request.body.subagents) {
          const sub = request.body.subagents;
          next = sub.action === "reset" ? { ...current, default_skills: NORMAL.default_skills, subagent_backend: "", subagent_model: "" }
            : { ...current, default_skills: KERNEL.default_skills, subagent_backend: sub.backend === "kernel" ? "" : "traecli", subagent_model: sub.model || current.subagent_model };
        }
        configs.set(id, next);
        return response(next);
      }
      if (pathname.endsWith("/turns/stream")) return turnPost ? turnPost(request) : turnResponse();
      throw new Error("unexpected request: " + pathname);
    },
  });
  vm.runInContext(existingFunction("async function api(", "const esc ="), context);
  vm.runInContext(marked("mobile team mode"), context);
  vm.runInContext(marked("session config hydration"), context);
  vm.runInContext(existingFunction("async function persistPicks()", "// A changed pick belongs"), context);
  vm.runInContext(marked("session switch"), context);
  vm.runInContext(marked("composer cd command"), context);
  vm.runInContext(marked("turn session preparation"), context);
  vm.runInContext(marked("composer send"), context);
  context.wireTeamMode();
  return {
    context, elements, requests, notes, messages, consumed, configs, media, history, listeners, stored,
    state: (id = context.sessionId, base = context.store.base) => context.teamPickState(id, base),
    teamPosts: () => requests.filter((r) => r.method === "POST" && r.body?.subagents),
    turns: () => requests.filter((r) => r.url.endsWith("/turns/stream")),
    async choose(mode, backend = "kernel", model = "") {
      context.openTeamSheet();
      context.changeTeamDraft("mode", mode);
      context.changeTeamDraft("backend", backend);
      elements.teamworker.value = model;
      context.teamWorkerChanged();
      return context.applyTeamDraft();
    },
  };
}

test("normal, Kernel and TraeCLI apply only the supported subagent payloads", async () => {
  for (const [mode, backend, model, expected] of [
    ["normal", "traecli", "ignored", { subagents: { action: "reset" } }],
    ["team", "kernel", "ignored", { subagents: { action: "use", backend: "kernel" } }],
    ["team", "traecli", " worker-model ", { subagents: { action: "use", backend: "traecli", model: "worker-model" } }],
    ["team", "traecli", "  ", { subagents: { action: "use", backend: "traecli" } }],
  ]) {
    const h = harness();
    await h.context.loadSessionConfig();
    assert.equal(await h.choose(mode, backend, model), true);
    assert.deepEqual(h.teamPosts().map((r) => r.body), [expected]);
    assert.equal(h.state().pending, false);
  }
});

test("GET config restores sticky team state independently of backend and output text", async () => {
  for (const [cfg, mode, backend] of [[NORMAL, "normal", "kernel"], [KERNEL, "team", "kernel"], [{ ...KERNEL, subagent_backend: "kernel" }, "team", "kernel"], [TRAECLI, "team", "traecli"], [{ ...TRAECLI, default_skills: [] }, "normal", "traecli"]]) {
    const h = harness({ cfg: { ...cfg, model: "main-model", agent: "build", reasoning_effort: "high" } });
    await h.context.loadSessionConfig();
    assert.equal(h.state().choice.mode, mode);
    assert.equal(h.state().choice.backend, backend);
    assert.equal(h.state().choice.model, cfg.subagent_model);
    assert.equal(h.elements.omodel.value, "main-model");
    assert.equal(h.elements.oagent.value, "build");
    assert.equal(h.elements.oeffort.value, "high");
    assert.equal(h.teamPosts().length, 0);
  }
});

test("phone-only config refresh updates team state without replacing main model, agent or effort picks", async () => {
  const h = harness({ cfg: TRAECLI });
  h.elements.omodel.value = "main-model";
  h.elements.oagent.value = "build";
  h.elements.oeffort.value = "high";
  await h.context.loadSessionConfig({ teamOnly: true });
  assert.equal(h.state().choice.backend, "traecli");
  assert.equal(h.elements.omodel.value, "main-model");
  assert.equal(h.elements.oagent.value, "build");
  assert.equal(h.elements.oeffort.value, "high");
});

for (const order of ["team-first", "full-first"]) test("switching sessions restores full config despite the open team sheet (" + order + ")", async () => {
  const alpha = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const beta = { ...KERNEL, model: "", agent: "", reasoning_effort: "" };
  const fullRead = deferred(), teamRead = deferred(), events = [];
  let betaReads = 0;
  const h = harness({ cfg: alpha, configGet: (request) => {
    if (request.url.endsWith("/alpha/config")) return response(alpha);
    assert.equal(request.url, "https://a.test/sessions/beta/config");
    if (++betaReads === 1) { events.push("full requested"); return fullRead.promise; }
    if (betaReads === 2) { events.push("team requested"); return teamRead.promise; }
    return response(beta);
  } });
  const picks = () => [h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value];
  h.configs.set("beta", plain(beta));
  await h.context.loadSessionConfig();
  assert.deepEqual(picks(), ["main-model", "build", "high"]);
  h.context.switchSession("beta");
  const restoring = h.context.loadSessionConfig().then(() => events.push("full completed"));
  await settle();
  assert.equal(betaReads, 1);
  h.context.openTeamSheet();
  await settle();
  assert.equal(betaReads, 2);
  assert.deepEqual(events, ["full requested", "team requested"]);
  if (order === "team-first") {
    teamRead.resolve(response({ ...beta, model: "main-model", agent: "build", reasoning_effort: "high" }));
    await settle();
    events.push("team completed");
    assert.deepEqual(picks(), ["main-model", "build", "high"]);
    assert.equal(h.state().choice.backend, "kernel");
    fullRead.resolve(response(beta));
    await restoring;
  } else {
    fullRead.resolve(response(beta));
    await restoring;
    assert.deepEqual(picks(), ["", "", ""]);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(beta)));
    teamRead.resolve(response({ ...beta, model: "main-model", agent: "build", reasoning_effort: "high" }));
    await settle();
    events.push("team completed");
  }
  assert.deepEqual(events, ["full requested", "team requested", ...(order === "team-first" ? ["team completed", "full completed"] : ["full completed", "team completed"])]);
  assert.deepEqual(picks(), ["", "", ""]);
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(beta)));
  h.context.closeTeamSheet();
  h.elements.prompt.value = "beta ordinary message";
  await h.context.send();
  await settle();
  assert.equal(h.turns().length, 1);
  assert.equal(h.turns()[0].url, "https://a.test/sessions/beta/turns/stream");
  assert.deepEqual(h.turns()[0].body, { prompt: "beta ordinary message", confirm: true });
  assert.equal(h.teamPosts().length, 0);
});

test("a failed team-only read cannot cancel a pending full restore", async () => {
  const alpha = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const beta = { ...KERNEL, model: "beta-model", agent: "beta-agent", reasoning_effort: "low" };
  const fullRead = deferred();
  let betaReads = 0;
  const h = harness({ cfg: alpha, configGet: (request) => {
    if (request.url.endsWith("/alpha/config")) return response(alpha);
    if (++betaReads === 1) return fullRead.promise;
    if (betaReads === 2) return { ok: false, status: 503, json: async () => ({ error: "team read unavailable" }) };
    return response(beta);
  } });
  for (const [id, value] of [["omodel", "beta-model"], ["oagent", "beta-agent"], ["oeffort", "low"]]) h.elements[id].options.push({ value });
  await h.context.loadSessionConfig();
  h.context.switchSession("beta");
  const restoring = h.context.loadSessionConfig();
  h.context.openTeamSheet();
  await settle();
  assert.equal(betaReads, 2);
  assert.ok(h.notes.some((text) => text.includes("team read unavailable")));
  fullRead.resolve(response(beta));
  await restoring;
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["beta-model", "beta-agent", "low"]);
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(beta)));
  h.context.closeTeamSheet();
  h.elements.prompt.value = "beta configured message";
  await h.context.send();
  await settle();
  assert.deepEqual(h.turns()[0].body, { prompt: "beta configured message", confirm: true, model: "beta-model", agent: "beta-agent", reasoning_effort: "low" });
});

test("late full and team reads from the previous session cannot overwrite the current session or turn", async () => {
  const alpha = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const fullRead = deferred(), teamRead = deferred();
  let alphaReads = 0;
  const h = harness({ configGet: (request) => {
    if (request.url.endsWith("/beta/config")) return response(NORMAL);
    return ++alphaReads === 1 ? fullRead.promise : teamRead.promise;
  } });
  const reading = h.context.loadSessionConfig();
  h.context.openTeamSheet();
  await settle();
  assert.equal(alphaReads, 2);
  h.context.switchSession("beta");
  await h.context.loadSessionConfig();
  teamRead.resolve(response(alpha));
  fullRead.resolve(response(alpha));
  await reading;
  await settle();
  assert.equal(h.context.sessionId, "beta");
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["", "", ""]);
  assert.equal(h.state().choice.mode, "normal");
  h.elements.prompt.value = "current beta message";
  await h.context.send();
  await settle();
  assert.equal(h.turns()[0].url, "https://a.test/sessions/beta/turns/stream");
  assert.deepEqual(h.turns()[0].body, { prompt: "current beta message", confirm: true });
});

test("an older full read cannot roll back the team state restored after /reset", async () => {
  const before = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const after = { ...NORMAL, model: "main-model", agent: "build", reasoning_effort: "high" };
  const fullRead = deferred();
  let reads = 0;
  const h = harness({ cfg: before, configGet: () => ++reads === 1 ? fullRead.promise : response(after), turnPost: () => turnResponse() });
  h.context.applySessionConfig(before);
  const reading = h.context.loadSessionConfig();
  h.elements.prompt.value = "/reset";
  await h.context.send();
  await settle();
  assert.equal(reads, 2);
  assert.equal(h.turns()[0].body.prompt, "/reset");
  assert.equal(h.state().choice.mode, "normal");
  fullRead.resolve(response(before));
  await reading;
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["main-model", "build", "high"]);
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  assert.equal(h.elements.teamlabel.textContent, "普通对话");
});

test("a superseded full read cannot overwrite a newer full restore", async () => {
  const oldRead = deferred(), newRead = deferred();
  let reads = 0;
  const h = harness({ configGet: () => ++reads === 1 ? oldRead.promise : newRead.promise });
  const first = h.context.loadSessionConfig(), second = h.context.loadSessionConfig();
  newRead.resolve(response(NORMAL));
  await second;
  oldRead.resolve(response({ ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" }));
  await first;
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["", "", ""]);
  assert.equal(h.state().choice.mode, "normal");
});

test("first send creates a session and waits for config success before the actual turn POST", async () => {
  const save = deferred(), h = harness({ sid: "", configPost: () => save.promise, skills: ["review"] });
  h.elements.omodel.value = "main-model";
  await h.choose("team", "traecli", "worker-model");
  assert.equal(h.teamPosts().length, 0);
  assert.equal(h.elements.teamstatus.textContent, "待保存");
  const sending = h.context.send();
  await settle();
  assert.equal(h.context.sessionId, "fresh");
  assert.equal(h.context.pendingNew, "fresh");
  assert.equal(h.teamPosts().length, 1);
  assert.equal(h.state().pending, true);
  assert.equal(h.state().confirmed, null);
  assert.equal(h.elements.teamstatus.textContent, "保存中");
  assert.equal(h.turns().length, 0);
  assert.equal(h.elements.prompt.value, "一起完成这个任务");
  h.configs.set("fresh", { ...TRAECLI, model: "main-model" });
  save.resolve(response({ ...TRAECLI, model: "main-model" }));
  await sending;
  assert.equal(h.turns().length, 1);
  assert.equal(h.turns()[0].url, "https://a.test/sessions/fresh/turns/stream");
  assert.deepEqual(h.turns()[0].body, { prompt: "一起完成这个任务", confirm: true, skills: ["review"], model: "main-model" });
  assert.equal(h.state().pending, false);
  assert.equal(h.consumed.length, 1);
});

test("failed config blocks the first turn, retains selection, text, images and skills, and can retry", async () => {
  let fail = true;
  const h = harness({ sid: "", skills: ["review"], configPost: () => fail ? { ok: false, status: 503, json: async () => ({ error: "temporarily offline" }) } : response(KERNEL) });
  const images = [{ filename: "photo.png", data: "data:image/png;base64,test" }];
  h.context.pendingImages = images;
  await h.choose("team", "kernel", "never-send-to-kernel");
  await h.context.send();
  assert.equal(h.turns().length, 0);
  assert.equal(h.state().pending, true);
  assert.equal(h.state().choice.backend, "kernel");
  assert.equal(h.state().choice.model, "never-send-to-kernel");
  assert.equal(h.elements.prompt.value, "一起完成这个任务");
  assert.equal(h.context.pendingImages, images);
  assert.equal(h.consumed.length, 0);
  assert.equal(h.context.sendBusy, false);
  assert.equal(h.elements.teamtrigger.dataset.state, "error");
  assert.ok(h.notes.some((text) => /503 temporarily offline/.test(text)));
  h.context.openTeamSheet();
  assert.equal(h.elements.teamkernel.attrs["aria-checked"], "true");
  h.context.closeTeamSheet();
  fail = false;
  h.configs.set("fresh", KERNEL);
  await h.context.send();
  assert.equal(h.turns().length, 1);
  assert.equal(h.teamPosts().length, 2);
  assert.equal("model" in h.teamPosts()[1].body.subagents, false);
});

test("cancel, Escape and browser back discard sheet edits without persisting", async () => {
  for (const action of ["cancel", "escape", "back"]) {
    const h = harness();
    await h.context.loadSessionConfig();
    h.context.openTeamSheet();
    h.context.changeTeamDraft("mode", "team");
    h.context.changeTeamDraft("backend", "traecli");
    h.elements.teamworker.value = "draft-worker";
    h.context.teamWorkerChanged();
    if (action === "cancel") h.elements.teamcancel.click();
    if (action === "escape") h.context.teamSheetKeydown({ key: "Escape", preventDefault() {} });
    if (action === "back") h.listeners.get("popstate")();
    await settle();
    assert.equal(h.teamPosts().length, 0);
    assert.deepEqual(plain(h.state().choice), { mode: "normal", backend: "kernel", model: "" });
    assert.equal(h.elements.teampick.hidden, true);
    assert.equal(h.elements.teamtrigger.attrs["aria-expanded"], "false");
    assert.equal(h.context.document.activeElement, h.elements.teamtrigger);
  }
});

test("saving can be dismissed, and send waits for the same request without duplicate POSTs", async () => {
  const save = deferred(), h = harness({ configPost: () => save.promise });
  await h.context.loadSessionConfig();
  const applying = h.choose("team", "kernel");
  await settle();
  assert.equal(h.elements.teamapply.disabled, true);
  assert.equal(h.elements.teamcancel.textContent, "收起");
  assert.equal(h.state().confirmed.mode, "normal");
  h.elements.teamcancel.click();
  const sending = h.context.send();
  await settle();
  assert.equal(h.teamPosts().length, 1);
  assert.equal(h.turns().length, 0);
  h.configs.set("alpha", KERNEL);
  save.resolve(response(KERNEL));
  assert.equal(await applying, true);
  await sending;
  assert.equal(h.turns().length, 1);
});

test("repeated send taps share one preparation and cannot create duplicate sessions or turns", async () => {
  const save = deferred(), h = harness({ sid: "", configPost: () => save.promise });
  await h.choose("team", "kernel");
  const first = h.context.send(), second = h.context.send();
  await settle();
  assert.equal(h.requests.filter((r) => r.url === "https://a.test/sessions").length, 1);
  assert.equal(h.teamPosts().length, 1);
  assert.equal(h.turns().length, 0);
  h.configs.set("fresh", KERNEL);
  save.resolve(response(KERNEL));
  await Promise.all([first, second]);
  assert.equal(h.turns().length, 1);
});

test("an open sheet preserves edits and displays a failed save without acknowledging it", async () => {
  const h = harness({ configPost: () => { throw new Error("network disconnected"); } });
  await h.context.loadSessionConfig();
  assert.equal(await h.choose("team", "traecli", "worker-model"), false);
  assert.equal(h.elements.teampick.hidden, false);
  assert.equal(h.elements.teamworker.value, "worker-model");
  assert.equal(h.elements.teamerror.hidden, false);
  assert.match(h.elements.teamerror.textContent, /network disconnected/);
  assert.equal(h.elements.teamapply.disabled, false);
  assert.equal(h.state().confirmed.mode, "normal");
  assert.equal(h.state().pending, true);
});

test("a successful HTTP status with mismatched structured config cannot submit a turn", async () => {
  const h = harness({ sid: "", configPost: () => response(NORMAL) });
  await h.choose("team", "traecli", "worker-model");
  await h.context.send();
  assert.equal(h.turns().length, 0);
  assert.equal(h.state().pending, true);
  assert.match(h.state().error, /尚未确认/);
});

test("late GETs cannot overwrite pending intent or an acknowledged team save", async () => {
  const oldRead = deferred(), h = harness({ configGet: () => oldRead.promise });
  const reading = h.context.loadSessionConfig();
  await h.choose("team", "traecli", "worker-model");
  oldRead.resolve(response(NORMAL));
  await reading;
  assert.equal(h.state().choice.mode, "team");
  assert.equal(h.state().choice.backend, "traecli");
  assert.equal(h.state().pending, false);
  assert.equal(h.elements.teamlabel.textContent, "团队协作 · TraeCLI");
});

test("session switches restore each selection and never inherit it into a new session", async () => {
  const h = harness({ cfg: TRAECLI });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.context.changeTeamDraft("mode", "normal");
  h.context.switchSession("fresh");
  assert.equal(h.elements.teampick.hidden, true);
  await h.context.loadSessionConfig();
  assert.equal(h.state().choice.mode, "normal");
  h.context.switchSession("alpha");
  assert.equal(h.state().choice.backend, "traecli");
  assert.equal(h.elements.teamlabel.textContent, "团队协作 · TraeCLI");
  assert.equal(h.teamPosts().length, 0);
});

test("navigation during first-session creation cannot overwrite the selected session or submit", async () => {
  const created = deferred(), h = harness({ sid: "", createSession: () => created.promise });
  await h.choose("team", "kernel");
  const sending = h.context.send();
  await settle();
  h.context.switchSession("beta");
  created.resolve(response({ id: "fresh" }));
  await sending;
  assert.equal(h.context.sessionId, "beta");
  assert.equal(h.teamPosts().length, 0);
  assert.equal(h.turns().length, 0);
  assert.equal(h.state("").pending, true);
});

test("session or server changes during save prevent the old input from becoming a turn", async () => {
  for (const change of ["session", "server"]) {
    const save = deferred(), h = harness({ configPost: () => save.promise });
    await h.context.loadSessionConfig();
    const applying = h.choose("team", "traecli", "worker-model");
    await settle();
    h.context.closeTeamSheet();
    const sending = h.context.send();
    await settle();
    if (change === "session") h.context.switchSession("beta");
    else { h.context.store.base = "https://other.test"; h.context.restoreTeamPick(); }
    save.resolve(response(TRAECLI));
    await applying;
    await sending;
    assert.equal(h.turns().length, 0);
    assert.equal(h.state("alpha", "https://a.test").confirmed.backend, "traecli");
    assert.equal(h.state().choice.mode, "normal");
    assert.notEqual(h.elements.teamlabel.textContent, "团队协作 · TraeCLI");
  }
});

test("input edits and active composition during config saving preserve the new draft", async () => {
  for (const composing of [false, true]) {
    const save = deferred(), h = harness({ sid: "", configPost: () => save.promise });
    await h.choose("team", "kernel");
    const sending = h.context.send();
    await settle();
    if (composing) h.context.promptComposing = true;
    else h.elements.prompt.value = "修改后的问题";
    save.resolve(response(KERNEL));
    await sending;
    assert.equal(h.turns().length, 0);
    assert.equal(h.elements.prompt.value, composing ? "一起完成这个任务" : "修改后的问题");
  }
});

test("keyboard navigation uses real backend radios, traps focus and respects model IME composition", async () => {
  const h = harness();
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.context.teamSheetKeydown({ key: "ArrowDown", preventDefault() {} });
  assert.equal(h.elements.teamcollab.attrs["aria-checked"], "true");
  h.elements.teamkernel.focus();
  h.context.teamSheetKeydown({ key: "ArrowRight", preventDefault() {} });
  assert.equal(h.elements.teamtraecli.attrs["aria-checked"], "true");
  assert.equal(h.elements.teamworkerwrap.hidden, false);
  h.elements.teamapply.focus();
  h.context.teamSheetKeydown({ key: "Tab", preventDefault() {} });
  assert.equal(h.context.document.activeElement, h.elements.teamclose);
  h.elements.teamworker.oncompositionstart();
  h.context.teamSheetKeydown({ key: "Escape", preventDefault() { assert.fail("IME key must not dismiss the sheet"); } });
  assert.equal(h.elements.teampick.hidden, false);
  h.elements.teamworker.oncompositionend();
});

test("worker model stays separate from the main model and blank preserves the saved worker", async () => {
  const h = harness({ cfg: { ...TRAECLI, model: "main-model" } });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  assert.match(h.elements.teamworkerhint.textContent, /主模型.*model.*worker-model/);
  h.elements.teamworkerclear.click();
  assert.equal(h.elements.omodel.value, "main-model");
  assert.equal(await h.context.applyTeamDraft(), true);
  assert.deepEqual(h.teamPosts()[0].body, { subagents: { action: "use", backend: "traecli" } });
  assert.equal(h.state().choice.model, "worker-model");
});

// The Rust driver owns command parsing. The turn fixtures below supply its
// persisted result; the assertions exercise the actual client send and GET path.
test("mobile /use and /reset preserve the complete input and restore authoritative config", async () => {
  for (const [input, before, after] of [
    ["/use traecli worker-model", NORMAL, TRAECLI],
    ["/reset", TRAECLI, NORMAL],
    [" \r\n :use kernel\r\n\r\n 第一行\r\n第二行  \r\n", TRAECLI, { ...KERNEL, subagent_model: "worker-model" }],
    ["/use", TRAECLI, TRAECLI],
    ["/use xxx", KERNEL, KERNEL],
    ["/reset extra", TRAECLI, TRAECLI],
    ["/useful kernel", KERNEL, KERNEL],
    ["/resetting", TRAECLI, TRAECLI],
    ["/reset/other", TRAECLI, TRAECLI],
  ]) {
    const h = harness({ cfg: before, turnPost: (request) => {
      assert.equal(request.body.prompt, input);
      assert.equal("subagents" in request.body, false);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    await h.context.loadSessionConfig();
    h.elements.prompt.value = input;
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 1);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal("subagents" in h.turns()[0].body, false);
    assert.equal(h.teamPosts().length, 0);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    assert.equal(h.elements.teamlabel.textContent, h.context.teamChoiceLabel(h.state().choice));
  }
});

test("commands wait for an earlier UI save, then win without a later replay of the old selection", async () => {
  for (const [input, backend, earlier, after] of [
    ["/use traecli worker-model", "kernel", KERNEL, TRAECLI],
    ["/reset", "traecli", TRAECLI, NORMAL],
  ]) {
    const save = deferred();
    let turnCount = 0;
    const h = harness({ configPost: () => save.promise, turnPost: (request) => {
      assert.equal(request.body.prompt, turnCount++ ? "后续消息" : input);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    await h.context.loadSessionConfig();
    const applying = h.choose("team", backend, backend === "traecli" ? "worker-model" : "");
    await settle();
    h.context.closeTeamSheet();
    h.elements.prompt.value = input;
    const sending = h.context.send();
    await settle();
    assert.equal(h.teamPosts().length, 1);
    assert.equal(h.turns().length, 0);
    assert.equal(h.elements.prompt.value, input);
    h.configs.set("alpha", plain(earlier));
    save.resolve(response(earlier));
    await Promise.all([applying, sending]);
    await settle();
    const writeIndex = h.requests.findIndex((r) => r.body?.subagents);
    const turnIndex = h.requests.findIndex((r) => r.url.endsWith("/turns/stream"));
    assert.ok(writeIndex < turnIndex);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.ok(h.requests.slice(turnIndex + 1).some((r) => r.method === "GET" && r.url.endsWith("/config")));
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    h.elements.prompt.value = "后续消息";
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 2);
    assert.equal(h.teamPosts().length, 1);
    assert.equal(h.state().pending, false);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("a first /use or /reset carries the UI intent to the new session and keeps the raw command last", async () => {
  for (const [input, after] of [["/use traecli worker-model", TRAECLI], ["/reset", NORMAL]]) {
    const save = deferred();
    const h = harness({ sid: "", configPost: () => save.promise, turnPost: (request) => {
      assert.equal(request.url, "https://a.test/sessions/fresh/turns/stream");
      assert.equal(request.body.prompt, input);
      h.configs.set("fresh", plain(after));
      return turnResponse();
    } });
    await h.choose("team", "kernel");
    h.elements.prompt.value = input;
    const sending = h.context.send();
    await settle();
    assert.equal(h.context.sessionId, "fresh");
    assert.equal(h.context.pendingNew, "fresh");
    assert.equal(h.teamPosts()[0].url, "https://a.test/sessions/fresh/config");
    assert.equal(h.turns().length, 0);
    h.configs.set("fresh", plain(KERNEL));
    save.resolve(response(KERNEL));
    await sending;
    await settle();
    const writesAndTurns = h.requests.filter((r) => r.body?.subagents || r.url.endsWith("/turns/stream"));
    assert.deepEqual(writesAndTurns.map((r) => r.url), ["https://a.test/sessions/fresh/config", "https://a.test/sessions/fresh/turns/stream"]);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal(h.teamPosts().length, 1);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("a failed mode save retains /use and /reset and retries before submitting either command", async () => {
  for (const [input, after] of [["/use traecli worker-model", TRAECLI], ["/reset", NORMAL]]) {
    let fail = true;
    const h = harness({ sid: "", configPost: () => {
      if (fail) return { ok: false, status: 503, json: async () => ({ error: "save unavailable" }) };
      h.configs.set("fresh", plain(KERNEL));
      return response(KERNEL);
    }, turnPost: (request) => {
      assert.equal(request.body.prompt, input);
      h.configs.set("fresh", plain(after));
      return turnResponse();
    } });
    await h.choose("team", "kernel");
    h.elements.prompt.value = input;
    await h.context.send();
    assert.equal(h.turns().length, 0);
    assert.equal(h.elements.prompt.value, input);
    assert.equal(h.state().pending, true);
    assert.ok(h.notes.some((text) => text.includes("save unavailable")));
    fail = false;
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 1);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal(h.teamPosts().length, 2);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    assert.equal(h.state().pending, false);
  }
});

test("a GET started before the command cannot replace its newer authoritative result", async () => {
  for (const [input, before, after] of [["/use traecli worker-model", NORMAL, TRAECLI], ["/reset", TRAECLI, NORMAL]]) {
    const oldRead = deferred();
    let reads = 0;
    const h = harness({ cfg: before, configGet: () => ++reads === 1 ? oldRead.promise : response(after), turnPost: (request) => {
      assert.equal(request.body.prompt, input);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    h.context.applySessionConfig(before);
    const reading = h.context.loadSessionConfig({ teamOnly: true });
    h.elements.prompt.value = input;
    await h.context.send();
    await settle();
    assert.equal(reads, 2);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    oldRead.resolve(response(before));
    await reading;
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("a UI application made after command submission is a newer intent and wins when its save completes", async () => {
  const finished = deferred(), save = deferred();
  const h = harness({ cfg: TRAECLI, configPost: () => save.promise, turnPost: (request) => {
    assert.equal(request.body.prompt, "/reset");
    return turnResponse(() => finished.promise);
  } });
  await h.context.loadSessionConfig();
  h.elements.prompt.value = "/reset";
  const sending = h.context.send();
  await settle();
  assert.equal(h.turns()[0].body.prompt, "/reset");
  assert.equal(h.context.turns.size, 1);
  const applying = h.choose("team", "kernel");
  await settle();
  const turnIndex = h.requests.findIndex((r) => r.url.endsWith("/turns/stream"));
  const writeIndex = h.requests.findIndex((r) => r.body?.subagents);
  assert.ok(turnIndex < writeIndex);
  h.configs.set("alpha", plain(NORMAL));
  finished.resolve({ done: false, value: Buffer.from("event: done\n\n") });
  await sending;
  await settle();
  assert.equal(h.state().pending, true);
  assert.equal(h.state().choice.backend, "kernel");
  assert.equal(h.elements.teamstatus.textContent, "保存中");
  h.configs.set("alpha", plain(KERNEL));
  save.resolve(response(KERNEL));
  await applying;
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(KERNEL)));
  assert.equal(h.state().pending, false);
});

test("cancelled UI edits leave /reset on the existing turn path", async () => {
  const h = harness({ cfg: TRAECLI, turnPost: (request) => {
    assert.equal(request.body.prompt, "/reset");
    h.configs.set("alpha", plain(NORMAL));
    return turnResponse();
  } });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.context.changeTeamDraft("mode", "normal");
  h.context.closeTeamSheet();
  assert.equal(h.state().choice.backend, "traecli");
  h.elements.prompt.value = "/reset";
  await h.context.send();
  await settle();
  assert.equal(h.turns()[0].body.prompt, "/reset");
  assert.equal(h.teamPosts().length, 0);
  assert.equal(h.state().choice.mode, "normal");
});

test("switching sessions during a mode save keeps the command draft and cannot submit it in the new session", async () => {
  for (const [input, after] of [["/use traecli worker-model", TRAECLI], ["/reset", NORMAL]]) {
    const save = deferred();
    const h = harness({ configPost: () => save.promise, turnPost: (request) => {
      assert.equal(request.url, "https://a.test/sessions/alpha/turns/stream");
      assert.equal(request.body.prompt, input);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    await h.context.loadSessionConfig();
    const applying = h.choose("team", "kernel");
    await settle();
    h.context.closeTeamSheet();
    h.elements.prompt.value = input;
    const sending = h.context.send();
    await settle();
    h.context.switchSession("beta");
    h.configs.set("alpha", plain(KERNEL));
    save.resolve(response(KERNEL));
    await Promise.all([applying, sending]);
    assert.equal(h.turns().length, 0);
    assert.equal(h.stored.get("draft.alpha"), input);
    h.context.switchSession("alpha");
    assert.equal(h.elements.prompt.value, input);
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 1);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal(h.teamPosts().length, 1);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("desktop keeps its send behavior and cannot open the phone team sheet", async () => {
  const h = harness({ phone: false });
  h.context.openTeamSheet();
  assert.equal(h.elements.teampick.hidden, true);
  assert.equal(h.history.pushes, 0);
  await h.context.send();
  assert.equal(h.turns().length, 1);
  assert.equal(h.teamPosts().length, 0);
});

test("the entry is phone-only, at least 44px, and the accessible sheet uses the existing theme and viewport", () => {
  const phoneStart = css.indexOf("@media (max-width:700px)"), desktop = css.slice(0, phoneStart), phoneCss = css.slice(phoneStart);
  assert.match(desktop, /#teamcontrols, #teampick \{ display:none; \}/);
  assert.match(phoneCss, /#teamcontrols \{ display:flex;/);
  assert.match(phoneCss, /#teampick:not\(\[hidden\]\) \{ display:block; \}/);
  assert.match(desktop, /#teamtrigger \{[^}]*min-height:44px;[^}]*border-radius:999px;/);
  assert.match(desktop, /#teampickpanel \{[^}]*max-height:calc\(var\(--app-height, 100vh\) \* \.88\);[^}]*env\(safe-area-inset-bottom\)[^}]*background:var\(--panel\)/);
  assert.match(desktop, /#teambody \{[^}]*min-height:0; overflow-y:auto; overscroll-behavior:contain; touch-action:pan-y;/);
  assert.match(html, /id="teampickpanel" role="dialog" aria-modal="true" aria-labelledby="teamtitle" aria-describedby="teamnote"/);
  assert.match(html, /id="teamworker"[^>]*aria-describedby="teamworkerhint"/);
  assert.match(css, /prefers-reduced-motion:reduce\) \{ #teampickpanel, #teampickscrim \{ animation:none; \}/);
  assert.ok(html.indexOf('<div id="teamcontrols">') < html.indexOf('<div id="composer">'));
});
