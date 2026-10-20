// Run with: node --test src/bin/ai/serve/app_native.test.cjs
// Validate the phone-action spec before it reaches the Android bridge.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const slice = script.slice(
  script.indexOf("const esc ="),
  script.indexOf("function sessionTitle("),
);

function load() {
  const context = vm.createContext({ window: {} });
  vm.runInContext(slice, context);
  return context;
}

test("the complete embedded page script parses", () => {
  assert.doesNotThrow(() => new vm.Script(script));
});

test("parses the three allowlisted alarm fields and drops everything else", () => {
  const app = load();
  const alarm = app.parsePhoneAction(JSON.stringify({
    action: "alarm",
    hour: 7,
    minute: 30,
    label: "Wake up",
    intent: "com.example.evil",
  }));
  assert.equal(alarm.action, "alarm");
  assert.equal(alarm.hour, 7);
  assert.equal(alarm.minute, 30);
  assert.equal(alarm.label, "Wake up");
  assert.equal(Object.hasOwn(alarm, "intent"), false);
});

test("rejects out-of-range or non-numeric alarm times", () => {
  const app = load();
  assert.equal(app.parsePhoneAction(JSON.stringify({ action: "alarm", hour: 24, minute: 0 })), null);
  assert.equal(app.parsePhoneAction(JSON.stringify({ action: "alarm", hour: 6, minute: 60 })), null);
  assert.equal(app.parsePhoneAction(JSON.stringify({ action: "alarm", hour: "6", minute: 0 })), null);
  assert.equal(app.parsePhoneAction(JSON.stringify({ action: "alarm", hour: 6.5, minute: 0 })), null);
});

test("bounds note text and accepts location", () => {
  const app = load();
  assert.equal(app.parsePhoneAction(JSON.stringify({ action: "note", text: "  " })), null);
  const note = app.parsePhoneAction(JSON.stringify({ action: "note", title: "t", text: "body" }));
  assert.equal(note.action, "note");
  assert.equal(note.text, "body");
  const location = app.parsePhoneAction(JSON.stringify({ action: "location", extra: 1 }));
  assert.equal(location.action, "location");
  assert.equal(Object.hasOwn(location, "extra"), false);
});

test("rejects unknown actions, invalid JSON, and non-objects", () => {
  const app = load();
  assert.equal(app.parsePhoneAction(JSON.stringify({ action: "launch", package: "x" })), null);
  assert.equal(app.parsePhoneAction("{"), null);
  assert.equal(app.parsePhoneAction("[]"), null);
  assert.equal(app.parsePhoneAction("null"), null);
});

test("does not turn a fenced aserve block into a tappable button", () => {
  const app = load();
  const rendered = app.renderMarkdown(
    '```aserve\n{"action":"alarm","hour":7,"minute":30}\n```',
  );
  assert.match(rendered, /<pre data-lang="aserve">/);
  assert.doesNotMatch(rendered, /data-aserve-run/);
  assert.doesNotMatch(rendered, /aserve-card/);
});

const phoneFixtures = {
  now: { action: "now" },
  calendar_read: { action: "calendar_read", days: 7, max: 10 },
  calendar_add: { action: "calendar_add", title: "Meeting", begin_local: "2026-10-21T15:00", duration_min: 60, location: "Room 1", description: "Agenda" },
  sms: { action: "sms", contact: "", number: "10086", text: "你好" },
  alarm: { action: "alarm", hour: 9, minute: 30, label: "Wake up" },
  note: { action: "note", title: "Note", text: "Content" },
  location: { action: "location" },
};

test("every advertised phone action has a valid browser payload", () => {
  const tool = JSON.parse(fs.readFileSync(path.join(__dirname, "../tool_descriptions/phone_action.json"), "utf8"));
  assert.deepEqual(Object.keys(phoneFixtures).sort(), [...tool.parameters.properties.action.enum].sort());
  const app = load();
  for (const spec of Object.values(phoneFixtures)) {
    const parsed = app.parsePhoneAction(JSON.stringify({ ...spec, auto: false, package: "untrusted" }));
    assert.deepEqual(JSON.parse(JSON.stringify(parsed)), spec, spec.action);
  }
});

test("SMS accepts contact lookup but rejects missing content and unsafe numbers", () => {
  const app = load(), parse = (spec) => app.parsePhoneAction(JSON.stringify(spec));
  assert.deepEqual(JSON.parse(JSON.stringify(parse({ action: "sms", contact: " 妈妈 ", text: "你好" }))),
    { action: "sms", contact: "妈妈", number: "", text: "你好" });
  for (const number of ["10086?body=bad", "10086;10010", "100\n86", "smsto:10086", "+()- .", 10086]) {
    assert.equal(parse({ action: "sms", number, text: "你好" }), null, String(number));
  }
  for (const spec of [
    { text: "hello" }, { number: "10086" }, { number: "10086", text: " " },
    { number: "10086", text: "x".repeat(1001) }, { contact: "x".repeat(121), text: "hi" },
    { number: "1".repeat(41), text: "hi" }, { contact: {}, text: "hi" },
  ]) assert.equal(parse({ action: "sms", ...spec }), null);
});

test("calendar payloads retain local dates, defaults, and strict bounds", () => {
  const app = load(), parse = (spec) => app.parsePhoneAction(JSON.stringify(spec));
  assert.deepEqual(JSON.parse(JSON.stringify(parse({ action: "calendar_read" }))),
    { action: "calendar_read", days: 7, max: 10 });
  for (const spec of [{ days: 0 }, { days: 31 }, { max: 21 }, { max: "10" }, { days: 1.5 }]) {
    assert.equal(parse({ action: "calendar_read", ...spec }), null);
  }
  const event = { action: "calendar_add", title: "Meeting", begin_local: "2028-02-29T15:00" };
  assert.deepEqual(JSON.parse(JSON.stringify(parse(event))), { ...event, duration_min: 60, location: "", description: "" });
  for (const begin_local of ["2026-02-29T15:00", "2026-04-31T15:00", "2026-13-01T15:00", "2026-10-21T24:00", "2026-10-21T15:60", "2026-10-21T15:00Z", "2026-10-21T15:00:00", 1792566000]) {
    assert.equal(parse({ ...event, begin_local }), null, String(begin_local));
  }
  for (const spec of [{ duration_min: 4 }, { duration_min: 601 }, { title: " " }, { title: "x".repeat(201) }, { description: "x".repeat(1001) }, { location: {} }]) {
    assert.equal(parse({ ...event, ...spec }), null);
  }
});

function nativeHarness(storage = new Map(), spec = { action: "alarm", hour: 9, minute: 30 }) {
  const calls = [], posts = [], errors = [], timers = new Map();
  const prompt = '@@aserve-native ' + JSON.stringify(spec);
  const pending = { id: 7, token: "turn-token", prompt, input: true };
  let rejectPost = false;
  const context = vm.createContext({
    window: {}, sessionId: "original", store: { base: "https://server" },
    sessionStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
    headers: () => ({ Authorization: "Bearer original" }),
    setTimeout(fn) { const id = timers.size + 1; timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); },
    addMsg: (...args) => errors.push(args),
    async fetch(url, options = {}) {
      if (options.method === "POST") {
        posts.push({ url, body: JSON.parse(options.body), headers: options.headers });
        if (rejectPost) throw new Error("network unavailable");
        return { ok: true };
      }
      return { ok: true, json: async () => ({ running: true, pending }) };
    },
  });
  context.window.AServeNative = { perform: (...args) => calls.push(args) };
  vm.runInContext(slice, context);
  // The production request path must work without the send closure's setPhase.
  vm.runInContext(script.slice(script.indexOf("async function api("), script.indexOf("const esc =")), context);
  vm.runInContext(script.slice(script.indexOf("const nativePending ="), script.indexOf("async function queueImageFiles(")), context);
  return {
    context, calls, posts, errors, timers, pending,
    request: () => context.handleNativeRequest(7, "turn-token", prompt),
    settle: () => new Promise((resolve) => setImmediate(resolve)),
    result(message = { ok: true, data: { action: "alarm" } }) {
      context.window.AServeNativeResult(JSON.stringify({ id: calls[0][0], ...message }));
    },
    failPost(value) { rejectPost = value; },
  };
}

test("every phone action reaches the bridge and returns its acknowledgement unchanged", async () => {
  for (const spec of Object.values(phoneFixtures)) {
    const h = nativeHarness(new Map(), spec);
    const request = h.request();
    await h.settle();
    assert.equal(h.calls.length, 1, spec.action);
    assert.equal(h.calls[0][1], spec.action);
    assert.deepEqual(JSON.parse(h.calls[0][2]), { auto: true, ...spec });
    const result = { ok: true, data: { action: spec.action, ...(spec.action === "sms" ? { sent: false } : {}) } };
    h.result(result);
    await request;
    assert.equal(h.posts.length, 1);
    assert.deepEqual(JSON.parse(h.posts[0].body.text), result);
    await h.request();
    assert.equal(h.calls.length, 1, "replay must not repeat " + spec.action);
  }
});

test("native request reaches the APK and returns its result to the server", async () => {
  const h = nativeHarness();
  const request = h.request();
  await h.settle();
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0][1], "alarm");
  assert.equal(JSON.parse(h.calls[0][2]).auto, true);
  h.result();
  await request;
  assert.equal(h.posts.length, 1);
  assert.equal(JSON.parse(h.posts[0].body.text).ok, true);
  assert.equal(h.timers.size, 0);
});

test("duplicate deliveries and failed acknowledgements never execute twice", async () => {
  const storage = new Map();
  const h = nativeHarness(storage);
  h.failPost(true);
  const first = h.request();
  await h.request();
  await h.settle();
  assert.equal(h.calls.length, 1);
  h.result();
  await first;
  h.failPost(false);
  assert.equal(h.timers.size, 1);
  const [timer, retry] = [...h.timers.entries()][0];
  h.timers.delete(timer);
  retry();
  await h.settle();
  assert.equal(h.calls.length, 1);
  assert.equal(h.posts.length, 2);
  assert.equal(h.posts[0].body.text, h.posts[1].body.text);
  await h.request();
  assert.equal(h.posts.length, 2);
  const reloaded = nativeHarness(storage);
  await reloaded.request();
  assert.equal(reloaded.calls.length, 0);
  assert.equal(reloaded.posts[0].body.text, h.posts[0].body.text);
});

test("replayed stale native requests do not reach the bridge", async () => {
  const h = nativeHarness();
  h.pending.token = "new-turn";
  await h.request();
  assert.equal(h.calls.length, 0);
  assert.equal(h.posts.length, 0);
});

test("native reply keeps original server, session and credentials after navigation", async () => {
  const h = nativeHarness();
  const first = h.request();
  await h.settle();
  h.context.sessionId = "other";
  h.context.store.base = "https://other";
  h.context.headers = () => ({ Authorization: "Bearer other" });
  h.result();
  await first;
  assert.equal(h.posts[0].url, "https://server/sessions/original/confirm");
  assert.equal(h.posts[0].headers.Authorization, "Bearer original");
});

test("native failure and timeout return explicit errors instead of hanging", async () => {
  for (const timeout of [false, true]) {
    const h = nativeHarness();
    const first = h.request();
    await h.settle();
    if (timeout) [...h.timers.values()][0]();
    else h.result({ ok: false, error: "no clock app" });
    await first;
    const result = JSON.parse(h.posts[0].body.text);
    assert.equal(result.ok, false);
    assert.match(result.error, timeout ? /手机没有响应/ : /no clock app/);
  }
});

test("reload during native execution reports uncertainty without repeating the action", async () => {
  const storage = new Map();
  const original = nativeHarness(storage);
  const first = original.request();
  await original.settle();
  const reloaded = nativeHarness(storage);
  await reloaded.request();
  assert.equal(reloaded.calls.length, 0);
  const result = JSON.parse(reloaded.posts[0].body.text);
  assert.equal(result.ok, false);
  assert.match(result.error, /结果未知/);
  original.result();
  await first;
});

test("note and location use the same native result path", async () => {
  for (const spec of [{ action: "note", text: "test note" }, { action: "location" }]) {
    const h = nativeHarness();
    h.pending.prompt = "@@aserve-native " + JSON.stringify(spec);
    const first = h.context.handleNativeRequest(7, "turn-token", h.pending.prompt);
    await h.settle();
    assert.equal(h.calls[0][1], spec.action);
    h.result({ ok: true, data: { action: spec.action } });
    await first;
    assert.equal(JSON.parse(h.posts[0].body.text).data.action, spec.action);
  }
});

test("a non-Android page returns an explicit error", async () => {
  const h = nativeHarness();
  delete h.context.window.AServeNative;
  await h.request();
  assert.equal(h.calls.length, 0);
  const result = JSON.parse(h.posts[0].body.text);
  assert.equal(result.ok, false);
  assert.match(result.error, /Android/);
});