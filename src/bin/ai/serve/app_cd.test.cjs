// Run with: node --test src/bin/ai/serve/app_cd.test.cjs
// `/cd` must reach the session's virtual working directory on the server. Sent
// as a turn it would run inside the turn child, which exits and takes the
// change with it, so the page answers the command locally and never opens a
// stream. Everything else (multi-line text, images) still goes to the model.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function harness({ sessionId = "s1", images = [], respond } = {}) {
  const calls = [];
  const cards = [];
  const context = vm.createContext({
    sessionId,
    pendingImages: images,
    api: async (url, opts) => {
      calls.push({ url, opts });
      return respond
        ? respond(url, opts)
        : { ok: true, json: async () => ({ stdout: "Working directory: /tmp\n", stderr: "" }) };
    },
    addMsg: (role, text, cls) => { cards.push({ role, text, cls }); return {}; },
    addFold: (title, body) => {
      const card = { fold: title, body, cls: [] };
      cards.push(card);
      return { box: { classList: { add: (c) => card.cls.push(c) } } };
    },
    $: () => ({ style: {}, value: "" }),
    setPromptValue: () => {},
    clearDraft: (sid) => { cards.push({ cleared: sid }); },
    shortLine: (s) => s.slice(0, 80),
    scrollToLatest: () => {},
    localStorage: { setItem: () => {} },
  });
  const code = script.slice(script.indexOf("const CD_LINE_RE"), script.indexOf("async function send()"));
  vm.runInContext(code, context);
  return {
    calls,
    cards,
    run: (text) => vm.runInContext(`runCdCommand(${JSON.stringify(text)})`, context),
    line: (text) => vm.runInContext(`cdCommandLine(${JSON.stringify(text)})`, context),
  };
}

test("a cd line posts to the session endpoint instead of starting a turn", async () => {
  const h = harness();
  assert.equal(await h.run("/cd /tmp"), true);
  assert.equal(h.calls.length, 1, "one request, and it is not a turn");
  assert.equal(h.calls[0].url, "/sessions/s1/cd");
  assert.deepEqual(JSON.parse(h.calls[0].opts.body), { command: "/cd /tmp" });
  assert.ok(h.cards.some((c) => c.role === "user" && c.text === "/cd /tmp"));
  const fold = h.cards.find((c) => c.fold);
  assert.match(fold.fold, /^activity log \(1\) · Working directory: \/tmp$/);
  assert.match(fold.body, /Working directory: \/tmp/);
  assert.ok(fold.cls.includes("tool-activity"));
});

test("a bare cd line and the :cd spelling are both handled", async () => {
  for (const text of ["/cd", ":cd .."]) {
    const h = harness();
    assert.equal(await h.run(text), true, text);
    assert.equal(h.calls[0].url, "/sessions/s1/cd");
  }
});

test("multi-line text, other prompts and pending images still go to the model", async () => {
  for (const text of ["/cd /tmp\nand also list the files", "what is in /tmp", "/cdx"]) {
    const h = harness();
    assert.equal(await h.run(text), false, JSON.stringify(text));
    assert.equal(h.calls.length, 0);
  }
  const withImage = harness({ images: [{ data: "x" }] });
  assert.equal(await withImage.run("/cd /tmp"), false, "an image cannot ride along with cd");
  assert.equal(withImage.calls.length, 0);
});

test("no session yet creates one, then applies the cd there", async () => {
  const h = harness({
    sessionId: "",
    respond: (url) => url === "/sessions"
      ? { ok: true, json: async () => ({ id: "fresh" }) }
      : { ok: true, json: async () => ({ stdout: "Working directory: /tmp\n", stderr: "" }) },
  });
  assert.equal(await h.run("/cd /tmp"), true);
  assert.deepEqual(h.calls.map((c) => c.url), ["/sessions", "/sessions/fresh/cd"]);
});

test("a rejected cd shows the server error and no activity card", async () => {
  const h = harness({
    respond: () => ({
      ok: false, status: 400, statusText: "Bad Request",
      json: async () => ({ error: "no such directory" }),
    }),
  });
  assert.equal(await h.run("/cd /nope"), true);
  const err = h.cards.find((c) => c.cls && c.cls.includes("err"));
  assert.ok(err, "the rejection must be visible");
  assert.match(err.text, /no such directory/);
  assert.equal(h.cards.some((c) => c.fold), false);
});

test("send() consults the cd bridge before opening a turn stream", () => {
  const call = script.indexOf("if (await runCdCommand(text)) return;");
  assert.ok(call > 0, "send() must call the bridge");
  assert.ok(call < script.indexOf('"/turns/stream"'), "the bridge runs before the turn request");
});