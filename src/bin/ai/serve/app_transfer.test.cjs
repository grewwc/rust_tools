// Run with: node --test src/bin/ai/serve/app_transfer.test.cjs
// `/export`, `/export-md` and `/import` must move archive bytes between the
// browser and the daemon instead of becoming model turns. The page fetches an
// attachment and hands it to the download path, or uploads one picked archive
// and switches to the imported session.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function harness({ sessionId = "s1", respond, pickedFile } = {}) {
  const calls = [];
  const cards = [];
  const downloads = [];
  const switched = [];
  const input = {
    style: {},
    files: null,
    handlers: {},
    addEventListener(type, handler) { this.handlers[type] = handler; },
    remove() {},
    click() {
      this.files = pickedFile ? [pickedFile] : null;
      if (this.handlers.change) this.handlers.change();
    },
  };
  const context = vm.createContext({
    sessionId,
    pendingImages: [],
    api: async (url, opts) => {
      calls.push({ url, opts });
      return respond
        ? respond(url, opts)
        : {
            ok: true,
            headers: { get: () => 'attachment; filename="session-s1.zip"' },
            blob: async () => ({ type: "application/zip", size: 3 }),
          };
    },
    addMsg: (role, text, cls) => { cards.push({ role, text, cls }); return {}; },
    $: () => ({ style: {}, value: "" }),
    setPromptValue: () => {},
    clearDraft: () => {},
    scrollToLatest: () => {},
    switchSession: (id) => switched.push(id),
    loadSessions: async () => {},
    nativeBridgeReady: () => false,
    document: {
      createElement: (tag) => tag === "input" ? input : { style: {}, click() {}, remove() {} },
      body: { appendChild() {} },
    },
    URL: { createObjectURL: () => "blob:x", revokeObjectURL: () => {} },
    setTimeout: () => {},
    navigator: {},
    FileReader: class {
      readAsDataURL(file) {
        this.result = "data:application/zip;base64," + Buffer.from(file.bytes).toString("base64");
        this.onload();
      }
    },
  });
  const start = script.indexOf("// BEGIN composer export/import commands");
  const end = script.indexOf("// END composer export/import commands");
  vm.runInContext(script.slice(start, end), context);
  return {
    calls,
    cards,
    downloads,
    switched,
    run: (text) => vm.runInContext(`runTransferCommand(${JSON.stringify(text)})`, context),
  };
}

test("export fetches the session archive and downloads it", async () => {
  const h = harness();
  assert.equal(await h.run("/export"), true);
  assert.equal(h.calls.length, 1, "one request, and it is not a turn");
  assert.equal(h.calls[0].url, "/sessions/s1/export");
  assert.ok(h.cards.some((c) => c.role === "user" && c.text === "/export"));
  assert.ok(h.cards.some((c) => c.role === "asst" && /session-s1\.zip/.test(c.text)));
});

test("export-md names the download from the command argument", async () => {
  const h = harness();
  assert.equal(await h.run("/export-md notes.md"), true);
  assert.equal(h.calls[0].url, "/sessions/s1/export-md");
  assert.ok(h.cards.some((c) => c.role === "asst" && /notes\.md/.test(c.text)));
});

test("import uploads one archive and switches to the returned session", async () => {
  const h = harness({
    pickedFile: { name: "migrate.zip", bytes: "zip-bytes" },
    respond: (url) => url === "/sessions/import"
      ? { ok: true, json: async () => ({ id: "new1" }) }
      : null,
  });
  assert.equal(await h.run("/import as=fixed"), true);
  assert.equal(h.calls[0].url, "/sessions/import");
  const body = JSON.parse(h.calls[0].opts.body);
  assert.equal(body.filename, "migrate.zip");
  assert.equal(body.id, "fixed");
  assert.equal(body.data, Buffer.from("zip-bytes").toString("base64"));
  assert.deepEqual(h.switched, ["new1"]);
});

test("multi-line text and other prompts still go to the model", async () => {
  for (const text of ["/export\nand then explain", "please export", "/exportx"]) {
    const h = harness();
    assert.equal(await h.run(text), false, JSON.stringify(text));
    assert.equal(h.calls.length, 0);
  }
});

test("send() consults the transfer bridge before opening a turn stream", () => {
  const call = script.indexOf('if (typeof runTransferCommand === "function" && await runTransferCommand(text)) return;');
  assert.ok(call > 0, "send() must call the bridge");
  assert.ok(call < script.indexOf('"/turns/stream"'), "the bridge runs before the turn request");
});
