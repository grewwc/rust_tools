// Run with: node --test src/bin/ai/serve/app_frame.test.cjs
// A captured phone frame must appear in the conversation itself, so the user
// can see the very picture the assistant is looking at.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function fakeElement(tag) {
  return {
    tagName: tag,
    className: "",
    textContent: "",
    src: "",
    alt: "",
    loading: "",
    children: [],
    onclick: null,
    __root: null,
    parent: null,
    appendChild(child) {
      child.__root = this.__root || this;
      child.parent = this;
      this.children.push(child);
      return child;
    },
    append(...kids) {
      for (const kid of kids) this.appendChild(kid);
    },
    querySelectorAll() {
      return [];
    },
    contains(node) {
      for (let n = node; n; n = n.parent) if (n === this) return true;
      return false;
    },
  };
}

function harness() {
  const frameSrc = script.slice(script.indexOf("const phoneFrameUrls"), script.indexOf("async function queueImageFiles"));
  const revokeSrc = script.slice(script.indexOf("function revokePreviews("), script.indexOf("// A history fold is titled"));
  const slice = [
    "const previewUrls = new Map();",
    frameSrc,
    revokeSrc,
    "globalThis.__phoneFrameUrls = phoneFrameUrls;",
    "globalThis.__previewUrls = previewUrls;",
  ].join("\n");
  const transcript = fakeElement("div");
  transcript.__root = transcript;
  const box = fakeElement("div");
  box.__root = box;
  const turns = new Map([["sid", { box }]]);
  const created = [];
  const revoked = [];
  const context = vm.createContext({
    document: {
      createElement: (tag) => {
        const el = fakeElement(tag);
        created.push(el);
        return el;
      },
    },
    turns,
    atob,
    Blob,
    URL: {
      createObjectURL: () => {
        created.push.length; // keep the shape obvious
        return "blob:frame-1";
      },
      revokeObjectURL: (url) => revoked.push(url),
    },
    $: (id) => (id === "msgs" ? transcript : null),
    showViewer: () => {},
  });
  vm.runInContext(slice, context);
  return { context, transcript, box, turns, revoked };
}

const JPEG_BASE64 = Buffer.from("fake-jpeg-bytes").toString("base64");

test("a captured frame is rendered into the live turn with a caption", () => {
  const h = harness();
  const note = h.context.renderPhoneFrame(
    { image_base64: JPEG_BASE64, mime_type: "image/jpeg", captured_at: 0, capture_source: "pre_tap" },
    "sid"
  );
  assert.ok(note, "no frame card was rendered");
  assert.equal(h.box.children.length, 1, "the frame must land in the live turn");
  const [img, cap] = h.box.children[0].children;
  assert.equal(img.tagName, "img");
  assert.equal(img.src, "blob:frame-1");
  assert.equal(cap.className, "phone-frame-cap");
  assert.match(cap.textContent, /屏幕画面/);
});

test("the caption names the moment the frame was taken", () => {
  const h = harness();
  const pre = h.context.phoneFrameCaption({ capture_source: "pre_tap" });
  assert.match(pre, /点按前/);
  // The bubble trigger delivers the frame of the tap itself.
  const bubble = h.context.phoneFrameCaption({ capture_source: "bubble" });
  assert.match(bubble, /点按时/);
  const live = h.context.phoneFrameCaption({ capture_source: "live" });
  assert.match(live, /当前画面/);
  // A frame without a capture time must not print a broken stamp.
  const bare = h.context.phoneFrameCaption({});
  assert.ok(!/NaN|undefined/.test(bare), "caption leaked an invalid time: " + bare);
});

test("a result without an image renders nothing", () => {
  const h = harness();
  assert.equal(h.context.renderPhoneFrame({}, "sid"), null);
  assert.equal(h.context.renderPhoneFrame({ image_base64: "" }, "sid"), null);
  assert.equal(h.box.children.length, 0);
});

test("the frame URL is released when the view is rebuilt", () => {
  const h = harness();
  h.context.renderPhoneFrame({ image_base64: JPEG_BASE64, mime_type: "image/jpeg" }, "sid");
  assert.equal(h.context.__phoneFrameUrls.size, 1);
  h.context.revokePreviews(h.box);
  assert.equal(h.context.__phoneFrameUrls.size, 0);
  assert.deepEqual(h.revoked, ["blob:frame-1"]);
});

test("the captured frame stays a thumbnail instead of filling the screen", () => {
  const style = Array.from(html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g))
    .map((match) => match[1]).join("\n");
  // The frame element carries the `msg` and `phone-frame` classes itself, so a
  // descendant selector never matches and the image would render at its
  // natural pixel size, covering the whole viewport.
  assert.doesNotMatch(style, /\.msg \.phone-frame/);
  assert.match(style, /\.msg\.phone-frame img/);
  const rule = style.match(/\.msg\.phone-frame img \{([^}]*)\}/)[1];
  assert.match(rule, /width:auto/, "the frame must keep its aspect ratio");
  assert.match(rule, /max-width:100%/, "the frame must not overflow the bubble");
  const cap = rule.match(/max-height:([^;]+);/);
  assert.ok(cap, "the frame image must be capped by a viewport height");
  const dvh = cap[1].match(/(\d+)dvh/);
  assert.ok(dvh, `the frame cap must use dvh, got ${cap[1]}`);
  assert.ok(Number(dvh[1]) <= 40, `the frame must stay a thumbnail, got ${cap[1]}`);
  assert.match(script, /点按查看大图/, "the caption must offer the full-screen view");
});
