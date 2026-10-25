// Run with: node --test src/bin/ai/serve/app_model_picker.test.cjs
// Exercise the model sheet without a browser: selection state, the default
// row, stale options after an info refresh, and keyboard navigation.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const between = (start, end) => {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing source slice: " + start);
  return script.slice(a, b);
};

function node(tag = "div", doc) {
  let text = "";
  const n = {
    tagName: tag, children: [], attrs: {}, dataset: {}, className: "", title: "",
    hidden: false, disabled: false, value: "", type: "",
    set textContent(v) { text = String(v); n.children = []; },
    get textContent() { return text + n.children.map((c) => c.textContent).join(""); },
    appendChild(c) { n.children.push(c); return c; },
    append(...cs) { n.children.push(...cs); },
    setAttribute(k, v) { n.attrs[k] = String(v); },
    getAttribute(k) { return n.attrs[k] ?? null; },
    removeAttribute(k) { delete n.attrs[k]; },
    focus() { doc.activeElement = n; },
    scrollIntoView() {},
    querySelectorAll(selector) {
      const out = [];
      const wantOption = selector === '[role="option"]';
      const walk = (x) => {
        for (const child of x.children) {
          if (wantOption && child.attrs.role === "option") out.push(child);
          walk(child);
        }
      };
      walk(n);
      return out;
    },
    querySelector(selector) { return n.querySelectorAll(selector)[0] || null; },
  };
  return n;
}

function select() {
  let value = "";
  return {
    options: [],
    get value() { return value; },
    set value(v) { value = this.options.some((o) => o.value === v) ? v : ""; },
    set innerHTML(_) { this.options = []; value = ""; },
    appendChild(o) { this.options.push(o); },
    get selectedIndex() { return this.options.findIndex((o) => o.value === value); },
  };
}

function harness() {
  const doc = { activeElement: null, createElement: (tag) => node(tag, doc) };
  const elements = {
    omodel: select(),
    omodelbtn: node("button", doc),
    omodelbtnlabel: node("span", doc),
    modelpick: node("div", doc),
    modelpickscrim: node("div", doc),
    modelpickpanel: node("section", doc),
    modelpickclose: node("button", doc),
    modelpicklist: node("div", doc),
  };
  elements.modelpick.hidden = true;
  let picks = 0;
  const context = vm.createContext({
    $: (id) => elements[id],
    document: doc,
    savePick: () => { picks += 1; },
  });
  vm.runInContext(between("let modelSheetReturnFocus", "// Skills are pending next-turn choices"), context);
  const add = (value, text, title = "") => {
    const option = { value, textContent: text, title };
    elements.omodel.appendChild(option);
    return option;
  };
  add("", "model: default", "model: server default (deepseek-v4-flash)");
  add("deepseek-v4-flash", "deepseek-v4-flash");
  add("gpt-5.6-sol", "gpt-5.6-sol");
  context.syncModelTrigger();
  return {
    context, elements, doc,
    picks: () => picks,
    rows() { return elements.modelpicklist.querySelectorAll('[role="option"]'); },
  };
}

test("trigger mirrors the committed model and starts on the default", () => {
  const h = harness();
  assert.equal(h.elements.omodelbtn.disabled, false);
  assert.equal(h.elements.omodelbtnlabel.textContent, "model: default");
  assert.equal(h.elements.omodelbtn.attrs["aria-label"], "model override: default");
  h.elements.omodel.value = "deepseek-v4-flash";
  h.context.syncModelTrigger();
  assert.equal(h.elements.omodelbtnlabel.textContent, "model: deepseek-v4-flash");
  assert.equal(h.elements.omodelbtn.attrs["aria-label"], "model override: deepseek-v4-flash");
});

test("opening renders a themed row for every option and marks the current one", () => {
  const h = harness();
  h.context.openModelSheet();
  assert.equal(h.elements.modelpick.hidden, false);
  assert.equal(h.elements.omodelbtn.attrs["aria-expanded"], "true");
  const rows = h.rows();
  assert.equal(rows.length, 3);
  assert.equal(rows[0].attrs["aria-selected"], "true");
  assert.match(rows[0].textContent, /server default/);
  assert.match(rows[0].textContent, /deepseek-v4-flash/);
  assert.equal(rows[1].attrs["aria-selected"], "false");
  assert.equal(rows[2].attrs["aria-selected"], "false");
  assert.equal(h.doc.activeElement, rows[0]);
});

test("choosing a model writes the hidden select, saves, and closes", () => {
  const h = harness();
  h.context.openModelSheet();
  h.rows()[1].onclick();
  assert.equal(h.elements.omodel.value, "deepseek-v4-flash");
  assert.equal(h.picks(), 1);
  assert.equal(h.elements.modelpick.hidden, true);
  assert.equal(h.elements.omodelbtn.attrs["aria-expanded"], "false");
  assert.equal(h.elements.omodelbtnlabel.textContent, "model: deepseek-v4-flash");
  assert.equal(h.doc.activeElement, h.elements.omodelbtn);
});

test("the default row clears the override", () => {
  const h = harness();
  h.elements.omodel.value = "gpt-5.6-sol";
  h.context.syncModelTrigger();
  h.context.openModelSheet();
  h.rows()[0].onclick();
  assert.equal(h.elements.omodel.value, "");
  assert.equal(h.picks(), 1);
  assert.equal(h.elements.omodelbtnlabel.textContent, "model: default");
});

test("a row removed by an info refresh cannot select a missing model", () => {
  const h = harness();
  h.elements.omodel.value = "gpt-5.6-sol";
  h.context.syncModelTrigger();
  h.context.openModelSheet();
  const rows = h.rows();
  h.elements.omodel.options = h.elements.omodel.options.filter((o) => o.value !== "gpt-5.6-sol");
  rows[2].onclick();
  assert.equal(h.picks(), 0);
  assert.equal(h.elements.modelpick.hidden, true);
});

test("arrow keys move the sheet focus and Escape closes it", () => {
  const h = harness();
  h.context.openModelSheet();
  const rows = h.rows();
  const event = (key) => ({ key, preventDefault() { this.prevented = true; } });
  h.context.modelSheetKeydown(event("Tab"));
  assert.equal(h.doc.activeElement, rows[1]);
  h.context.modelSheetKeydown(event("ArrowDown"));
  assert.equal(h.doc.activeElement, rows[2]);
  h.context.modelSheetKeydown(event("End"));
  assert.equal(h.doc.activeElement, rows[2]);
  h.context.modelSheetKeydown(event("Escape"));
  assert.equal(h.elements.modelpick.hidden, true);
  assert.equal(h.doc.activeElement, h.elements.omodelbtn);
});

test("markup keeps the native select as the value store", () => {
  assert.match(html, /<select id="omodel" aria-label="model override" hidden><\/select>/);
  assert.match(html, /id="omodelbtn"[\s\S]*?aria-haspopup="listbox"[\s\S]*?aria-controls="modelpicklist"/);
  assert.match(html, /id="modelpickpanel" role="dialog" aria-modal="true"/);
  assert.match(html, /id="modelpicklist" role="listbox"/);
});
