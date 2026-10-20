// Run with: node --test src/bin/ai/serve/app_typography.test.cjs
// Typography contract of the embedded client: the default reading weight is
// one step above Regular, the secondary chrome stays Regular, and the reading
// surface carries a hair of tracking that code blocks opt out of.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const rawStyle = html.match(/<style>([\s\S]*?)<\/style>/)[1];
// Comments carry prose about the rules; they must not be parsed as rules.
const style = rawStyle.replace(/\/\*[\s\S]*?\*\//g, "");

// Every rule that targets the body, in source order.
function bodyRules() {
  const out = [];
  const re = /(^|\n)([^\n{}]*\bbody\b[^{]*)\{([^}]*)\}/g;
  let m;
  while ((m = re.exec(style)) !== null) out.push({ selector: m[2].trim(), body: m[3], at: m.index });
  return out;
}

test("the default reading weight is a token, and it outranks the font shorthand", () => {
  assert.match(style, /--fw-text:\s*450/, "the reading weight token is missing");
  assert.match(style, /--fw-meta:\s*400/, "the secondary weight token is missing");

  const rules = bodyRules();
  const shorthand = rules.find((r) => /font:/.test(r.body));
  const weight = rules.find((r) => /font-weight:\s*var\(--fw-text\)/.test(r.body));
  assert.ok(shorthand, "the body no longer carries a font shorthand");
  assert.ok(weight, "the body does not use the reading weight token");
  // The shorthand resets font-weight to normal, so the weight rule has to come
  // after it in the stylesheet.
  assert.ok(weight.at > shorthand.at, "the reading weight is declared before the font shorthand that resets it");
});

test("the secondary chrome keeps the Regular weight", () => {
  const rule = style.match(/([^{}]*)\{([^}]*font-weight:\s*var\(--fw-meta\)[^}]*)\}/);
  assert.ok(rule, "no rule assigns the secondary weight");
  for (const selector of ["#info", "#sessmeta", "#turnstatus", ".msg details", ".msg .prog"]) {
    assert.ok(rule[1].includes(selector), "the secondary weight misses " + selector);
  }
});

test("the reading surface tracks slightly and code keeps its own metrics", () => {
  const msg = style.match(/\.msg\s*\{([^}]*)\}/);
  assert.ok(msg, "the message bubble rule is missing");
  assert.match(msg[1], /letter-spacing:\s*\.01em/, "the reading surface has no tracking");
  assert.match(
    style,
    /\.msg\.md pre,\s*\.msg\.md code\s*\{[^}]*letter-spacing:\s*normal/,
    "code blocks inherit the reading tracking",
  );
});
