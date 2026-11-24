"""Run against a built mcp_browser with MCP_BROWSER_CHROME set to Chrome."""

import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
import unittest
from urllib.parse import quote


class BrowserRegressions(unittest.TestCase):
    RPC_TIMEOUT_SECONDS = 30

    @classmethod
    def setUpClass(cls):
        profile = tempfile.TemporaryDirectory(prefix="mcp-browser-cdp-")
        cls.addClassCleanup(profile.cleanup)
        tmp = profile.name
        runtime_tmp = Path(tmp) / "tmp"
        runtime_tmp.mkdir()
        env = os.environ.copy()
        env.pop("MCP_BROWSER_WS_URL", None)
        env.pop("MCP_BROWSER_OP_TIMEOUT_MS", None)
        env.update(
            MCP_BROWSER_DRIVER="cdp",
            MCP_BROWSER_HEADLESS="1",
            MCP_BROWSER_USER_DATA_DIR=f"{tmp}/profile",
            TMPDIR=str(runtime_tmp),
            XDG_CONFIG_HOME=f"{tmp}/config",
            XDG_CACHE_HOME=f"{tmp}/cache",
        )
        binary = Path(__file__).resolve().parents[3] / "target/debug/mcp_browser"
        cls.server = subprocess.Popen(
            [os.environ.get("MCP_BROWSER_TEST_BIN", str(binary))],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        cls.addClassCleanup(cls.stop)
        os.set_blocking(cls.server.stdin.fileno(), False)
        os.set_blocking(cls.server.stdout.fileno(), False)
        cls.output = bytearray()
        cls.sequence = 0
        cls.rpc("initialize")
        cls.rpc("notifications/initialized", notify=True)
        tools = {tool["name"]: tool["inputSchema"] for tool in cls.rpc("tools/list")["result"]["tools"]}
        assert len(tools) == 16, tools
        assert set(tools["click"]["properties"]) == {"selector", "ref"}, tools["click"]

    @classmethod
    def stop(cls):
        cls.server.stdin.close()
        try:
            cls.server.wait(timeout=8)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait(timeout=8)
        cls.server.stdout.close()

    @classmethod
    def rpc(cls, method, params=None, *, notify=False):
        cls.sequence += 1
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            message["id"] = cls.sequence
        deadline = time.monotonic() + cls.RPC_TIMEOUT_SECONDS
        try:
            cls.write_request((json.dumps(message) + "\n").encode("utf-8"), deadline)
            if notify:
                return None
            reply = json.loads(cls.read_reply(deadline))
        except TimeoutError as exc:
            cls.server.kill()
            raise TimeoutError(f"MCP {method} id={cls.sequence} exceeded the stdio deadline") from exc
        assert reply["id"] == cls.sequence, reply
        return reply

    @classmethod
    def write_request(cls, data, deadline):
        fd = cls.server.stdin.fileno()
        data = memoryview(data)
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise TimeoutError("waiting for writable MCP stdin")
            try:
                data = data[os.write(fd, data):]
            except BlockingIOError:
                continue

    @classmethod
    def read_reply(cls, deadline):
        fd = cls.server.stdout.fileno()
        # Read chunks, not readline(): a partial JSON line cannot extend the deadline.
        while b"\n" not in cls.output:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError("waiting for a complete MCP reply")
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise EOFError("MCP server closed stdout before replying")
            cls.output.extend(chunk)
        line, _, cls.output = cls.output.partition(b"\n")
        return line

    def call(self, name, **arguments):
        reply = self.rpc("tools/call", {"name": name, "arguments": arguments})
        self.assertIn("result", reply, reply)
        return reply["result"]["content"][0]["text"]

    def js(self, expression):
        return json.loads(self.call("evaluate_js", expression=expression))

    def page(self, markup):
        self.call("navigate", url="data:text/html;charset=utf-8," + quote("<!doctype html>" + markup))

    def snapshot_field(self, name):
        return next(item for item in json.loads(self.call("snapshot"))["elements"] if item["name"] == name)

    def click_error(self, selector, reason):
        reply = self.rpc("tools/call", {"name": "click", "arguments": {"selector": selector}})
        self.assertIn("error", reply, reply)
        self.assertEqual(reply["error"]["code"], -32000, reply)
        self.assertIn(reason, reply["error"]["message"], reply)

    def test_password_text_toggle(self):
        self.page('''<input id="field" aria-label="Toggle field">
          <script>
            let protectedField = false;
            function render() {
              field.type = protectedField ? 'password' : 'text';
              field.value = protectedField ? 'hidden-fixture-value' : 'public-value';
            }
            render();
          </script>''')
        self.assertEqual(self.snapshot_field("Toggle field")["value"], "public-value")
        self.js("protectedField = true; render()")
        self.assertNotIn("value", self.snapshot_field("Toggle field"))
        self.js("window.toggle = setInterval(() => { protectedField = !protectedField; render(); }, 4)")
        public = redacted = 0
        unexpected = []
        try:
            for _ in range(90):
                sample = self.snapshot_field("Toggle field")
                if "value" not in sample:
                    redacted += 1
                elif sample["value"] == "public-value":
                    public += 1
                else:
                    unexpected.append(sample["value"])
        finally:
            self.js("clearInterval(window.toggle)")
        print(f"Toggle snapshots: public={public}, omitted={redacted}, unexpected={len(unexpected)}", flush=True)
        self.assertFalse(unexpected, "a snapshot returned a password-state or stale AX value")
        self.assertGreater(public, 0)
        self.assertGreater(redacted, 0)

    def test_other_live_values(self):
        self.page('''<textarea id="message" aria-label="Message">default</textarea>
          <input aria-label="Amount" type="range" min="0" max="8" value="3">
          <select aria-label="Choice"><option>Unchosen</option>
            <optgroup label="Group"><option selected label="Selected label">Option text</option></optgroup>
          </select><script>message.value = 'live text';</script>''')
        self.assertEqual(self.snapshot_field("Message")["value"], "live text")
        self.assertEqual(self.snapshot_field("Amount")["value"], 3)
        self.assertEqual(self.snapshot_field("Choice")["value"], "Selected label")

    def test_zero_area_target_never_hits_parent(self):
        self.page('''<div id="parent" style="position:relative;width:180px;height:48px;background:grey">
          <div id="zero" style="position:absolute;left:10px;top:10px;width:80px;height:0"></div></div>
          <script>window.presses = 0; parentEl = document.querySelector('#parent');
            parentEl.addEventListener('mousedown', () => ++window.presses);
          </script>''')
        self.click_error("#zero", "zero-area geometry")
        self.assertEqual(self.js("window.presses"), 0)

    def test_shadow_cover_over_slotted_light_dom(self):
        for mode in ("open", "closed"):
            with self.subTest(mode=mode):
                self.check_shadow_cover(mode)

    def check_shadow_cover(self, mode):
        self.page('''<div id="host"><button id="slotted" slot="content">Target</button></div>
          <script>
            window.targetClicks = window.coverClicks = 0;
            const root = host.attachShadow({mode: 'SHADOW_MODE'});
            root.innerHTML = `<style>
              .frame {position:relative; width:160px; height:40px}
              ::slotted(button) {width:160px; height:40px}
              #slot-cover {position:absolute; inset:0; background:white}
            </style><div class="frame"><slot name="content"></slot><div id="slot-cover"></div></div>`;
            root.querySelector('#slot-cover').onclick = () => ++window.coverClicks;
            slotted.onclick = () => ++window.targetClicks;
          </script>'''.replace("SHADOW_MODE", mode))
        self.click_error("#slotted", "div#slot-cover")
        self.assertEqual(self.js("[targetClicks, coverClicks]"), [0, 0])
        self.js("root.querySelector('#slot-cover').remove()")
        self.assertIn('"hit_test":"clear"', self.call("click", selector="#slotted"))
        self.assertEqual(self.js("targetClicks"), 1)

    def test_fractional_closed_shadow_uses_same_hit_and_dispatch_point(self):
        self.page('''<style>body {margin:0}</style>
          <div id="host"><button id="slotted" slot="content">Target</button></div>
          <script>
            const root = host.attachShadow({mode: 'closed'});
            root.innerHTML = `<style>
              .frame {position:relative; width:300px; height:120px}
              ::slotted(button) {position:absolute; left:100.25px; top:40px; width:80px; height:40px}
              #fractional-cover {position:absolute; left:141px; top:40px; width:40px; height:40px; background:white}
            </style><div class="frame"><slot name="content"></slot><div id="fractional-cover"></div></div>`;
            window.inputs = [];
            for (const node of [slotted, root.querySelector('#fractional-cover')]) {
              for (const event of ['mousedown', 'click']) {
                node.addEventListener(event, e => inputs.push([node.id, e.type, e.isTrusted]));
              }
            }
          </script>''')
        self.assertEqual(self.js('''(() => {
          const rect = slotted.getBoundingClientRect();
          return [rect.x + rect.width / 2, root.elementFromPoint(140, 60).id, root.elementFromPoint(140.25, 60).id];
        })()'''), [140.25, "slotted", "fractional-cover"])
        self.assertIn('"hit_test":"clear"', self.call("click", selector="#slotted"))
        self.assertEqual(self.js("inputs"), [["slotted", "mousedown", True], ["slotted", "click", True]])

    def test_hover_loses_viewport_without_detaching(self):
        self.page('''<div id="receiver" style="position:fixed;left:350px;top:220px;width:240px;height:80px;background:grey">
          <button id="moving" style="width:180px;height:60px">Move into inactive document</button></div>
          <script>
            const target = moving;
            const inactive = document.implementation.createHTMLDocument('Inactive');
            window.presses = window.clicks = 0;
            target.onmousemove = e => { if (e.isTrusted) inactive.body.append(target); };
            receiver.onmousedown = e => { if (e.isTrusted) ++presses; };
            receiver.onclick = e => { if (e.isTrusted) ++clicks; };
          </script>''')
        self.click_error("#moving", "no usable layout viewport")
        self.assertEqual(self.js("[target.isConnected, target.ownerDocument.defaultView, target.getClientRects().length, presses, clicks]"),
                         [True, None, 0, 0, 0])

    def test_hover_adopts_target_into_same_origin_iframe(self):
        self.page('''<iframe id="destination" srcdoc="<!doctype html><p id='frame-root'>Destination</p>"
            style="position:fixed;left:8px;top:8px;width:260px;height:120px"></iframe>
          <div id="receiver" style="position:fixed;left:350px;top:220px;width:240px;height:80px;background:grey">
            <button id="moving" style="width:180px;height:60px">Move into iframe</button></div>
          <script>
            const target = moving;
            window.inputs = [];
            target.onmousemove = e => {
              if (e.isTrusted) destination.contentDocument.body.append(target);
            };
            receiver.onmousedown = e => inputs.push([e.type, e.target.id, e.isTrusted]);
            receiver.onclick = e => inputs.push([e.type, e.target.id, e.isTrusted]);
          </script>''')
        self.assertTrue(self.js("!!destination.contentDocument.querySelector('#frame-root')"))
        self.click_error("#moving", "changed document or viewport")
        self.assertEqual(self.js('''[
          target.ownerDocument === destination.contentDocument,
          target.ownerDocument.defaultView === destination.contentWindow,
          target.isConnected, target.getClientRects().length, inputs
        ]'''), [True, True, True, 1, []])

    def test_pointer_events_none_never_presses_ancestor(self):
        self.page('''<div id="receiver" style="width:210px;height:80px;background:grey">
          <button id="passive" style="width:180px;height:60px;pointer-events:none">Pass-through button</button></div>
          <script>
            window.inputs = [];
            receiver.onmousedown = e => inputs.push([e.type, e.target.id, e.isTrusted]);
            receiver.onclick = e => inputs.push([e.type, e.target.id, e.isTrusted]);
          </script>''')
        self.assertEqual(self.js('''(() => {
          const rect = passive.getBoundingClientRect();
          return document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2).id;
        })()'''), "receiver")
        self.click_error("#passive", "pointer-events: none")
        self.assertEqual(self.js("inputs"), [])

    def test_nested_selected_option_label(self):
        self.page('''<select aria-label="Nested choice"><option id="nested" selected></option></select>
          <script>
            const span = document.createElement('span');
            span.textContent = 'Nested';
            const inertScript = document.createElement('script');
            inertScript.type = 'text/plain';
            inertScript.textContent = ' NOT LABEL';
            nested.append('Prefix ', span, inertScript);
          </script>''')
        self.assertEqual(self.js("nested.label"), "Prefix Nested")
        self.assertEqual(self.snapshot_field("Nested choice")["value"], "Prefix Nested")
        # Chrome retains style/noscript text and direct template children, but
        # not text inside template.content or SVG script elements.
        native = self.js('''(() => {
          const style = document.createElement('style');
          style.textContent = ' STYLE';
          const noscript = document.createElement('noscript');
          noscript.textContent = ' NOSCRIPT';
          const template = document.createElement('template');
          template.content.append(' NOT TEMPLATE LABEL');
          template.append(' TEMPLATE CHILD');
          const svgScript = document.createElementNS('http://www.w3.org/2000/svg', 'script');
          svgScript.setAttribute('type', 'text/plain');
          svgScript.textContent = ' NOT SVG LABEL';
          nested.append(style, noscript, template, svgScript);
          return nested.label;
        })()''')
        self.assertEqual(native, "Prefix Nested STYLE NOSCRIPT TEMPLATE CHILD")
        self.assertEqual(self.snapshot_field("Nested choice")["value"], native)
        self.js("nested.label = '  Explicit label  '")
        self.assertEqual(self.snapshot_field("Nested choice")["value"], "Explicit label")

    def test_target_removed_between_probe_and_press_reports_unknown(self):
        self.page('''<div id="receiver" style="position:fixed;left:210px;top:130px;width:240px;height:80px;background:grey">
          <button id="leaving" style="width:180px;height:60px">Remove after probe</button></div>
          <script>
            const target = leaving;
            const original = Element.prototype.getClientRects;
            Element.prototype.getClientRects = function() {
              const rects = original.call(this);
              if (this === target) queueMicrotask(() => target.remove());
              return rects;
            };
            window.inputs = [];
            receiver.onmousedown = e => inputs.push([e.type, e.target.id, e.isTrusted]);
            receiver.onclick = e => inputs.push([e.type, e.target.id, e.isTrusted]);
          </script>''')
        result = self.call("click", selector="#leaving")
        verdict = next(json.loads(line) for line in result.splitlines() if line.startswith("{"))
        self.assertEqual(verdict["hit_test"], "clear")
        self.assertEqual(verdict["status"], "unknown")
        self.assertIn("detached", verdict["reason"])
        self.assertIsNone(verdict["evidence"])
        self.assertEqual(self.js("inputs"), [["mousedown", "receiver", True], ["click", "receiver", True]])

    def test_detaches_after_quad_read_on_native_hover(self):
        self.page('''<div id="receiver" style="position:fixed;left:300px;top:180px;width:240px;height:80px;background:grey">
          <button id="leaving" style="width:180px;height:60px">Remove on hover</button></div>
          <script>
            window.detached = false; window.presses = 0; window.clicks = 0;
            receiver.addEventListener('mousedown', () => ++window.presses);
            receiver.addEventListener('click', () => ++window.clicks);
            leaving.addEventListener('mousemove', e => {
              if (e.isTrusted) { window.detached = true; leaving.remove(); }
            });
          </script>''')
        self.click_error("#leaving", "target detached")
        self.assertEqual(self.js("[window.detached, window.presses, window.clicks]"), [True, 0, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
