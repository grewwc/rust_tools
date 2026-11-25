"""Synthetic Chrome profile round trip. Set MCP_BROWSER_CHROME to a Chrome executable."""

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import select
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest


class FixturePage(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<!doctype html><title>Profile test fixture</title>")
        if self.path == "/signin":
            self.wfile.write(b"""<script>
                document.cookie = 'fixture_login=synthetic; Max-Age=3600; Path=/';
                localStorage.setItem('fixture_login', 'synthetic-storage');
                fetch('/fixture-saved');
            </script>""")
        elif self.path == "/fixture-saved":
            self.server.saved.set()

    def log_message(self, *args):
        pass


class ProfileSync(unittest.TestCase):
    def setUp(self):
        workspace_bin = Path(__file__).resolve().parents[3] / "target/debug/mcp_browser"
        self.binary = os.environ.get("MCP_BROWSER_TEST_BIN", str(workspace_bin))
        directory = tempfile.TemporaryDirectory(prefix="mb-", dir="/tmp" if sys.platform == "linux" else None)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        runtime_tmp = self.root / "tmp"
        runtime_tmp.mkdir()
        self.env = os.environ.copy()
        self.env.pop("CHROME_USER_DATA_DIR", None)
        for name in ("OP_TIMEOUT_MS", "PROFILE_DIRECTORY", "PROFILE_KEYSTORE", "USER_DATA_DIR"):
            self.env.pop(f"MCP_BROWSER_{name}", None)
        self.env.update(
            MCP_BROWSER_DRIVER="cdp",
            MCP_BROWSER_HEADLESS="1",
            MCP_BROWSER_WS_URL=" \t",  # Must self-launch, even with a blank WS override.
            TMPDIR=str(runtime_tmp),
            HOME=str(self.root),
            CHROME_CONFIG_HOME=str(self.root / "config"),
            XDG_CONFIG_HOME=str(self.root / "config"),
            XDG_CACHE_HOME=str(self.root / "cache"),
        )

    @contextlib.contextmanager
    def fixture_page(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), FixturePage) as server:
            server.saved = threading.Event()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}/", server.saved
            finally:
                server.shutdown()
                thread.join()

    def stdio(self, profile, commands):
        kwargs = {"umask": 0o022} if os.name == "posix" else {}
        env = self.env.copy()
        if profile is not None:
            env["MCP_BROWSER_USER_DATA_DIR"] = str(profile)
        completed = subprocess.run(
            [self.binary], env=env,
            input="".join(json.dumps(dict(command, jsonrpc="2.0")) + "\n" for command in commands),
            text=True, capture_output=True, timeout=75, **kwargs,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return [json.loads(line) for line in completed.stdout.splitlines()], completed.stderr

    @contextlib.contextmanager
    def live_stdio(self, profile):
        env = dict(self.env, MCP_BROWSER_USER_DATA_DIR=str(profile))
        with subprocess.Popen(
            [self.binary], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, umask=0o022,
        ) as server:
            seq = 0

            def rpc(method, **params):
                nonlocal seq
                seq += 1
                request = dict(jsonrpc="2.0", id=seq, method=method, params=params)
                server.stdin.write((json.dumps(request) + "\n").encode())
                deadline = time.monotonic() + 35
                line = bytearray()
                while not line.endswith(b"\n"):
                    remaining = deadline - time.monotonic()
                    self.assertTrue(remaining > 0 and select.select([server.stdout], [], [], remaining)[0],
                                    f"timed out waiting for {method}")
                    chunk = os.read(server.stdout.fileno(), 65536)
                    self.assertTrue(chunk, f"server closed stdout before replying to {method}")
                    line.extend(chunk)
                reply = json.loads(line)
                self.assertEqual(reply["id"], seq)
                return reply

            try:
                yield rpc
            finally:
                try:
                    _, stderr = server.communicate(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.communicate(timeout=10)
                    raise
            self.assertEqual(server.returncode, 0, stderr.decode(errors="replace"))

    def chrome_round_trip(self, profile, url, script):
        commands = [
            {"method": "initialize", "id": 1},
            {"method": "notifications/initialized"},
            {"method": "tools/list", "id": 2},
            {"method": "tools/call", "id": 3, "params": {
                "name": "navigate", "arguments": {"url": url}}},
            {"method": "tools/call", "id": 4, "params": {
                "name": "evaluate_js", "arguments": {"expression": script}}},
        ]
        # EOF exercises the same graceful shutdown hook used by a normal stdio session.
        replies, stderr = self.stdio(profile, commands)
        self.assertEqual([reply["id"] for reply in replies], [1, 2, 3, 4])
        self.assertEqual(len(replies[1]["result"]["tools"]), 16, stderr)
        for reply in replies:
            self.assertIn("result", reply, reply)
        return json.loads(replies[3]["result"]["content"][0]["text"])

    def sync_profile(self, source, dest, *options, profile="Default"):
        return subprocess.run(
            [self.binary, "--sync-profile", "--from", str(source), "--profile", profile,
             "--profile-dir", str(dest), "--keychain", "mock", *options],
            env=self.env, text=True, capture_output=True, timeout=35,
        )

    @unittest.skipUnless(os.name == "posix", "Unix pipe selection and link fixtures")
    def test_serve_rechecks_profile_before_launch_but_not_on_session_reuse(self):
        source = self.root / "source"
        (source / "Default").mkdir(parents=True)
        (source / "Local State").write_text("{}", encoding="utf-8")
        dest = self.root / "dedicated"
        copied = self.sync_profile(source, dest)
        self.assertEqual(copied.returncode, 0, copied.stderr)
        selected = dest / "Default"
        outside = self.root / "external-file"
        outside.write_text("synthetic", encoding="utf-8")
        alias = self.root / "dedicated-alias"
        alias.symlink_to(dest, target_is_directory=True)

        with self.live_stdio(alias) as rpc:
            def call(name, **arguments):
                return rpc("tools/call", name=name, arguments=arguments)

            def navigate():
                return call("navigate", url="about:blank")

            def refuses_launch(reason):
                reply = navigate()
                self.assertIn(reason, reply.get("error", {}).get("message", ""), reply)

            self.assertEqual(len(rpc("tools/list")["result"]["tools"]), 16)
            # Make each change after the server has finished its read-only preflight.
            moved = self.root / "original-profile"
            selected.rename(moved)
            selected.symlink_to(moved, target_is_directory=True)
            try:
                refuses_launch("selected Chrome profile must be a real directory")
            finally:
                selected.unlink()
                moved.rename(selected)

            network = selected / "Network"
            network.mkdir()
            for name, symlink, reason in (
                ("Cookies", True, "symbolic link"),
                ("Cookies", False, "multiple hard links"),
                ("Cookies-wal", False, "multiple hard links"),
            ):
                with self.subTest(name=name, symlink=symlink):
                    link = network / name
                    if symlink:
                        link.symlink_to(outside)
                    else:
                        os.link(outside, link)
                    try:
                        refuses_launch(reason)
                    finally:
                        link.unlink()

            reply = navigate()
            self.assertIn("result", reply, reply)
            reply = call("evaluate_js", expression="globalThis.profileReuseFixture = true")
            self.assertIn("result", reply, reply)
            link = selected / "validation-fixture"
            os.link(outside, link)
            try:
                reply = call("evaluate_js", expression="globalThis.profileReuseFixture")
                self.assertIn("result", reply, reply)
                self.assertIs(json.loads(reply["result"]["content"][0]["text"]), True)
                reply = call("close_browser")
                self.assertIn("result", reply, reply)
                refuses_launch("multiple hard links")
            finally:
                link.unlink()

    def test_gc_preserves_nested_persistent_profile(self):
        runtime = Path(self.env["TMPDIR"])
        def stale_root(pid):
            path = runtime / f"mcp_browser-profile-{pid}"
            path.mkdir(mode=0o700)
            marker = path / ".mcp_browser_temp_profile"
            marker.write_text(f"{pid}\n", encoding="ascii")
            marker.chmod(0o600)
            return path
        ancestor = stale_root(2147483644)
        cleanup = stale_root(2147483643)
        cleanup_lock = cleanup.with_name(f".{cleanup.name}.mcp_browser.lock")
        cleanup_lock.touch(mode=0o600)
        nested_marker_root = stale_root(2147483642)
        unrelated = runtime / "mcp_browser-profile-2147483641"
        unrelated.mkdir(mode=0o700)
        persistent = ancestor / "persist"
        for directory in (persistent, nested_marker_root / "persist"):
            directory.mkdir(mode=0o700)
            marker = directory / ".mcp_browser_profile.json"
            marker.write_text(json.dumps({"version": 1, "profile_directory": "Default",
                "keychain": "mock", "source": "fixture", "synced_at_unix": 1}), encoding="utf-8")
            marker.chmod(0o600)
        sentinel = persistent / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        self.stdio(persistent, [{"method": "initialize", "id": 1}, {"method": "tools/list", "id": 2}])
        self.assertTrue(sentinel.is_file(), "startup GC deleted the explicitly configured profile")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertTrue((nested_marker_root / "persist/.mcp_browser_profile.json").is_file(),
                        "GC deleted a nested marker outside the configured directory")
        self.assertFalse(cleanup.exists(), "GC did not reclaim an owned stale temporary root")
        self.assertNotIn(cleanup_lock, runtime.iterdir(), "GC left the reclaimed profile's sibling lock")
        self.assertTrue(unrelated.is_dir(), "GC deleted an unrelated directory based only on its name")

    @unittest.skipUnless(sys.platform == "linux", "Linux browser data location fixture")
    def test_serve_rejects_ordinary_browser_directory(self):
        config = Path(self.env["XDG_CONFIG_HOME"])
        ordinary = config / "google-chrome"
        child = ordinary / "Default"
        child.mkdir(mode=0o700, parents=True)
        commands = [{"method": "initialize", "id": 1}, {"method": "tools/list", "id": 2},
                    {"method": "tools/call", "id": 3, "params": {
                        "name": "navigate", "arguments": {"url": "about:blank"}}}]
        for destination in (ordinary, child, config):
            with self.subTest(destination=destination.name):
                before = list(destination.iterdir())
                replies, stderr = self.stdio(destination, commands)
                self.assertCountEqual(list(destination.iterdir()), before,
                                      "serve wrote into a directory overlapping ordinary browser data")
                self.assertEqual(replies[1]["result"]["tools"], [])
                self.assertIn("ordinary browser data", stderr)
                self.assertIn("ordinary browser data", replies[2]["error"]["message"])

    @unittest.skipUnless(os.name == "posix", "Unix profile permissions")
    def test_env_created_profile_is_private(self):
        dest = self.root / "env-only"
        with self.fixture_page() as (url, _):
            self.assertEqual(self.chrome_round_trip(dest, url, """(() => {
                document.cookie = 'private_fixture=synthetic; Max-Age=3600; Path=/';
                localStorage.setItem('private_fixture', 'storage');
                return document.cookie;
            })()"""), "private_fixture=synthetic")
            self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o700)
            cookies = [path for path in (dest / "Default/Cookies", dest / "Default/Network/Cookies") if path.is_file()]
            self.assertTrue(cookies, "Chrome did not save the synthetic cookie store")
            for path in [dest / ".mcp_browser_profile.json", dest / "Local State", *cookies]:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path.name)
            self.assertEqual(self.chrome_round_trip(dest, url, "localStorage.getItem('private_fixture')"), "storage")
            lock = dest.with_name(f".{dest.name}.mcp_browser.lock")
            self.assertTrue(lock.is_file(), "the persistent profile lock must survive shutdown")

    def test_serve_rejects_unmarked_custom_profile(self):
        dest = self.root / "unmarked"
        dest.mkdir(mode=0o700)
        sentinel = dest / "Local State"
        sentinel.write_text("synthetic unmarked data", encoding="utf-8")
        replies, stderr = self.stdio(dest, [{"method": "tools/list", "id": 1}])
        self.assertEqual(replies[0]["result"]["tools"], [])
        self.assertIn("has no MCP marker", stderr)
        self.assertIn("--sync-profile or --login", stderr)
        self.assertEqual(list(dest.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "synthetic unmarked data")

    @unittest.skipUnless(os.name == "posix", "Unix profile permissions")
    def test_serve_rejects_existing_world_readable_profile(self):
        dest = self.root / "public"
        dest.mkdir(mode=0o755)
        replies, stderr = self.stdio(dest, [{"method": "tools/list", "id": 1}])
        self.assertEqual(replies[0]["result"]["tools"], [])
        self.assertIn("Unix mode 0700", stderr)
        self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o755, "preflight must not chmod the user directory")
        self.assertFalse(list(dest.iterdir()))

    def test_default_temporary_profile_is_removed_on_eof(self):
        replies, stderr = self.stdio(None, [
            {"method": "tools/list", "id": 1},
            {"method": "tools/call", "id": 2, "params": {
                "name": "navigate", "arguments": {"url": "about:blank"}}},
        ])
        self.assertEqual(len(replies[0]["result"]["tools"]), 16, stderr)
        self.assertIn("result", replies[1], replies[1])
        runtime = Path(self.env["TMPDIR"])
        self.assertEqual(list(runtime.glob("mcp_browser-profile-*")), [])
        self.assertEqual(list(runtime.glob(".mcp_browser-profile-*.mcp_browser.lock")), [])

    @unittest.skipUnless(os.name == "posix", "Unix symlink fixtures")
    def test_serve_rejects_effective_profile_and_sensitive_file_links(self):
        source = self.root / "source"
        (source / "Default").mkdir(parents=True)
        (source / "Local State").write_text("{}", encoding="utf-8")
        dest = self.root / "managed"
        copied = self.sync_profile(source, dest)
        self.assertEqual(copied.returncode, 0, copied.stderr)
        alias = self.root / "alias"
        alias.symlink_to(dest, target_is_directory=True)
        commands = [{"method": "tools/list", "id": 1}]

        override = dest / "Profile 7"
        override.symlink_to(source / "Default", target_is_directory=True)
        self.env["MCP_BROWSER_PROFILE_DIRECTORY"] = "Profile 7"
        replies, stderr = self.stdio(dest, commands)
        self.assertEqual(replies[0]["result"]["tools"], [])
        self.assertIn("selected Chrome profile must be a real directory", stderr)
        self.env.pop("MCP_BROWSER_PROFILE_DIRECTORY")

        (dest / "Default/Network").mkdir()
        cookie = dest / "Default/Network/Cookies"
        cookie.symlink_to(source / "Local State")
        replies, stderr = self.stdio(alias, commands)
        self.assertEqual(replies[0]["result"]["tools"], [])
        self.assertIn("symbolic link", stderr)
        cookie.unlink()

        replies, stderr = self.stdio(alias, commands)
        self.assertEqual(len(replies[0]["result"]["tools"]), 16, stderr)
        self.assertEqual((source / "Local State").read_text(encoding="utf-8"), "{}")

    @unittest.skipUnless(os.name == "posix", "Unix hard links and shell test executable")
    def test_serve_rejects_hard_linked_profile_files_without_starting_chrome(self):
        source = self.root / "source"
        (source / "Default").mkdir(parents=True)
        outside = source / "Local State"
        outside.write_text("{}", encoding="utf-8")
        dest = self.root / "managed"
        copied = self.sync_profile(source, dest)
        self.assertEqual(copied.returncode, 0, copied.stderr)

        fake = self.root / "fake-chrome"
        started = self.root / "started"
        fake.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo synthetic; exit 0; fi\n'
                        f'touch {shlex.quote(str(started))}\nexit 5\n', encoding="utf-8")
        fake.chmod(0o700)
        self.env["MCP_BROWSER_CHROME"] = str(fake)
        tools = {"method": "tools/list", "id": 1}
        navigate = {"method": "tools/call", "id": 2, "params": {
            "name": "navigate", "arguments": {"url": "about:blank"}}}

        for rel in ("Local State", "Default/Cookies", "Default/Network/Cookies-wal",
                    "Default/Login Data", "Default/Web Data-journal", "Default/IndexedDB/fixture"):
            with self.subTest(path=rel):
                entry = dest / rel
                entry.parent.mkdir(parents=True, exist_ok=True)
                entry.unlink(missing_ok=True)
                os.link(outside, entry)
                try:
                    self.assertTrue(entry.samefile(outside))
                    self.assertEqual(entry.stat().st_nlink, 2)
                    replies, stderr = self.stdio(dest, [tools, navigate])
                    self.assertFalse(started.exists(), f"Chrome was started through {rel}")
                    self.assertEqual(replies[0]["result"]["tools"], [])
                    self.assertIn("hard links", stderr)
                    self.assertIn("hard links", replies[1]["error"]["message"])
                finally:
                    entry.unlink()
                    entry.write_text("{}", encoding="utf-8")
                    started.unlink(missing_ok=True)
                self.assertEqual(entry.stat().st_nlink, 1)

        replies, stderr = self.stdio(dest, [tools])
        self.assertEqual(len(replies[0]["result"]["tools"]), 16, stderr)
        self.assertFalse(started.exists())
        self.assertEqual(outside.read_text(encoding="utf-8"), "{}")

    @unittest.skipUnless(os.name == "posix", "Shell test executable")
    def test_headless_human_wait_uses_the_same_ws_resolution_as_preflight(self):
        fake = self.root / "fake-chrome"
        started = self.root / "started"
        fake.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo synthetic; exit 0; fi\n'
                        f'touch {shlex.quote(str(started))}\nexit 5\n', encoding="utf-8")
        fake.chmod(0o700)
        self.env["MCP_BROWSER_CHROME"] = str(fake)
        call = {"method": "tools/call", "id": 1,
                "params": {"name": "wait_for_human", "arguments": {}}}
        for ws in (None, "", " \t", "invalid-no-port"):
            with self.subTest(ws=ws):
                if ws is None:
                    self.env.pop("MCP_BROWSER_WS_URL", None)
                else:
                    self.env["MCP_BROWSER_WS_URL"] = ws
                replies, _ = self.stdio(None, [call])
                if ws and ws.strip():
                    self.assertIn("remote CDP endpoint", replies[0]["error"]["message"])
                else:
                    self.assertIn("status=unavailable", replies[0]["result"]["content"][0]["text"])
                self.assertFalse(started.exists(), "headless human wait must not start Chrome")

    @unittest.skipUnless(os.name == "posix", "Shell Chrome wrapper")
    def test_close_browser_reports_unconfirmed_exit(self):
        chrome = shlex.quote(self.env["MCP_BROWSER_CHROME"])
        wrapper = self.root / "exit-chrome"
        wrapper.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then exec {chrome} "$@"; fi\n'
                           f'{chrome} "$@"\nexit 37\n', encoding="utf-8")
        wrapper.chmod(0o700)
        self.env["MCP_BROWSER_CHROME"] = str(wrapper)
        replies, _ = self.stdio(self.root / "close-profile", [
            {"method": "tools/call", "id": 1, "params": {"name": "navigate", "arguments": {"url": "about:blank"}}},
            {"method": "tools/call", "id": 2, "params": {"name": "close_browser", "arguments": {}}},
        ])
        self.assertIn("result", replies[0], replies[0])
        self.assertIn("exit status: 37", replies[1]["error"]["message"])
        self.assertIn("profile writes were not confirmed", replies[1]["error"]["message"])

    def test_cookie_and_local_storage_survive_a_profile_clone(self):
        source = self.root / "source"
        dest = self.root / "dedicated"
        with self.fixture_page() as (url, _):
            result = self.chrome_round_trip(source, url, """(() => {
                document.cookie = 'fixture_session=synthetic; Max-Age=3600; Path=/';
                localStorage.setItem('fixture_state', 'synthetic-storage');
                return [document.cookie, localStorage.getItem('fixture_state')];
            })()""")
            expected = ["fixture_session=synthetic", "synthetic-storage"]
            self.assertEqual(result, expected)
            (source / "Default/CacheTest").mkdir()
            (source / "Default/CacheTest/fixture").write_text("unused", encoding="utf-8")
            (source / "Profile 42").mkdir()
            local_state = (source / "Local State").read_bytes()

            copied = self.sync_profile(source, dest)
            self.assertEqual(copied.returncode, 0, copied.stderr)
            self.assertEqual((dest / "Local State").read_bytes(), local_state)
            marker = json.loads((dest / ".mcp_browser_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["version"], 1)
            self.assertEqual(marker["source"], str(source))
            self.assertEqual(marker["profile_directory"], "Default")
            self.assertEqual(marker["keychain"], "mock")
            self.assertGreater(marker["synced_at_unix"], 0)
            self.assertNotIn("fixture_session", copied.stdout + copied.stderr)
            self.assertFalse((dest / "Profile 42").exists())
            self.assertFalse((dest / "Default/Cache").exists())
            self.assertFalse((dest / "Default/CacheTest").exists())
            self.assertFalse(os.path.lexists(dest / "SingletonLock"))
            self.assertIn("cookie decryption has not been verified", copied.stdout)
            self.assertIn("does not convert existing native cookies", copied.stderr)

            self.assertEqual(self.chrome_round_trip(dest, url, """[
                document.cookie, localStorage.getItem('fixture_state')
            ]"""), expected)
            self.assertFalse((dest / "DevToolsActivePort").exists())
            extra = dest / "fixture-to-replace"
            extra.write_text("old profile", encoding="utf-8")
            refused = self.sync_profile(source, dest)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("not empty", refused.stderr)
            self.assertTrue(extra.exists())
            if sys.platform == "linux":
                (dest / "SingletonLock").symlink_to(f"{socket.gethostname()}-2147483647")
                for name in ("DevToolsActivePort", "lockfile"):
                    (dest / name).write_text("stale fixture", encoding="utf-8")
            replaced = self.sync_profile(source, dest, "--force")
            self.assertEqual(replaced.returncode, 0, replaced.stderr)
            self.assertFalse(extra.exists())
            for name in ("SingletonLock", "DevToolsActivePort", "lockfile"):
                self.assertFalse(os.path.lexists(dest / name))

    @unittest.skipUnless(
        sys.platform == "linux" and os.environ.get("MCP_BROWSER_TEST_VIRTUAL_LOGIN") == "1",
        "set MCP_BROWSER_TEST_VIRTUAL_LOGIN=1 for Chrome's Linux virtual display",
    )
    def test_standalone_login_flushes_the_selected_profile_on_enter(self):
        self._check_login_close("enter")

    @unittest.skipUnless(
        sys.platform == "linux" and os.environ.get("MCP_BROWSER_TEST_VIRTUAL_LOGIN") == "1",
        "set MCP_BROWSER_TEST_VIRTUAL_LOGIN=1 for Chrome's Linux virtual display",
    )
    def test_standalone_login_flushes_after_interrupt(self):
        for event in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(event=event.name):
                self._check_login_close(event.name, event)
            # The terminal sends Ctrl-C to the entire foreground group.
            with self.subTest(group_event=event.name):
                self._check_login_close(f"group-{event.name}", event, group=True)

    def _check_login_close(self, scenario, interrupt=None, *, group=False):
        source = self.root / f"source-{scenario}"
        (source / "Profile 7").mkdir(parents=True)
        (source / "Local State").write_text(json.dumps({"profile": {
            "last_used": "Default", "info_cache": {"Profile 7": {"name": "Work"}}
        }}), encoding="utf-8")
        dest = self.root / f"dest-{scenario}"
        copied = self.sync_profile(source, dest, profile="Work")
        self.assertEqual(copied.returncode, 0, copied.stderr)
        self.assertFalse((dest / "Default").exists())
        if interrupt == signal.SIGTERM:
            (dest / "SingletonLock").symlink_to(f"{socket.gethostname()}-2147483647")
            for name in ("DevToolsActivePort", "lockfile"):
                (dest / name).write_text("stale fixture", encoding="utf-8")
        wrapper = self.root / "chrome-virtual-display"
        chrome = shlex.quote(os.environ["MCP_BROWSER_CHROME"])
        wrapper.write_text(f'#!/bin/sh\nexec {chrome} --ozone-platform=headless "$@"\n', encoding="utf-8")
        wrapper.chmod(0o700)
        with self.fixture_page() as (url, saved):
            # This is still a headed Chrome invocation; use a virtual platform, not a desktop window.
            env = dict(self.env, MCP_BROWSER_CHROME=str(wrapper), DISPLAY=":virtual")
            login = subprocess.Popen(
                [self.binary, "--login", "--profile-dir", str(dest), "--url", url + "signin"],
                env=env, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=group,
            )
            try:
                if not saved.wait(25):
                    login.kill()
                    self.fail(f"login page did not save the fixture: {login.communicate(timeout=10)[1]}")
                if interrupt is None:
                    stdout, stderr = login.communicate("\n", timeout=15)
                else:
                    if group:
                        os.killpg(login.pid, interrupt)
                    else:
                        login.send_signal(interrupt)
                    # Keep stdin open so EOF cannot stand in for the signal.
                    login.wait(timeout=15)
                    stdout, stderr = login.communicate(timeout=5)
            finally:
                if login.poll() is None:
                    login.kill()
                    login.communicate(timeout=10)
            self.assertEqual(login.returncode, 0, stderr)
            self.assertEqual(stdout, "", "standalone login must not speak JSON-RPC")
            if interrupt is None:
                self.assertIn("press Enter", stderr)
            self.assertFalse((dest / "DevToolsActivePort").exists())
            self.assertFalse((dest / "Default").exists(), "login must open the marker-selected profile")
            self.assertEqual(self.chrome_round_trip(dest, url, """[
                document.cookie, localStorage.getItem('fixture_login')
            ]"""), ["fixture_login=synthetic", "synthetic-storage"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
