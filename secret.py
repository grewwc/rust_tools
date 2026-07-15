#!/usr/bin/env python3
"""轻量级密钥加密/解密工具（与 Rust 端 `commonw::secret` 完全互通）。

复刻 src/commonw/secret.rs 的算法：
  - 密钥文件：~/.configW.secret（32 字节随机数，base64 编码，权限 0600）
    路径受 CONFIGW_PATH 环境变量影响（与 configw::config_path 一致）。
  - 加密格式：enc:<base64(nonce_8bytes + ciphertext)>
  - 算法：XOR 流密码；密钥流 = SHA256(secret ‖ nonce ‖ counter_le32) 逐块拼接后截断。

用法：
  python3 secret.py encrypt "my-api-key"        # 输出 enc:xxxx
  python3 secret.py decrypt "enc:xxxx"           # 输出明文
  python3 secret.py gen-secret [--force]         # 生成/轮换密钥文件
  python3 secret.py test                         # round-trip 自测

注意：每次加密用随机 nonce，相同明文产生不同密文；解密是确定性的。
      本方案依赖文件权限保护密钥，适合"静态数据保护"，非对抗性密码学方案。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import secrets as pysecrets
import sys
from pathlib import Path

NONCE_LEN = 8          # nonce 字节数（与 Rust 端一致）
SECRET_LEN = 32        # 密钥字节数
ENC_PREFIX = "enc:"    # 密文前缀


# ── 路径解析（对齐 configw::config_path + secret::secret_path）──────────────

def _expanduser(path: str) -> str:
    """与 Rust expanduser 一致：展开开头的 ~。"""
    return os.path.expanduser(path)


def config_path() -> Path:
    """对齐 configw::config_path：CONFIGW_PATH 优先，否则 ~/.configW。"""
    env = os.environ.get("CONFIGW_PATH", "").strip()
    if env:
        return Path(_expanduser(env))
    return Path(_expanduser("~/.configW"))


def secret_path() -> Path:
    """对齐 secret::secret_path：config_path().parent()/.configW.secret。

    config_path 的 parent 通常就是 home 目录；若 config_path 没有父级
    （极端情况），回退到自身。
    """
    cfg = config_path()
    parent = cfg.parent if cfg.parent != cfg else cfg
    return parent / ".configW.secret"


# ── 密钥文件（对齐 load_or_create_secret）─────────────────────────────────

def load_or_create_secret(path: Path | None = None) -> bytes:
    """读取现有密钥；不存在则生成 32 字节随机密钥并写入（权限 0600）。

    文件存在但格式不符（非 32 字节 base64）时抛错，与 Rust 行为一致。
    """
    path = path or secret_path()
    if path.exists():
        content = path.read_text().strip()
        try:
            key = base64.b64decode(content, validate=True)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"secret file {path} has invalid format "
                f"(expected 32-byte base64): {e}"
            ) from e
        if len(key) != SECRET_LEN:
            raise ValueError(
                f"secret file {path} has wrong length "
                f"(expected {SECRET_LEN} bytes, got {len(key)})"
            )
        return key

    # 生成新密钥
    key = pysecrets.token_bytes(SECRET_LEN)
    encoded = base64.b64encode(key).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    os.chmod(path, 0o600)
    print(f"[secret] generated new key at {path}", file=sys.stderr)
    return key


# ── 密钥流（对齐 derive_keystream）────────────────────────────────────────

def derive_keystream(secret: bytes, nonce: bytes, length: int) -> bytes:
    """SHA256(secret ‖ nonce ‖ counter_le32) 逐块拼接，截断到 length。

    counter 为 u32 小端，从 0 递增。每块 32 字节。
    """
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        h = hashlib.sha256()
        h.update(secret)
        h.update(nonce)
        h.update(counter.to_bytes(4, "little"))  # u32 LE
        stream.extend(h.digest())
        counter += 1
    return bytes(stream[:length])


# ── 加密 / 解密（对齐 encrypt / decrypt）──────────────────────────────────

def encrypt(plaintext: str, secret: bytes | None = None) -> str:
    """加密明文，返回 enc:<base64> 格式字符串。"""
    secret = secret if secret is not None else load_or_create_secret()
    nonce = pysecrets.token_bytes(NONCE_LEN)
    data = plaintext.encode("utf-8")
    keystream = derive_keystream(secret, nonce, len(data))
    ciphertext = bytes(b ^ k for b, k in zip(data, keystream))
    payload = nonce + ciphertext
    return ENC_PREFIX + base64.b64encode(payload).decode("ascii")


def decrypt(encoded: str, secret: bytes | None = None) -> str:
    """解密 enc:<base64> 字符串，返回明文。"""
    if not encoded.startswith(ENC_PREFIX):
        raise ValueError("missing 'enc:' prefix")
    b64 = encoded[len(ENC_PREFIX):]
    try:
        payload = base64.b64decode(b64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"base64 decode error: {e}") from e
    if len(payload) < NONCE_LEN:
        raise ValueError(
            f"payload too short (expected at least {NONCE_LEN} bytes for nonce)"
        )
    nonce, ciphertext = payload[:NONCE_LEN], payload[NONCE_LEN:]
    secret = secret if secret is not None else load_or_create_secret()
    keystream = derive_keystream(secret, nonce, len(ciphertext))
    plain = bytes(c ^ k for c, k in zip(ciphertext, keystream))
    try:
        return plain.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"invalid UTF-8 (wrong secret?): {e}") from e


def is_encrypted(s: str) -> bool:
    return s.startswith(ENC_PREFIX)


# ── CLI ──────────────────────────────────────────────────────────────────

def _cmd_encrypt(args: argparse.Namespace) -> int:
    print(encrypt(args.value))
    return 0


def _cmd_decrypt(args: argparse.Namespace) -> int:
    print(decrypt(args.value))
    return 0


def _cmd_gen_secret(args: argparse.Namespace) -> int:
    path = secret_path()
    if path.exists() and not args.force:
        print(f"secret file already exists at {path}; use --force to overwrite",
              file=sys.stderr)
        return 1
    # load_or_create_secret 不会覆盖已有文件，这里手动生成
    key = pysecrets.token_bytes(SECRET_LEN)
    encoded = base64.b64encode(key).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    os.chmod(path, 0o600)
    print(f"[secret] generated new key at {path}", file=sys.stderr)
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    """round-trip 自测：用当前 secret 文件加密再解密，验证闭环。"""
    secret = load_or_create_secret()
    samples = [
        "gpt-5.5",
        "http://dataagent-dev-llm.bytedance.net/v1/chat/completions",
        "ida-dev.api_key",
        "sk-Jj3wKxg5Yf807ocyEI6gagn3CA0fl4F7CMcCHRTz7ya8WBPx",
        "中文混合 test 123 🚀",
    ]
    ok = True
    for s in samples:
        enc = encrypt(s, secret)
        dec = decrypt(enc, secret)
        good = dec == s and enc != ENC_PREFIX + s and is_encrypted(enc)
        # 每次加密产生不同密文（nonce 随机）
        enc2 = encrypt(s, secret)
        nondeterministic = enc != enc2
        good = good and nondeterministic
        status = "OK " if good else "FAIL"
        print(f"[{status}] {s!r} -> {enc[:40]}{'...' if len(enc) > 40 else ''}")
        if not good:
            ok = False
    print("all passed" if ok else "SOME FAILED", file=sys.stderr)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="与 Rust commonw::secret 互通的 enc: 加解密工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encrypt", help="加密明文 -> enc:<base64>")
    p_enc.add_argument("value", help="明文")
    p_enc.set_defaults(func=_cmd_encrypt)

    p_dec = sub.add_parser("decrypt", help="解密 enc:<base64> -> 明文")
    p_dec.add_argument("value", help="enc:... 密文")
    p_dec.set_defaults(func=_cmd_decrypt)

    p_gen = sub.add_parser("gen-secret", help="生成/轮换密钥文件 (~/.configW.secret)")
    p_gen.add_argument("--force", action="store_true", help="覆盖已有密钥文件")
    p_gen.set_defaults(func=_cmd_gen_secret)

    p_test = sub.add_parser("test", help="round-trip 自测")
    p_test.set_defaults(func=_cmd_test)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
