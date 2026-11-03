#!/usr/bin/env python
"""AstrBot 插件热重载工具（本地开发用，不需要重启后端）。

用法（在本文件所在目录执行）：

    python hot_reload.py                 # 重载当前目录名对应的插件
    python hot_reload.py --name xxx      # 指定插件名
    python hot_reload.py --pull          # 先 git pull 再重载
    python hot_reload.py --all           # 重载全部插件（慢）
    python hot_reload.py --url http://127.0.0.1:6185 --data C:\\Users\\me\\.astrbot

原理：用 AstrBot 配置文件里的 dashboard.jwt_secret 现签一个短时效令牌，
调用 POST /api/v1/plugins/<名字>/reload。重载后读取插件写下的
build_stamp.json 里的 code_hash，与磁盘上代码实时计算的哈希比对，
确认"跑的是不是最新代码"——这比只看接口返回 200 可靠。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_DATA_DIR = os.path.expanduser(r"~\.astrbot")


def load_dashboard_cfg(data_dir: str) -> dict:
    path = os.path.join(data_dir, "data", "cmd_config.json")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 AstrBot 配置：{path}\n用 --data 指定 .astrbot 目录")
    with open(path, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    db = cfg.get("dashboard") or {}
    if not db.get("jwt_secret"):
        raise SystemExit("配置里没有 dashboard.jwt_secret，无法签发令牌")
    return db


def make_token(secret: str, ttl: int = 300) -> str:
    try:
        import jwt  # AstrBot 自带
    except ImportError:
        raise SystemExit(
            "缺少 PyJWT。请用 AstrBot 自带的解释器运行，例如：\n"
            r'  D:\Local\AstrBot\backend\python\python.exe hot_reload.py --pull'
        )
    return jwt.encode(
        {"username": "astrbot", "exp": int(time.time()) + ttl},
        secret, algorithm="HS256",
    )


def call(url: str, token: str, path: str, method: str = "POST",
         body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body if body is not None else {}).encode() if method == "POST" else None
    req = urllib.request.Request(
        url + path, method=method, data=data,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def disk_hash(plugin_dir: str) -> str:
    h = hashlib.md5()
    for root, dirs, files in os.walk(plugin_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "renderer")]
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            h.update(os.path.relpath(p, plugin_dir).replace("\\", "/").encode())
            with open(p, "rb") as f:
                # 与插件内 _build_hash 保持一致：统一换行符，否则 CRLF/LF 差异
                # 会让“代码是否生效”的校验永远报不生效。
                h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="插件名，默认取当前目录名")
    ap.add_argument("--url", default=None, help="dashboard 地址，默认从配置读取")
    ap.add_argument("--data", default=DEFAULT_DATA_DIR, help=".astrbot 数据目录")
    ap.add_argument("--pull", action="store_true", help="先 git pull")
    ap.add_argument("--all", action="store_true", help="重载全部插件")
    ap.add_argument("--wait", type=float, default=3.0, help="重载后等待秒数")
    args = ap.parse_args()

    plugin_dir = os.path.abspath(os.getcwd())
    name = args.name or os.path.basename(plugin_dir)

    if args.pull:
        print("[1/3] git pull ...")
        r = subprocess.run(["git", "pull", "--ff-only"], cwd=plugin_dir,
                           capture_output=True, text=True)
        print((r.stdout or r.stderr).strip()[-300:])

    db = load_dashboard_cfg(args.data)
    url = args.url or f"http://127.0.0.1:{db.get('port', 6185)}"
    token = make_token(db["jwt_secret"])

    if args.all:
        # 新版 /api/v1/plugins/reload 要求 plugin_id 必填，全量只能走 legacy 接口
        path = "/api/plugin/reload"
        body = {"name": None}
    else:
        path = "/api/v1/plugins/reload"
        body = {"plugin_id": name}
    print(f"[2/3] POST {url}{path}")
    code, resp = call(url, token, path, body=body)
    if code in (404, 405, 422) and not args.all:
        # 老版本 AstrBot 只认 /api/v1/plugins/<name>/reload
        path = f"/api/v1/plugins/{name}/reload"
        print(f"      回退 {path}")
        code, resp = call(url, token, path)
    print(f"      HTTP {code} {resp[:200]}")
    if code != 200:
        print("重载失败。常见原因：插件名不对（用 --name）、服务未启动、端口不对（用 --url）")
        return 1

    time.sleep(args.wait)
    stamp_path = os.path.join(args.data, "data", "plugin_data", name, "build_stamp.json")
    if not os.path.exists(stamp_path):
        print("[3/3] 未找到 build_stamp.json（插件版本 < v0.5.1 不会写），无法自动核对。")
        return 0
    with open(stamp_path, encoding="utf-8") as f:
        stamp = json.load(f)
    running, on_disk = stamp.get("code_hash"), disk_hash(plugin_dir)
    print(f"[3/3] 运行中: v{stamp.get('version')} code={running} @ {stamp.get('loaded_at')}")
    print(f"      磁盘上: code={on_disk}")
    if running == on_disk:
        print("✅ 已生效：运行中的就是最新代码。")
        return 0
    print("❌ 未生效：接口返回成功但运行的是旧代码，或重载后磁盘又被改动过。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
