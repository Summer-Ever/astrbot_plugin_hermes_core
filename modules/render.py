"""本地 mermaid 渲染：mermaid.min.js + 本机 Chrome/Edge 无头截图。

不依赖 mermaid.ink / kroki 等云服务（墙内不可达），也不需要下载
puppeteer 自带的 Chrome（体积大且下载源常被墙）。只装一个纯 JS 的
puppeteer-core（无浏览器下载），复用系统已安装的 Chrome 或 Edge。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

_MERMAID_URLS = [
    "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js",
    "https://unpkg.com/mermaid@11/dist/mermaid.min.js",
    "https://registry.npmmirror.com/mermaid/11.12.2/files/dist/mermaid.min.js",
]

_BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    # macOS / Linux 兜底
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]

_RENDER_MJS = r"""
import puppeteer from 'puppeteer-core';
import { existsSync } from 'fs';

const [, , htmlPath, outPath, widthArg, heightArg, scaleArg] = process.argv;
const exeCandidates = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/usr/bin/chromium-browser', '/usr/bin/chromium', '/usr/bin/google-chrome',
];
const exe = exeCandidates.find(p => existsSync(p));
if (!exe) { console.error('NO_BROWSER'); process.exit(2); }

const scale = +(scaleArg || '2');
const browser = await puppeteer.launch({
  executablePath: exe,
  headless: true,
  args: ['--disable-gpu', '--no-sandbox', '--hide-scrollbars',
         '--force-device-scale-factor=' + scale, '--font-render-hinting=none'],
  defaultViewport: { width: +(widthArg || 1000), height: +(heightArg || 800), deviceScaleFactor: scale },
});
try {
  const page = await browser.newPage();
  await page.goto('file:///' + htmlPath.replace(/\\/g, '/'), { waitUntil: 'load', timeout: 30000 });
  try { await page.waitForFunction('window.__done === true', { timeout: 20000 }); }
  catch (e) { /* 超时也截图，便于排错 */ }
  const st = await page.evaluate(() => ({ err: window.__err || null }));
  if (st.err) { console.error('PAGE_ERR ' + st.err); process.exit(3); }
  const el = await page.$('#out');
  if (el) { await el.screenshot({ path: outPath }); }
  else { await page.screenshot({ path: outPath, fullPage: true }); }
  console.log('OK ' + outPath);
} finally {
  await browser.close();
}
"""

_HTML_TMPL = """<!doctype html><html><head><meta charset="utf-8">
<style>
  body { margin:0; padding:16px; background:#ffffff;
         font-family:"Microsoft YaHei","PingFang SC",sans-serif; }
  #out { display:inline-block; }
  .err { color:#c0392b; font:14px monospace; white-space:pre-wrap; }
</style>
</head><body>
<div id="out">RENDERING</div>
<script src="mermaid.min.js"></script>
<script>
const M = window.mermaid
  || (__esbuild_esm_mermaid_nm && __esbuild_esm_mermaid_nm.mermaid
      && (__esbuild_esm_mermaid_nm.mermaid.default || __esbuild_esm_mermaid_nm.mermaid));
const out = document.getElementById('out');
window.__done = false; window.__err = null;
if (!M) { out.textContent = 'mermaid 加载失败'; window.__err = 'no mermaid global'; window.__done = true; }
else {
  M.initialize({ startOnLoad:false, theme: __THEME__, securityLevel:'loose' });
  M.render('g1', __CODE__).then(r => {
    out.innerHTML = r.svg; window.__done = true;
  }).catch(e => {
    out.className = 'err'; out.textContent = String(e);
    window.__err = String(e); window.__done = true;
  });
}
</script>
</body></html>
"""


class RenderError(Exception):
    pass


# ---------------------------------------------------------------- 准备阶段
#
# 首次渲染要先下 mermaid.min.js（几 MB）再 npm i puppeteer-core，两个都是
# 同步网络 IO，动辄几十秒到几分钟。这些**绝不能**在事件循环里直接跑：
# 会卡住 AstrBot 全部适配器与所有 dashboard 请求。所以：
#   1. 本文件其余函数保持同步（命令行自测、后台线程都能用）；
#   2. 所有 async 入口统一走 asyncio.to_thread；
#   3. 用 _prep_lock() 串行化准备过程，避免并发渲染重复下载；
#   4. 提供 warmup()，插件启动后在后台把依赖备好，用户第一次要图就不用等。

_MERMAID_MIN_BYTES = 100_000
_LOCK: "asyncio.Lock | None" = None
_LOCK_LOOP: "object | None" = None
# 记录"已经确认就绪"的目录，避免每次都 stat 一遍磁盘
_READY: set[str] = set()


def _prep_lock() -> "asyncio.Lock":
    """惰性创建锁，并绑定到当前事件循环。

    插件热重载会重新执行本模块，但旧 Lock 可能仍绑定在旧循环上；Python 3.10+
    在跨循环使用时会直接抛 RuntimeError。这里按循环重建，避免重载后首次渲染报错。
    """
    global _LOCK, _LOCK_LOOP
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _LOCK is None or (_LOCK_LOOP is not None and loop is not None and _LOCK_LOOP is not loop):
        _LOCK = asyncio.Lock()
        _LOCK_LOOP = loop
    return _LOCK


def _find_browser() -> str | None:
    for p in _BROWSER_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _find_node() -> str | None:
    return shutil.which("node")


def _download(url: str, dest: str, timeout: int = 20) -> None:
    """下载到临时文件，校验通过后再原子替换，避免半截文件被当缓存。"""
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    try:
        if os.path.getsize(tmp) < _MERMAID_MIN_BYTES:
            raise RenderError(f"下载内容过小（{os.path.getsize(tmp)}B），可能被拦截")
        with open(tmp, "rb") as f:
            head = f.read(200).lstrip()
        if head.startswith(b"<") or head.startswith(b"{"):
            raise RenderError("下载到的不是 JS（可能是错误页）")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def ensure_mermaid_js(data_dir: str) -> str:
    """确保 mermaid.min.js 已缓存，返回其路径。**同步**，只在工作线程里调用。"""
    d = os.path.join(data_dir, "renderer")
    os.makedirs(d, exist_ok=True)
    js = os.path.join(d, "mermaid.min.js")
    if os.path.exists(js) and os.path.getsize(js) > _MERMAID_MIN_BYTES:
        return js
    last_err: Exception | None = None
    for url in _MERMAID_URLS:
        try:
            _download(url, js, timeout=20)
            return js
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RenderError(f"无法下载 mermaid.min.js（检查网络）：{last_err}")


def _ensure_node_deps(data_dir: str) -> str:
    """确保 puppeteer-core 已安装到 renderer 目录，返回该目录。**同步**，只在工作线程里调用。

    注意：ESM import 不认 NODE_PATH，必须让 render.mjs 与 node_modules
    处于同一目录树，node 才能解析到 puppeteer-core。
    """
    d = os.path.join(data_dir, "renderer")
    nm = os.path.join(d, "node_modules", "puppeteer-core")
    if os.path.isdir(nm):
        return d
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        raise RenderError("未找到 npm")
    os.makedirs(d, exist_ok=True)
    proc = subprocess.run(
        [npm, "i", "puppeteer-core", "--no-audit", "--no-fund",
         "--registry", "https://registry.npmmirror.com"],
        cwd=d, capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0 or not os.path.isdir(nm):
        raise RenderError(f"puppeteer-core 安装失败：{proc.stderr[-500:] or proc.stdout[-500:]}")
    return d


def _needs_prepare(data_dir: str) -> bool:
    if data_dir in _READY:
        return False
    d = os.path.join(data_dir, "renderer")
    js = os.path.join(d, "mermaid.min.js")
    nm = os.path.join(d, "node_modules", "puppeteer-core")
    if (os.path.exists(js) and os.path.getsize(js) > _MERMAID_MIN_BYTES
            and os.path.isdir(nm)):
        _READY.add(data_dir)
        return False
    return True


def _prepare_blocking(data_dir: str) -> None:
    """下载/安装，纯同步。只能通过 asyncio.to_thread 调用。"""
    ensure_mermaid_js(data_dir)
    _ensure_node_deps(data_dir)
    _READY.add(data_dir)


async def warmup(data_dir: str) -> tuple[bool, str]:
    """后台预热（插件 initialize 时挂成 task）。返回 (是否就绪, 失败原因)。"""
    try:
        if _needs_prepare(data_dir):
            async with _prep_lock():
                if _needs_prepare(data_dir):
                    await asyncio.to_thread(_prepare_blocking, data_dir)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


async def render_mermaid(code: str, out_png: str, data_dir: str,
                         theme: str = "default", width: int = 1000,
                         scale: int = 2) -> str:
    """把 mermaid 源码渲染为 PNG。成功返回 out_png。

    所有磁盘/网络/子进程准备工作都在线程或子进程里完成，事件循环只 await。
    """
    node = await asyncio.to_thread(_find_node)
    if not node:
        raise RenderError("未找到 Node.js，请先安装（https://nodejs.org 或 winget install OpenJS.NodeJS.LTS）")
    if not await asyncio.to_thread(_find_browser):
        raise RenderError("未找到 Chrome/Edge，无法本地渲染")

    if _needs_prepare(data_dir):
        async with _prep_lock():
            # 双检：等锁期间别人可能已经装好了
            if _needs_prepare(data_dir):
                await asyncio.to_thread(_prepare_blocking, data_dir)

    d = os.path.join(data_dir, "renderer")
    mjs = os.path.join(d, "render.mjs")
    html = (_HTML_TMPL
            .replace("__THEME__", json.dumps(theme))
            .replace("__CODE__", json.dumps(code)))
    html_path = os.path.join(d, f"r_{hashlib.md5(code.encode()).hexdigest()[:12]}.html")

    def _write_inputs() -> None:
        os.makedirs(d, exist_ok=True)
        with open(mjs, "w", encoding="utf-8") as f:
            f.write(_RENDER_MJS)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    await asyncio.to_thread(_write_inputs)

    env = dict(os.environ)
    proc = await asyncio.create_subprocess_exec(
        node, mjs, html_path, out_png, str(width), str(width), str(scale),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env=env, cwd=d,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        raise RenderError("渲染超时（120s）")
    text = (out or b"").decode("utf-8", "replace").strip()
    if proc.returncode != 0 or not os.path.exists(out_png):
        raise RenderError(f"渲染失败：{text[-500:]}")
    return out_png


if __name__ == "__main__":
    # python -m modules.render <out.png> —— 命令行自测
    demo = 'flowchart TD\n    A["你好"] --> B["世界"]'
    print(render_mermaid(demo, sys.argv[1], os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
