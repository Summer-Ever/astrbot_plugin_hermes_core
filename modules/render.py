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


def _find_browser() -> str | None:
    for p in _BROWSER_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _download(url: str, dest: str, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def ensure_mermaid_js(data_dir: str) -> str:
    """确保 mermaid.min.js 已缓存，返回其路径。"""
    d = os.path.join(data_dir, "renderer")
    os.makedirs(d, exist_ok=True)
    js = os.path.join(d, "mermaid.min.js")
    if os.path.exists(js) and os.path.getsize(js) > 100_000:
        return js
    last_err: Exception | None = None
    for url in _MERMAID_URLS:
        try:
            _download(url, js)
            if os.path.getsize(js) > 100_000:
                return js
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RenderError(f"无法下载 mermaid.min.js（检查网络）：{last_err}")


def _ensure_node_deps(data_dir: str) -> str:
    """确保 puppeteer-core 已安装到 renderer 目录，返回该目录。

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
        cwd=d, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not os.path.isdir(nm):
        raise RenderError(f"puppeteer-core 安装失败：{proc.stderr[-500:] or proc.stdout[-500:]}")
    return d


def _find_node() -> str | None:

    return shutil.which("node")


async def render_mermaid(code: str, out_png: str, data_dir: str,
                         theme: str = "default", width: int = 1000,
                         scale: int = 2) -> str:
    """把 mermaid 源码渲染为 PNG。成功返回 out_png。"""
    node = _find_node()
    if not node:
        raise RenderError("未找到 Node.js，请先安装（https://nodejs.org 或 winget install OpenJS.NodeJS.LTS）")
    if not _find_browser():
        raise RenderError("未找到 Chrome/Edge，无法本地渲染")

    js = ensure_mermaid_js(data_dir)
    vendor = _ensure_node_deps(data_dir)

    d = os.path.join(data_dir, "renderer")
    os.makedirs(d, exist_ok=True)
    mjs = os.path.join(d, "render.mjs")
    with open(mjs, "w", encoding="utf-8") as f:
        f.write(_RENDER_MJS)

    html = (_HTML_TMPL
            .replace("__THEME__", json.dumps(theme))
            .replace("__CODE__", json.dumps(code)))
    html_path = os.path.join(d, f"r_{hashlib.md5(code.encode()).hexdigest()[:12]}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

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
