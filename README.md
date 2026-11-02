# astrbot_plugin_hermes_core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AstrBot](https://img.shields.io/badge/For-AstrBot-blue)](https://github.com/AstrBotDevs/AstrBot)

复刻 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 学习闭环的 AstrBot 插件：让 AstrBot 的 bot 越用越懂你、能力越用越强，而不替换 AstrBot 任何原生功能。

## 功能对照（vs Hermes）

| Hermes 能力 | 本插件实现 |
|---|---|
| SQLite + WAL + FTS5 会话存储 | `modules/db.py`，trigram 分词支持中文，FTS 不可用时自动退化 LIKE |
| 长期记忆注入（MEMORY.md/USER.md） | `on_llm_request` 钩子按当前消息检索 top-k 记忆 + 相关历史，经 `extra_user_content_parts` 临时注入（保护 prompt 缓存） |
| 周期性记忆 nudge / LLM 总结 | 后台循环调用你已配置的 Provider，把新对话压缩为记忆条目（JSON 协议） |
| 技能自动生成（Markdown Skill） | 总结时一并提炼技能；模型也可通过 `save_skill` 工具主动沉淀，同名自动更新 |
| 工具化记忆检索 | 5 个 llm_tool：`search_memory` / `save_memory` / `list_skills` / `read_skill` / `save_skill` |
| 定时任务 | 复用插件后台 asyncio 循环（默认 30 分钟，可配置） |

## 安装

方式一（推荐）：AstrBot WebUI → 插件管理 → 从仓库安装，填入本仓库地址。

方式二（手动）：

```bash
cd AstrBot/data/plugins/
git clone https://github.com/Summer-Ever/astrbot_plugin_hermes_core.git
```

然后在 WebUI → 插件管理 → 重载，并确保已配置至少一个 LLM Provider（总结功能复用它）。

## 命令

- `/hermes 状态` — 记忆/技能/消息统计
- `/hermes 总结` — 立即触发一次总结
- `/hermes 记住 <内容>` — 手动写入记忆
- `/hermes 记忆` — 查看最近记忆（含 id，可配合删除）
- `/hermes 忘记 <id>` — 删除某条记忆
- `/hermes 技能` — 列出技能；`/hermes 技能 <名称>` 查看全文
- `/hermes 导出` — 导出 MEMORY.md 到插件数据目录

## 配置（WebUI 可视化）

- `memory_injection`：开关请求前记忆注入
- `inject_memories_topk` / `inject_history_topk`：注入条数
- `auto_summarize` + `summarize_interval_minutes` + `min_new_messages`：自动总结节奏
- `auto_skill`：允许总结时自动提炼技能
- `summary_prompt`：总结器提示词，可自定义输出协议

## QQ 官方机器人注意事项

- QQ 官方 API 仅支持纯文本回复，所有输出已按纯文本设计
- 官方 bot 为被动回复模式，技能生成通知会发到最近活跃的会话，可能受平台频控影响
- 消息、记忆、技能数据存于 AstrBot `data/plugin_data/astrbot_plugin_hermes_core/hermes.db`

## 与 Hermes 的差异

- 记忆检索为关键词（FTS5/LIKE）而非向量嵌入——零额外依赖，中文场景够用
- 技能暂不落 Markdown 文件树，存于 SQLite `skills` 表（`/hermes 导出` 可导出记忆）
- 无 Honcho 用户建模、无子代理——那两块依赖 Hermes 自身运行时，与 AstrBot 架构不兼容
