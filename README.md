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
| （本插件扩展）多用户隔离 | 记忆按 `用户 > 群 > 全局` 三级作用域存储与召回，群聊中不同群友互不污染 |
| （本插件扩展）接管 AstrBot 记忆 | 原生对话历史一键导入 + 内置记忆/其他记忆插件冲突检测 |

## 记忆隔离：群聊多人使用不串味

Hermes 本身是单用户私人助理，没有多用户概念；本插件为此增加了作用域层：

- `memory_scope=user`（默认）：记忆归属到具体发言人，召回时按 **本人 → 本群公共 → 全局** 优先级排序；群友 A 的偏好不会注入给群友 B
- `memory_scope=group`：以群/会话为单位共享记忆，适合"群规、群项目"这类公共知识
- `memory_scope=global`：全机器人共享（等同旧版行为，不建议群聊开启）
- `per_user_history=true`（默认）：历史检索增加 `sender_id` 过滤，不会把同群其他人的对话当上下文注入
- 技能库全机器人共享——技能是"能力"而非隐私，这也是 Hermes 的设计
- 总结器会收到带 `[用户 ID(昵称)]` 标记的对话，产出记忆时自动归属到对应人；归属不明的降级为群公共记忆

## 接管 AstrBot 的记忆体系

本插件在 `on_llm_request` / `on_llm_response` 两个官方钩子上完成"注入 + 落库"，因此可以把 AstrBot 的记忆职责完全收拢过来：

1. **关闭其他记忆来源**（避免双重注入）：WebUI 停用其他记忆类插件（如 livingmemory、hindsight 等）；若你的版本内置了试验性长期记忆，在 配置 → AI 相关选项 中关闭
2. **导入原生历史**：在需要的会话里执行 `/hermes 导入历史`，把 AstrBot 原有对话记录并入本插件记忆库，切换无感
3. **冲突自检**：`exclusive_memory=true`（默认）时，插件启动会在日志里列出检测到的其他记忆插件并提醒
4. **统一管理入口**：此后记忆的查看、删除、导出、总结、技能全部走 `/hermes` 命令组

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
- `/hermes 导入历史` — 导入 AstrBot 原生对话历史（接管记忆时用）

## 配置（WebUI 可视化）

- `memory_scope`：`user` / `group` / `global`，记忆隔离级别
- `per_user_history`：历史检索按发言人过滤
- `exclusive_memory`：独占记忆模式，启动时检测其他记忆插件
- `memory_injection`：开关请求前记忆注入
- `inject_memories_topk` / `inject_history_topk`：注入条数
- `auto_summarize` + `summarize_interval_minutes` + `min_new_messages`：自动总结节奏
- `auto_skill`：允许总结时自动提炼技能
- `summary_prompt`：总结器提示词，可自定义输出协议

## QQ 官方机器人注意事项

- AstrBot 的 QQ 官方适配器（v4.19.6+）支持收发文本、图片、语音、视频、文件（大文件自动分块上传），主动消息推送也已支持——但主动消息受 QQ 官方平台频控/额度限制，被动回复有响应时限
- 本插件输出以纯文本为主（记忆列表、技能全文等）；超长文本可能受单条消息长度限制被截断，技能全文建议优先用 `/hermes 技能 <名称>` 单条查看
- 建议在 QQ 群设置里把机器人「可访问消息」设为全部群消息、并开启「主动说话」，总结循环的技能通知才能正常送达
- 消息、记忆、技能数据存于 AstrBot `data/plugin_data/astrbot_plugin_hermes_core/hermes.db`

## 与 Hermes 的差异

- 记忆检索为关键词（FTS5/LIKE）而非向量嵌入——零额外依赖，中文场景够用
- 技能暂不落 Markdown 文件树，存于 SQLite `skills` 表（`/hermes 导出` 可导出记忆）
- 无 Honcho 用户建模、无子代理——那两块依赖 Hermes 自身运行时，与 AstrBot 架构不兼容
