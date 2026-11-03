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
| 工具化记忆检索 | 6 个 llm_tool：`search_memory` / `save_memory` / `list_skills` / `read_skill` / `save_skill` / `render_mermaid` |
| （本插件扩展）WebUI 管理页面 | AstrBot 侧边栏出现「记忆与技能」页面：概览/记忆/技能/模型/设置/工具，可视化检索、编辑、合并记忆与技能，管理总结模型链 |

## Mermaid 图表渲染（本地，无需外网渲染服务）

直接在对话里让 bot"画个流程图/时序图/架构图"，模型会调用 `render_mermaid`
工具把 mermaid 源码在**本机渲染成 PNG 并发到聊天**。也可用命令：

```
/hermes 图 flowchart TD\n A[开始] --> B[结束]
```

实现方式与在线服务（mermaid.ink / kroki）完全不同——那些服务在墙内经常
不可达，而且 bot 容易在不可用服务上反复重试浪费时间。本插件的做法：

1. 首次使用时从 jsdelivr / npmmirror 缓存一份 `mermaid.min.js`（约 3.5MB）
2. 通过 `npm` 安装纯 JS 的 `puppeteer-core`（不下载浏览器，约 3MB）
3. 调用**系统已安装的 Chrome 或 Edge** 无头模式截图

依赖：Node.js（渲染引擎）+ Chrome/Edge（截图，Windows 10/11 自带 Edge）。
首次渲染约 10 秒（下载依赖），之后约 1~2 秒。
| 定时任务 | 复用插件后台 asyncio 循环（默认 30 分钟检查一次，可配置） |
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

### 关于 ID：QQ 官方机器人没有"QQ 号"

这点必须说清楚，否则会踩坑。AstrBot 的 QQ 官方适配器（`qq_official`）里：

| 场景 | `get_sender_id()` | `get_group_id()` | 会话 umo 第三段 |
|---|---|---|---|
| 群消息 | `member_openid`（群内成员 openid） | `group_openid` | `group_openid` |
| C2C 私聊 | `user_openid` | 空 | `user_openid` |

- `member_openid` **按群维度不同**：同一个人在群 A 和群 B 是两套 ID
- `member_openid` ≠ `user_openid`：同一个人在群聊和私聊是两套 ID
- openid **与机器人 AppID 绑定**：换 bot 后全部 ID 变化

因此本插件做了两件事：

1. **群公共记忆的键用 `(platform, group_openid)`，不用 umo** —— 所以开关 `unique_session`（会改写 umo 为 `sender_group`）不会让既有群记忆失联
2. **身份别名表**：`/hermes 绑定 <另一个ID>` 可把一个人的多个 openid 归一，跨群/跨私聊的记忆合并

> 用 `NapCat/aiocqhttp` 系列适配器时 `sender_id` 就是真实 QQ 号，上面的碎片化问题不存在。

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

- `/hermes 状态` — 记忆/技能/消息统计（含已合并条数）
- `/hermes 总结` — 立即触发一次总结
- `/hermes 整理` — 把碎片化的相似记忆合并成一条；加 `预览` 只看不写
- `/hermes 技能整理` — 把重复的技能合并成一个；加 `预览` 只看不写
- `/hermes 记住 <内容>` — 手动写入记忆
- `/hermes 记忆` — 查看最近记忆（含 id，可配合删除）
- `/hermes 忘记 <id>` — 删除某条记忆
- `/hermes 技能` — 列出技能；`/hermes 技能 <名称>` 查看全文
- `/hermes 导出` — 导出 MEMORY.md 到插件数据目录
- `/hermes 导入历史` — 导入 AstrBot 原生对话历史（接管记忆时用）
- `/hermes 我的ID` — 查看当前平台的 sender_id / 归属 ID / 群 ID / umo（排查隔离问题）
- `/hermes 绑定 <ID>` — 把一个别名 ID 归并到当前身份（跨群、跨私聊合并记忆）
- `/hermes 身份` — 查看所有身份绑定；`/hermes 解绑 <ID>` 取消绑定
- `/hermes 清空记录` — 清空对话上下文但**保留记忆与技能**：先预演条数，再加 `全群 确认`（管理员，会同时清 AstrBot 原生会话）或 `我 确认`（仅自己的消息记录）

## 总结是怎么触发的（按会话独立）

三个参数容易误解，实际规则是**按会话独立判断**（会话 = 一个群窗口，或一个私聊窗口）：

| 参数 | 真正的含义 | 不是 |
|---|---|---|
| `summarize_interval_minutes` | 每 N 分钟醒一次，**逐个会话**检查 | 不是「每 N 分钟总结一次」 |
| `min_new_messages` | **单个会话**攒够这么多条才总结 | 不是「全库合计够 N 条」 |
| `summarize_grace_hours` | 某会话的消息超过 N 小时还没总结，即使不够条数也强制总结 | 不是时效性要求 |

举例（阈值 8 条 / 兜底 6 小时）：

- 群 A 攒了 10 条 → 总结群 A，一次 LLM 调用
- 群 B 只有 3 条但已经放了 7 小时 → 兜底总结群 B
- 私聊 C 刚聊 2 条 → 跳过，等下轮继续攒或等兜底

**每个会话独立一次 LLM 调用、串行执行，绝不把多个会话塞进同一个 prompt**；各会话的总结水位（`meta` 表 `summarized:<会话ID>`）互不影响。

一个必须说清的点：**会话不等于人**。同一个群里的 A 和 B 共享一个会话，他们的话会进同一次总结——这是刻意的，因为群聊本来就是多人交织的对话，按人拆开会丢失「谁在回谁」的上下文，而且 LLM 调用次数会翻倍。归属安全靠三层保证：每句带 `[用户ID(昵称)]` 标记 → 模型产出 `owner` 字段 → 写入前校验 owner 必须出现在本批对话的 ID 里，否则降级为群公共记忆（不会硬塞给某个人）。

如果确实要求「完全按人隔离总结」，把 AstrBot 的 `unique_session` 打开即可：群里每个人变成独立会话，自然就按人拆开了（代价是失去群公共上下文）。

## 记忆会自己整理（v0.6.0）

不做整理的话，「喜欢吃香蕉」和「喜欢吃苹果」会变成两条八竿子打不着的记忆，
用的时候模型只捞到一条；技能更糟——同一个绘图能力被总结三次，就变成三个技能，
每次用之前都得读一遍。

整理要花一次模型调用，所以**不逐条整理**，而是先排队、攒够阈值再一次性处理
（详见下面的「什么时候整理」）。

攒够之后会自动跑一次**巩固**：

1. **本地找候选**：拿刚写的记忆和同一用户/同一群的历史记忆比字面相似度（字符二元组 Jaccard），
   只把相近的挑出来。没有候选就结束，不花钱。
2. **模型判断**：把候选组一次性交给 LLM，让它决定怎么合——
   - 互补的合并：`喜欢吃香蕉` + `喜欢吃苹果` → `喜欢吃香蕉、苹果`
   - 冲突的取最新：`喜欢猫` → 后来 `不喜欢猫了`，只留最新状态
   - 无关的保持分开，不硬凑

字面相似度**只用来召回**，不直接决定合并。原因很实在：`喜欢吃香蕉 / 喜欢吃苹果` 相似度只有 0.33，
而 `养了只猫叫咪咪 / 养了只狗叫旺财` 有 0.38——字面分不出该不该合，得靠模型读语义。

### 什么时候整理：攒够 N 条才合并（v0.6.1）

新记忆不会立刻合并，而是进一个**待整理队列**，等队列攒够阈值再一次性处理：

| 配置项 | 默认 | 含义 |
|---|---|---|
| `consolidate_every_n_memories` | 10 | 累计新增 10 条记忆才整理一次；`0` = 关闭自动，只能手动 `/hermes 整理` |
| `skill_consolidate_every_n` | 5 | 累计新增/更新 5 个技能才合并一次；`0` = 关闭自动，只能手动 `/hermes 技能整理` |

这样 100 条新记忆最多触发 10 次整理，而不是每次总结都多花一次调用。
队列里的 id 存在 `meta.consolidate_pending_memories`（上限 200 条，超出丢最旧的），
模型手动 `save_memory`、`/hermes 记住` 存的记忆也会进队列，不会漏掉。

`/hermes 状态` 里能看到进度：

```
自动整理: 记忆 3/10 条、技能 1/5 个 触发一次（每次约 1 次模型调用）
```

整理失败（模型超时、报错）时**不清队列**，下一轮自动重试；成功才清零。
手动执行 `/hermes 整理` / `/hermes 技能整理` 会顺手清空对应队列，避免重复整理。

技能的重复问题走另一条路：

- 总结前会把**已有技能清单**放进提示词，要求同类流程直接沿用已有技能名，不要另起新名（从源头减少重复）
- 同名技能不再直接覆盖，而是**先读旧版再合并**，保留旧版里仍有效的步骤
- `/hermes 技能整理` 把名字/触发条件相近的多个技能合并成一个，`预览` 可先看计划

**误合并能回滚**：合并后旧条不删除，只把 `merged_into` 指向新条，检索时排除。真合错了，
直接把 `merged_into` 置回 0 就能恢复（也保留完整历史，方便排查）。

## WebUI 管理页面（v0.7.0）

装好后，AstrBot WebUI 的侧边栏会出现插件页面 **「Hermes · 记忆与技能」**（插件详情里也有入口）。
页面零依赖（无 CDN、无构建），样式跟随面板明暗主题，共六个标签：

| 标签 | 能做什么 |
|---|---|
| **概览** | 消息/记忆/技能统计、数据库大小、总结模型、整理队列进度条、最近记忆与技能使用 Top |
| **记忆** | 关键词/作用域/类型筛选 + 分页；编辑、删除、新增；勾选多条**手动合并**（可改写合并文案）；勾选「显示已合并」查看历史 |
| **技能** | 左列表右编辑器；查看/修改技能名、触发条件与 Markdown 正文，支持**渲染预览**；新建/删除 |
| **模型** | 总结用的模型链：主模型 + 有序降级链、容错策略、连通性测试、调用日志（详见下一节） |
| **设置** | 直接编辑插件配置（与 AstrBot 配置页同一份文件），schema 驱动的表单，bool 开关/下拉/数值/文本各自呈现；标「需重载」的项会提示 |
| **工具** | 立即总结、记忆整理（预览/执行）、技能整理（预览/执行）、导出 MEMORY.md、身份别名绑定 |

安全设计：

- 手动合并记忆要求选中条目同用户/同群，跨边界的合并会被后端直接拒绝
- 配置保存会按 schema 校验类型与取值范围，非法值不会写入
- 所有删除操作都有二次确认弹窗；记忆合并是软删除，可回溯

实现：页面在 `views/hermes/`，接口通过 `context.register_web_api` 注册（`webapi.py`），
前端经 AstrBot 的 plugin page bridge 调用，全部数据留在本机。

## 模型链：总结不必用对话模型（v0.8.0）

对话模型要求聪明、会玩工具；总结只需要「读得懂、写得准」。把总结绑死在对话模型上，
遇到笨模型、短上下文、或者临时抽风，后台总结就整批失败——而且没人看见。

「模型」标签解决这件事：

- **自动发现**：列出 AstrBot 里配置的全部对话模型（含上下文长度、启用状态、哪个是当前对话模型），不另存密钥
- **主模型 + 降级链**：主模型失败后按顺序尝试降级模型，支持上移/下移排序；可一键「沿用 AstrBot 对话配置」
- **逐模型重试**：每个模型失败后按退避基数指数重试（`llm_max_retries`），整条链有总尝试上限，不会无限烧
- **错误分类**：超时/连接/限流/鉴权/上下文超长/服务端/空响应各有处理策略——鉴权错重试没有意义，直接换模型；可勾选哪些类别参与重试
- **失败冷却**：连续失败的模型进入冷却期（鉴权错 4 倍时长），后续轮次直接跳过，不每轮都撞；全部在冷却时仍会强制尝试，宁可撞也不停摆
- **按上下文裁剪**：同一个 prompt 按目标模型的 `max_context_tokens` 动态截断，小上下文模型当降级模型也能接住；另有全局单次字符上限防止大模型吞整段历史烧钱
- **连通性测试**：对任意模型发一条极短请求，验证密钥/网络/额度，显示延迟和失败原因
- **调用日志**：最近 80 次调用的时间、用途、模型、尝试次数、耗时、结果与错误摘要；单模型成功率统计

所有总结、记忆整理、技能合并都走这条链（`modules/router.py`）；单个会话总结失败不拖累其他会话，
失败的那批消息下一轮自动重试。

## 清上下文 vs 清记忆：两套独立存储

很容易混淆的一点：**上下文和记忆在物理上是分开存的**。

| | 存在哪 | 清掉会怎样 |
|---|---|---|
| 对话上下文 | AstrBot `conversations` 表 + 插件 `messages` 表 | 模型忘记刚才聊了什么，但记忆不受影响 |
| 长期记忆 | 插件 `memories` 表 | 人格画像丢失，但当前对话还在 |

关键结论：**清空上下文不会改变你的会话身份 ID**。`unified_msg_origin` 是由平台事件实时拼出来的确定性字符串（`平台:消息类型:group_openid/user_openid`），不来自数据库，删表也不会变；重新开会话只会生成新的 `conversation_id`（UUID），插件不依赖这个 UUID，只依赖 openid，所以记忆照常召回。

真正会让 ID 变化、导致记忆变成孤儿的情况只有三种：换机器人 AppID（openid 命名空间随 bot 变化）、换适配器（`qq_official` ↔ `aiocqhttp/NapCat`，后者返回真实 QQ 号）、用户换号。这三种都无法自动迁移，需要重新建立记忆或做数据映射。

## 开发：改完代码怎么生效（热重载，不用重启）

AstrBot **没有文件监听**——把新代码放到 `data/plugins/` 下不会自动生效，
但**也不需要重启整个后端**，调一次重载接口即可：

```bash
# 在插件目录（data/plugins/astrbot_plugin_hermes_core）里执行
python hot_reload.py --pull        # 拉代码 + 重载 + 核对是否真的生效

# Windows 直接双击 hot_reload.bat 即可（自动用 AstrBot 自带 python）
```

脚本做的事：

1. `git pull --ff-only`（`--pull` 时）
2. 用 `cmd_config.json` 里的 `dashboard.jwt_secret` 现签一个 5 分钟令牌
3. `POST /api/v1/plugins/reload`（body `{"plugin_id": "<插件名>"}`，WebUI 的重载按钮调的是同一个服务；
   全量重载走 legacy 的 `POST /api/plugin/reload`）
4. **核对**：读插件启动时写的 `build_stamp.json` 里的 `code_hash`，与磁盘上代码实时算出的哈希比对，
   输出「✅ 已生效」或「❌ 未生效」

> 为什么第 4 步必要：接口返回 200 只代表"重载流程跑完了"，不代表你改的代码真的被加载。
> 有了构建戳，`/hermes 状态` 也会显示 `版本: v0.5.1 (code 6b17dc0b56)`，一眼能看出跑的是哪份代码。

常用参数：`--name 插件名`（目录名与插件名不同时）、`--all`（重载全部插件，慢）、
`--url http://127.0.0.1:6185`、`--data C:\Users\你\.astrbot`。

注意：**改的是 `data/plugins/` 下那份代码**。如果你在别处（比如另一个 git clone）改，
得先把改动同步过去——脚本的 `--pull` 只处理 git 远端同步的情况。

### 更省事的办法：在聊天里重载

装 [`astrbot_plugin_hot_reload`](https://github.com/Summer-Ever/astrbot_plugin_hot_reload) 后，
不用开电脑，直接在 QQ 里发：

```
/reload hermes          # 或直接对机器人说「重载 hermes 插件」
/reload all             # 重载全部插件
```

它是独立插件，hermes_core 自己加载失败时它仍然可用，正好用来把坏掉的插件拉回来。

## 配置（WebUI 可视化）

- `memory_scope`：`user` / `group` / `global`，记忆隔离级别
- `per_user_history`：历史检索按发言人过滤
- `exclusive_memory`：独占记忆模式，启动时检测其他记忆插件
- `memory_injection`：开关请求前记忆注入
- `inject_memories_topk` / `inject_history_topk`：注入条数
- `auto_summarize` + `summarize_interval_minutes` + `min_new_messages` + `summarize_grace_hours`：自动总结节奏（见下）
- `auto_skill`：允许总结时自动提炼技能
- `auto_consolidate`：自动整理总开关（默认开）
- `consolidate_every_n_memories`：攒够多少条新记忆才整理一次（默认 10，`0` = 关闭自动）
- `skill_consolidate_every_n`：攒够多少个新/更新技能才合并一次（默认 5，`0` = 关闭自动）
- `consolidate_threshold`：召回相似记忆的阈值（默认 0.3，只影响候选范围）
- `consolidate_min_memories` / `consolidate_max_clusters`：最少多少条才整理、单次最多整理几组（控花费）
- `skill_merge_same_name`：同名技能先读旧版再合并（默认开）
- `skill_merge_threshold`：召回重叠技能的阈值（默认 0.2，是否合并仍由模型判断）
- `skill_consolidate_max`：技能不超过此数时全量交给模型整理（默认 40）
- `llm_primary_provider` / `llm_fallback_providers`：总结模型链（推荐用「模型」页维护）
- `llm_max_retries` / `llm_timeout` / `llm_backoff` / `llm_total_attempts`：逐模型重试次数、单次超时、退避基数、整链尝试上限
- `llm_cooldown_seconds`：失败模型冷却秒数（鉴权错自动 4 倍）
- `llm_max_prompt_chars`：单次送入模型的最大字符数
- `llm_retry_on`：哪些错误类别重试同一模型（默认超时/连接/限流/服务端/空响应）
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
