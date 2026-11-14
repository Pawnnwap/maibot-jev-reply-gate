# JEV 接话判断

用 JEV 兼容决策服务或兼容 LLM 的 logprobs 增强麦麦的私聊回复，包含三个独立功能：

- **接话判断**：在 planner 请求前判断是否继续回复；用户只回“嗯嗯”“好呀”“收到”等收尾或附和时，直接移除 `reply` 工具，不生成回复。
- **重复低效过滤**：回复生成后按“重复度”判断，必要时请求重生成；最多重试 3 次，超过后放行最后一版。
- **未完成事项检测**：每轮 bot 发话结束约 2 分钟后检查是否有未兑现的承诺或待办；有则触发一次主动搭话，没有则静默到下一轮。

## 两种服务模式

- `service.mode = "jev"`：直接调用兼容 `/v1/systemone` 的决策服务。
- `service.mode = "llm"`：调用 OpenAI 兼容 Chat Completions，自动用首 token logprobs 模拟 systemone；内置 Spark X2.6 的默认地址与模型。

LLM 模式首次使用时会自动探测 API 是否支持 logprobs、assistant prefill 和关闭思考；探测结果会写入日志。不支持时保留正常回复，不阻塞聊天。

## 环境要求

- MaiBot 1.4.x，插件 SDK 2.22+
- JEV 模式：一个兼容 `/v1/systemone` 的服务
- LLM 模式：一个支持 logprobs 的 OpenAI 兼容 Chat Completions 服务
- 当前功能只处理私聊；群聊不会触发

## 安装

从插件市场安装，或把本仓库放到 MaiBot 实例的 `plugins/` 目录后重载插件/重启 MaiBot。插件会在首次加载后生成 `config.toml`。

## 配置

```toml
[features]
reply_enabled = true
duplicate_enabled = true
unfinished_enabled = true

[service]
mode = "jev"

[jev]
api_url = "http://127.0.0.1:8090"
model = ""
timeout_seconds = 5.0

[llm]
base_url = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
api_key = ""
model = "spark-x2.6"
timeout_seconds = 5.0

[decision]
reply_threshold = 0.5

[unfinished]
scan_delay_seconds = 120.0
threshold = 0.5

[duplicate]
threshold = 0.4
max_retries = 3
recent_window = 30
rule_min_shared_chars = 6
```

- `features.reply_enabled` / `features.duplicate_enabled` / `features.unfinished_enabled`：分别控制接话判断、重复低效过滤和未完成事项检测；关闭后，对应设置区块会在下次打开配置页时折叠。
- `service.mode`：`jev` 或 `llm`，二选一。
- `decision.reply_threshold`：回复概率达到该值才继续接话，越高越安静。
- `unfinished.scan_delay_seconds`：最后一条 bot 消息后等待多久扫描，最大 120 秒；收到新私聊消息会取消本轮扫描。
- `unfinished.threshold`：判定存在未完成事项的概率阈值。
- `duplicate.threshold`：判定重复低效的概率阈值。
- `duplicate.max_retries`：最多请求重生成几次；受主程序 Hook 重试上限 3 约束。
- `duplicate.recent_window`：规则比对回看最近多少条消息，只比对其中 bot 自身的发言。
- `duplicate.rule_min_shared_chars`：最长共享片段达到多少字，才把规则比对结果作为依据附给 JEV。

`[prompts]` 集中放置所有判断提示词，均带内置默认值。WebUI 只展示最常用的判定标准：

- `reply_criteria_true` / `reply_criteria_false`：需要回复 / 不用回复的情形；规则会追加到 question。
- `duplicate_criteria`：重复判断标准；规则放在 question。

聊天记录按 `用户:` / `助手:` 合并，最新用户消息前标 `【最新消息】`；判断规则放在 question，选项只有 Yes/No；不再发送 system prompt。问题模板、规则比对模板、未完成事项意图等技术性 Prompt 默认隐藏在配置文件中，需要时可直接在 `config.toml` 覆盖，无需在 WebUI 中暴露。

模板占位符：`{criteria}`、`{shared}`、`{ratio}`、`{record}`、`{last_message}`、`{probability}`。

LLM 模式的 `api_key` 只保存在本地 `config.toml`，不要提交到版本控制。

## 行为说明

- 接话判断在 `maisaka.planner.before_request` 执行；判无需回复时从 `tool_definitions` 移除 `reply` 工具，planner 无法调用回复。
- 重复过滤仍在 `maisaka.replyer.after_response`；未完成事项仍由 `send_service.after_send` 调度、`chat.receive.after_process` 重置。
- 决策服务不可用、超时或返回异常时，保留正常回复并记录日志。
- 未完成事项只触发一次主动任务，不搜索、不代写回复。
- 重复低效达到重试上限后放行当前版本，避免无限重试。
- LLM 探针会尝试 prefill / 非 prefill 与思考关闭参数的兼容组合；任何组合能返回 A/B 标签即视为可用。

## License

GPL-v3.0-or-later
