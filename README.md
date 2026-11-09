# JEV 接话判断

用本机 JEV 兼容决策服务增强麦麦的私聊回复，包含三个独立功能：

- **接话判断**：用户只回“嗯嗯”“好呀”“收到”等收尾或附和时，不再机械接话。
- **重复低效过滤**：回复生成后按“重复度”判断，必要时请求重生成；最多重试 3 次，超过后放行最后一版。
- **未完成事项检测**：每轮 bot 发话结束约 2 分钟后检查是否有未兑现的承诺或待办；有则触发一次主动搭话，没有则静默到下一轮。

## 环境要求

- MaiBot 1.4.x，插件 SDK 2.22+
- 一个兼容 `/v1/systemone` 的 JEV 决策服务，默认地址 `http://127.0.0.1:8090`
- 当前功能只处理私聊；群聊不会触发

## 安装

从插件市场安装，或把本仓库放到 MaiBot 实例的 `plugins/` 目录后重载插件/重启 MaiBot。插件会在首次加载后生成 `config.toml`。

## 配置

```toml
[plugin]
enabled = true
config_version = "1.2.0"

[jev]
api_url = "http://127.0.0.1:8090"
model = ""
timeout_seconds = 8.0

[decision]
reply_threshold = 0.3

[unfinished]
enabled = true
scan_delay_seconds = 120.0
threshold = 0.5

[duplicate]
enabled = true
threshold = 0.6
max_retries = 3
```

- `decision.reply_threshold`：回复概率达到该值才继续接话，越高越安静。
- `unfinished.scan_delay_seconds`：最后一条 bot 消息后等待多久扫描，最大 120 秒；收到新私聊消息会取消本轮扫描。
- `unfinished.threshold`：判定存在未完成事项的概率阈值。
- `duplicate.threshold`：判定重复低效的概率阈值。
- `duplicate.max_retries`：最多请求重生成几次；受主程序 Hook 重试上限 3 约束。

## 行为说明

- 决策服务不可用、超时或返回异常时，保留正常回复并记录日志，不阻塞聊天。
- 未完成事项只触发一次主动任务，不搜索、不代写回复。
- 重复低效达到重试上限后放行当前版本，避免无限重试。

## License

GPL-v3.0-or-later
