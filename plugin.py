"""用 JEV 决策服务判断麦麦是否继续回复最新私聊消息。"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import asyncio
import json
import time

from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

PLUGIN_ID = "github.pawnnwap.jev-reply-gate"
HISTORY_LIMIT = 20
MAX_DRAFT_CHARS = 1200
SYSTEM_PROMPT = """你叫麦麦，正在和对方私聊。判断麦麦现在还要不要继续回复最新一条用户消息。
只依据聊天记录和下面的拟回复判断。

1. 用户只回“嗯嗯”“好呀”“好的”“收到”“谢谢”“哈哈”“晚安”“先这样”“行吧”等短词，通常表示听到了、同意或准备结束。
2. 如果麦麦上一条已经回答、确认或处理了用户的最新需求，这些短回应就是收尾信号，不必再回“不客气”“好的呀”“哈哈”。
3. 只有最新消息提出新问题、请求、新信息、新话题、纠正，明确表达情绪并希望被接住，或继续追问；或者拟回复会给出用户尚未得到的新内容、确认下一步行动时，才继续回复。
4. 用户只是对麦麦的提议说“好呀”时，如果拟回复要落实提议、确认细节或执行下一步，就回复；如果只是再客套一句，就不回复。
5. 不确定时，优先不回复。"""
QUESTION_INSTRUCTIONS = """结合最近的私聊记录和下面的拟回复，麦麦现在应该继续回复最新这条用户消息吗？
拟回复（仅供判断，不是用户的话）：
{draft_reply}"""
QUESTION_CRITERIA = {
    "true": "应该回复：最新消息提出了问题、请求、新信息、新话题、纠正，或拟回复包含用户尚未得到的新内容、需要落实的下一步。",
    "false": "不必回复：最新消息只是嗯嗯、好呀、好的、收到、谢谢、哈哈、晚安等附和或收尾，麦麦已经回应过用户的最新需求；或拟回复只是想再附和、客套一句。",
}
UNFINISHED_SYSTEM_PROMPT = """检查上一轮私聊结束后，麦麦是否还有没做完的事。只依据聊天记录判断。
重点看麦麦明确承诺、但还没兑现的事，如“我去搜一下”“稍后给你结论”“等我查查”“马上帮你弄”。
用户提出明确请求（帮我看/帮我查/怎么解决），麦麦只回“收到/好呀/稍后”而没有行动或结论，也算未完成。
正常收尾、闲聊、道别，或已经给出结论/完成动作，不算未完成。
用户后来取消请求、说“不用了”，或话题已明确转移，不算未完成。"""
UNFINISHED_QUESTION = "上一轮对话结束后，麦麦是否还有明确承诺或待办没有完成？只有确实需要继续行动时才选 Yes。"
UNFINISHED_CRITERIA = {
    "true": "有未完成事项：麦麦明确说要搜索、查证、整理、稍后给结论或帮用户做什么，但还没有结果或完成动作；或用户提出了明确请求而麦麦尚未处理。",
    "false": "没有未完成事项：麦麦已经回答完用户的问题、给出结论或完成动作；只是普通收尾、道别、闲聊；用户已取消请求或转移话题；或没有需要继续处理的事。",
}
DUPLICATE_SYSTEM_PROMPT = """比较麦麦最近一条回复和候选回复。只依据给出的聊天上下文和两条文本判断。
两条回复意思相同、候选只是换个说法，或候选几乎没有新增信息，就算重复。
候选带来新信息、回应用户最新消息、推进话题，就不算重复。"""
DUPLICATE_QUESTION_TEMPLATE = """麦麦最近一条回复：
「{last_bot}」

候选回复：
「{candidate}」

候选回复是否重复低效（与最近一条回复意思重复、没有新增信息、没有推进对话）？"""
DUPLICATE_CRITERIA = {
    "true": "重复低效：候选回复和麦麦最近一条回复意思相同，或只是换皮复读，没有新增信息，没有推进对话。",
    "false": "不重复低效：候选回复有新信息、回应了用户最新消息、推进了话题，或完成了用户请求；即使简短也不算重复。",
}


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "toggle-left"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用 JEV 接话判断")
    config_version: str = Field(default="1.2.0", description="配置版本")


class JevConfig(PluginConfigBase):
    """JEV 服务连接配置。"""

    __ui_label__ = "JEV 服务"
    __ui_icon__ = "server"
    __ui_order__ = 1

    api_url: str = Field(
        default="http://127.0.0.1:8090",
        description="兼容 /v1/systemone 的决策服务地址；默认 http://127.0.0.1:8090",
    )
    model: str = Field(default="", description="请求中携带的模型名；留空使用服务默认值")
    timeout_seconds: float = Field(
        default=8.0,
        ge=1.0,
        le=15.0,
        description="单次决策请求超时秒数；超时后保留正常回复",
    )


class DecisionConfig(PluginConfigBase):
    """接话判断配置。"""

    __ui_label__ = "接话判断"
    __ui_icon__ = "message-circle"
    __ui_order__ = 2

    reply_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="JEV 给出的回复概率达到该值才继续回复；越高越容易保持安静",
    )


class UnfinishedConfig(PluginConfigBase):
    """未完成事项检测配置。"""

    __ui_label__ = "未完成事项"
    __ui_icon__ = "list-checks"
    __ui_order__ = 3

    enabled: bool = Field(default=True, description="bot 发话结束后扫描是否有未完成承诺或待办")
    scan_delay_seconds: float = Field(
        default=120.0,
        ge=10.0,
        le=120.0,
        description="最后一条 bot 消息后等待多久扫描；默认 120 秒",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="JEV 判定存在未完成事项的概率达到该值才触发主动搭话",
    )


class DuplicateConfig(PluginConfigBase):
    """重复低效回复过滤配置。"""

    __ui_label__ = "重复低效过滤"
    __ui_icon__ = "repeat"
    __ui_order__ = 4

    enabled: bool = Field(default=True, description="replyer 生成后判断是否重复低效，必要时请求重生成")
    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="JEV 判定重复低效的概率达到该值才请求重生成",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=3,
        description="最多请求重生成几次；受主程序 Hook 重试上限 3 约束，0 表示不重试",
    )


class JevReplyGateConfig(PluginConfigBase):
    """JEV 接话判断完整配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    jev: JevConfig = Field(default_factory=JevConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    unfinished: UnfinishedConfig = Field(default_factory=UnfinishedConfig)
    duplicate: DuplicateConfig = Field(default_factory=DuplicateConfig)


def _is_private_message(message: Dict[str, Any]) -> bool:
    message_info = message.get("message_info")
    return isinstance(message_info, dict) and not message_info.get("group_info")


def _message_user_id(message: Dict[str, Any]) -> str:
    message_info = message.get("message_info")
    if not isinstance(message_info, dict):
        return ""
    user_info = message_info.get("user_info")
    if not isinstance(user_info, dict):
        return ""
    return str(user_info.get("user_id") or "").strip()


def _resolve_bot_account_id(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        account_id = str(message.get("account_id") or "").strip()
        if account_id:
            return account_id
    return ""


def _is_bot_message(message: Dict[str, Any], bot_account_id: str = "") -> bool:
    user_id = _message_user_id(message)
    account_id = bot_account_id or str(message.get("account_id") or "").strip()
    return bool(user_id and account_id and user_id == account_id)


def _message_text(message: Dict[str, Any]) -> str:
    if message.get("is_notify") or message.get("is_command"):
        return ""
    text = message.get("processed_plain_text")
    if isinstance(text, str):
        normalized = " ".join(text.split())
        if normalized:
            return normalized
    if message.get("is_picture"):
        return "[图片]"
    if message.get("is_emoji"):
        return "[表情包]"
    return ""


def _collect_draft_text(reply_reason: str, reply_tool_args: Any) -> str:
    parts: List[str] = []
    for raw_value in (
        reply_reason,
        reply_tool_args.get("reply_reference") if isinstance(reply_tool_args, dict) else "",
    ):
        if not isinstance(raw_value, str):
            continue
        value = raw_value.strip()
        if value and value not in parts:
            parts.append(value)
    text = "\n".join(parts)
    return text[:MAX_DRAFT_CHARS]


def _build_jev_state(messages: List[Dict[str, Any]], bot_account_id: str, system_prompt: str) -> List[Dict[str, str]]:
    state: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for message in messages[-HISTORY_LIMIT:]:
        text = _message_text(message)
        if not text:
            continue
        state.append({"role": "assistant" if _is_bot_message(message, bot_account_id) else "user", "content": text})
    return state


def _systemone_url(api_url: str) -> str:
    normalized = api_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("JEV api_url 不能为空")
    if normalized.endswith("/v1/systemone"):
        return normalized
    return f"{normalized}/v1/systemone"


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"JEV 服务返回 HTTP {exc.code}: {detail[:500]}") from exc
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("JEV 服务返回的不是 JSON 对象")
    return data


def _extract_noul_probability(data: Dict[str, Any], question_id: str) -> float:
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("JEV 响应缺少 answers")
    answer = answers.get(question_id)
    if not isinstance(answer, dict):
        raise RuntimeError(f"JEV 响应缺少 {question_id} 答案")
    probability = answer.get("noul")
    if not isinstance(probability, (int, float)):
        raise RuntimeError(f"JEV 响应缺少 {question_id} 的 noul 概率")
    return float(probability)


def _extract_reply_probability(data: Dict[str, Any]) -> float:
    return _extract_noul_probability(data, "reply")


class JevReplyGatePlugin(MaiBotPlugin):
    """JEV 接话判断与未完成事项触发插件。"""

    config_model = JevReplyGateConfig

    def __init__(self) -> None:
        super().__init__()
        self._scan_tasks: Dict[str, asyncio.Task[None]] = {}
        self._scan_generations: Dict[str, int] = {}
        self._handled_rounds: Dict[str, Deque[str]] = {}
        self._last_user_message_ids: Dict[str, str] = {}

    async def on_load(self) -> None:
        return None

    async def on_unload(self) -> None:
        self._cancel_all_scans()
        self._last_user_message_ids.clear()
        self._handled_rounds.clear()

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        del scope, config_data, version
        config = cast(JevReplyGateConfig, self.config)
        if not config.plugin.enabled or not config.unfinished.enabled:
            self._cancel_all_scans()

    @HookHandler(
        "maisaka.replyer.before_request",
        name="jev_reply_gate",
        description="用 JEV 决策服务判断是否继续回复最新私聊消息",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=20000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def gate_reply(
        self,
        session_id: str = "",
        reply_message_id: str = "",
        reply_reason: str = "",
        reply_tool_args: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        config = cast(JevReplyGateConfig, self.config)
        if not config.plugin.enabled:
            return {"action": "continue"}
        session = str(session_id or "").strip()
        target_message_id = str(reply_message_id or "").strip()
        if not session or not target_message_id:
            return {"action": "continue"}
        try:
            messages = await self._fetch_recent_messages(session)
            latest_message = messages[-1] if messages else None
            if latest_message is None:
                return {"action": "continue"}
            bot_account_id = _resolve_bot_account_id(messages)
            if not _is_private_message(latest_message) or _is_bot_message(latest_message, bot_account_id):
                return {"action": "continue"}
            if latest_message.get("is_notify") or latest_message.get("is_command"):
                return {"action": "continue"}
            if str(latest_message.get("message_id") or "").strip() != target_message_id:
                return {"action": "continue"}
            if not _message_text(latest_message):
                return {"action": "continue"}
            draft_text = _collect_draft_text(reply_reason, reply_tool_args)
            probability = await self._ask_jev(messages, bot_account_id, draft_text, config)
        except Exception as exc:
            self.ctx.logger.warning("JEV 接话判断失败，保留正常回复: %s", exc)
            return {"action": "continue"}
        threshold = config.decision.reply_threshold
        if probability >= threshold:
            self.ctx.logger.debug(
                "JEV 允许回复: session=%s probability=%.3f threshold=%.2f",
                session,
                probability,
                threshold,
            )
            return {"action": "continue"}
        self.ctx.logger.info(
            "JEV 拦下收尾回复: session=%s probability=%.3f threshold=%.2f",
            session,
            probability,
            threshold,
        )
        return {
            "action": "abort",
            "custom_result": {
                "reason": (
                    f"主动保持安静（非故障）：最新私聊消息只是收尾或附和"
                    f"（回复概率 {probability:.2f}，阈值 {threshold:.2f}），麦麦已经回应过用户的最新需求。"
                    "请结束本轮，不要重复回复。"
                )
            },
        }

    @HookHandler(
        "maisaka.replyer.after_response",
        name="jev_duplicate_filter",
        description="用 JEV 判断回复是否重复低效，必要时请求 replyer 重生成",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=20000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def filter_repetitive_reply(self, **kwargs: Any) -> Dict[str, Any]:
        config = cast(JevReplyGateConfig, self.config)
        if not config.plugin.enabled or not config.duplicate.enabled:
            return {"action": "continue"}
        response = str(kwargs.get("response") or "").strip()
        session_id = str(kwargs.get("session_id") or "").strip()
        if not response or not session_id:
            return {"action": "continue"}
        retry_count = int(kwargs.get("retry_count") or 0)
        max_retries = min(int(kwargs.get("max_retries") or 0), config.duplicate.max_retries)
        if retry_count >= max_retries:
            self.ctx.logger.debug(
                "JEV 重复低效判断达到重试上限，放行当前回复: session=%s retry=%d/%d",
                session_id,
                retry_count,
                max_retries,
            )
            return {"action": "continue"}
        try:
            messages = await self._fetch_recent_messages(session_id)
            bot_account_id = _resolve_bot_account_id(messages)
            last_bot_message = self._find_last_bot_text(messages, bot_account_id)
            if not last_bot_message:
                return {"action": "continue"}
            last_user_message = self._find_last_user_text(messages, bot_account_id)
            probability = await self._ask_jev_duplicate(
                last_bot_message,
                last_user_message,
                response,
                config,
            )
        except Exception as exc:
            self.ctx.logger.warning("JEV 重复低效判断失败，保留当前回复: %s", exc)
            return {"action": "continue"}
        if probability < config.duplicate.threshold:
            self.ctx.logger.debug(
                "JEV 允许回复: session=%s 重复度%.0f%% 阈值%.0f%%",
                session_id,
                probability * 100,
                config.duplicate.threshold * 100,
            )
            return {"action": "continue"}
        self.ctx.logger.info(
            "JEV 判定回复重复低效，请求重生成: session=%s 重复度%.0f%% 重试%d/%d",
            session_id,
            probability * 100,
            retry_count,
            max_retries,
        )
        return {
            "action": "continue",
            "modified_kwargs": {
                **kwargs,
                "retry": True,
                "retry_reason": (
                    f"重复度{probability:.0%}，属于重复低效。"
                    "请换一个信息点或角度，避免复读最近说过的话，也不要只是把用户的话换个说法。"
                ),
            },
        }

    @HookHandler(
        "send_service.after_send",
        name="jev_unfinished_schedule",
        description="bot 发话成功后安排一次未完成事项扫描",
        mode=HookMode.OBSERVE,
        order=HookOrder.LATE,
        error_policy=ErrorPolicy.SKIP,
    )
    async def schedule_unfinished_scan(
        self,
        message: Any = None,
        sent: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if not sent or not isinstance(message, dict):
            return {"action": "continue"}
        config = cast(JevReplyGateConfig, self.config)
        if not config.plugin.enabled or not config.unfinished.enabled:
            return {"action": "continue"}
        session_id = str(message.get("session_id") or "").strip()
        reply_to = str(message.get("reply_to") or "").strip()
        if not session_id or not reply_to:
            return {"action": "continue"}
        if not _is_private_message(message):
            return {"action": "continue"}
        if self._last_user_message_ids.get(session_id) != reply_to:
            return {"action": "continue"}
        self._schedule_scan(session_id, reply_to)
        return {"action": "continue"}

    @HookHandler(
        "chat.receive.after_process",
        name="jev_unfinished_reset",
        description="收到用户消息时取消上一轮的未完成事项扫描",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
    )
    async def reset_unfinished_scan(self, message: Any = None, **kwargs: Any) -> Dict[str, Any]:
        del kwargs
        if not isinstance(message, dict):
            return {"action": "continue"}
        session_id = str(message.get("session_id") or "").strip()
        if not session_id or not _is_private_message(message):
            return {"action": "continue"}
        if message.get("is_command") or message.get("is_notify"):
            self._cancel_scan(session_id)
            return {"action": "continue"}
        account_id = str(message.get("account_id") or "").strip()
        if _is_bot_message(message, account_id):
            return {"action": "continue"}
        message_id = str(message.get("message_id") or "").strip()
        if message_id:
            self._last_user_message_ids[session_id] = message_id
        self._cancel_all_scans()
        return {"action": "continue"}

    async def _fetch_recent_messages(self, session_id: str) -> List[Dict[str, Any]]:
        raw_messages = await self.ctx.call_capability(
            "message.get_by_time_in_chat",
            chat_id=session_id,
            start_time=0.0,
            end_time=time.time(),
            limit=HISTORY_LIMIT,
            limit_mode="latest",
        )
        if isinstance(raw_messages, dict):
            error = raw_messages.get("error")
            if raw_messages.get("success") is False or error:
                raise RuntimeError(f"读取最近私聊消息失败: {error or raw_messages}")
        if not isinstance(raw_messages, list):
            raise RuntimeError("message.get_by_time_in_chat 未返回消息列表")
        return [message for message in raw_messages if isinstance(message, dict)]

    async def _request_jev_state(
        self,
        state_messages: List[Dict[str, str]],
        questions: Dict[str, Any],
        config: JevReplyGateConfig,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "state": {"messages": state_messages},
            "questions": questions,
        }
        model = config.jev.model.strip()
        if model:
            payload["model"] = model
        url = _systemone_url(config.jev.api_url)
        return await asyncio.to_thread(_post_json, url, payload, config.jev.timeout_seconds)

    async def _request_jev(
        self,
        messages: List[Dict[str, Any]],
        bot_account_id: str,
        system_prompt: str,
        questions: Dict[str, Any],
        config: JevReplyGateConfig,
    ) -> Dict[str, Any]:
        state_messages = _build_jev_state(messages, bot_account_id, system_prompt)
        return await self._request_jev_state(state_messages, questions, config)

    async def _ask_jev(
        self,
        messages: List[Dict[str, Any]],
        bot_account_id: str,
        draft_text: str,
        config: JevReplyGateConfig,
    ) -> float:
        data = await self._request_jev(
            messages,
            bot_account_id,
            SYSTEM_PROMPT,
            {
                "reply": {
                    "type": "noul",
                    "instructions": QUESTION_INSTRUCTIONS.format(draft_text=draft_text or "（没有额外说明。）"),
                    "criteria": QUESTION_CRITERIA,
                }
            },
            config,
        )
        return _extract_reply_probability(data)

    async def _ask_jev_unfinished(
        self,
        messages: List[Dict[str, Any]],
        bot_account_id: str,
        config: JevReplyGateConfig,
    ) -> float:
        data = await self._request_jev(
            messages,
            bot_account_id,
            UNFINISHED_SYSTEM_PROMPT,
            {
                "unfinished": {
                    "type": "noul",
                    "instructions": UNFINISHED_QUESTION,
                    "criteria": UNFINISHED_CRITERIA,
                }
            },
            config,
        )
        return _extract_noul_probability(data, "unfinished")

    async def _ask_jev_duplicate(
        self,
        last_bot_message: str,
        last_user_message: str,
        candidate: str,
        config: JevReplyGateConfig,
    ) -> float:
        state_messages: List[Dict[str, str]] = [{"role": "system", "content": DUPLICATE_SYSTEM_PROMPT}]
        if last_user_message:
            state_messages.append({"role": "user", "content": last_user_message})
        state_messages.append({"role": "assistant", "content": last_bot_message})
        data = await self._request_jev_state(
            state_messages,
            {
                "duplicate": {
                    "type": "noul",
                    "instructions": DUPLICATE_QUESTION_TEMPLATE.format(
                        last_bot=last_bot_message,
                        candidate=candidate,
                    ),
                    "criteria": DUPLICATE_CRITERIA,
                }
            },
            config,
        )
        return _extract_noul_probability(data, "duplicate")

    @staticmethod
    def _find_last_bot_text(messages: List[Dict[str, Any]], bot_account_id: str) -> str:
        for message in reversed(messages):
            if not _is_bot_message(message, bot_account_id):
                continue
            text = _message_text(message)
            if text:
                return text
        return ""

    @staticmethod
    def _find_last_user_text(messages: List[Dict[str, Any]], bot_account_id: str) -> str:
        for message in reversed(messages):
            if _is_bot_message(message, bot_account_id):
                continue
            text = _message_text(message)
            if text:
                return text
        return ""

    def _schedule_scan(self, session_id: str, round_key: str) -> None:
        config = cast(JevReplyGateConfig, self.config)
        if not config.plugin.enabled or not config.unfinished.enabled:
            return
        triggered = self._handled_rounds.get(session_id)
        if triggered and round_key in triggered:
            return
        old_task = self._scan_tasks.pop(session_id, None)
        if old_task is not None:
            old_task.cancel()
        generation = self._scan_generations.get(session_id, 0) + 1
        self._scan_generations[session_id] = generation
        task = asyncio.create_task(
            self._scan_after_delay(session_id, round_key, generation),
            name=f"{PLUGIN_ID}.unfinished.{session_id}",
        )
        self._scan_tasks[session_id] = task
        task.add_done_callback(
            lambda done, sid=session_id: self._scan_tasks.pop(sid, None) if self._scan_tasks.get(sid) is done else None
        )

    async def _scan_after_delay(self, session_id: str, round_key: str, generation: int) -> None:
        try:
            config = cast(JevReplyGateConfig, self.config)
            await asyncio.sleep(config.unfinished.scan_delay_seconds)
            await self._scan_unfinished(session_id, round_key, generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.ctx.logger.warning("JEV 未完成事项扫描失败: %s", exc)

    async def _scan_unfinished(self, session_id: str, round_key: str, generation: int) -> None:
        if self._scan_generations.get(session_id) != generation:
            return
        if self._last_user_message_ids.get(session_id) != round_key:
            return
        triggered = self._handled_rounds.get(session_id)
        if triggered and round_key in triggered:
            return
        config = cast(JevReplyGateConfig, self.config)
        if not config.plugin.enabled or not config.unfinished.enabled:
            return
        messages = await self._fetch_recent_messages(session_id)
        if not messages:
            return
        latest_message = messages[-1]
        bot_account_id = _resolve_bot_account_id(messages)
        if not _is_private_message(latest_message) or not _is_bot_message(latest_message, bot_account_id):
            return
        if str(latest_message.get("reply_to") or "").strip() != round_key:
            return
        if self._scan_generations.get(session_id) != generation:
            return
        probability = await self._ask_jev_unfinished(messages, bot_account_id, config)
        if self._scan_generations.get(session_id) != generation:
            return
        if probability < config.unfinished.threshold:
            self._mark_round_handled(session_id, round_key)
            self.ctx.logger.debug(
                "JEV 未完成事项扫描: session=%s probability=%.3f threshold=%.2f 无动作",
                session_id,
                probability,
                config.unfinished.threshold,
            )
            return
        intent = self._build_unfinished_intent(latest_message)
        if self._scan_generations.get(session_id) != generation:
            return
        self._mark_round_handled(session_id, round_key)
        result = await self.ctx.maisaka.proactive.trigger(
            session_id,
            intent,
            reason="JEV 检测到上一轮存在未完成的承诺或待办",
            metadata={
                "plugin_id": PLUGIN_ID,
                "kind": "unfinished_task",
                "probability": round(probability, 4),
                "round": round_key,
            },
        )
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"触发主动任务失败: {result.get('error')}")
        self.ctx.logger.info(
            "JEV 未完成事项触发主动搭话: session=%s probability=%.3f round=%s",
            session_id,
            probability,
            round_key,
        )

    @staticmethod
    def _build_unfinished_intent(latest_message: Dict[str, Any]) -> str:
        intent = (
            "上一轮对话结束后，检查发现你还有未完成的事，尤其是你答应过用户但还没兑现的承诺。"
            "请回看最近聊天，先完成这件事（需要查资料就调用工具或子代理），再自然地向用户补充结果；"
            "不要开启无关话题。"
        )
        last_text = _message_text(latest_message)
        if last_text:
            intent += f"\n你最后一条消息是：“{last_text[:200]}”"
        return intent

    def _mark_round_handled(self, session_id: str, round_key: str) -> None:
        rounds = self._handled_rounds.setdefault(session_id, deque(maxlen=50))
        if round_key not in rounds:
            rounds.append(round_key)

    def _cancel_scan(self, session_id: str) -> None:
        task = self._scan_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
        self._scan_generations[session_id] = self._scan_generations.get(session_id, 0) + 1

    def _cancel_all_scans(self) -> None:
        for task in list(self._scan_tasks.values()):
            task.cancel()
        self._scan_tasks.clear()
        self._scan_generations.clear()


def create_plugin() -> JevReplyGatePlugin:
    return JevReplyGatePlugin()
