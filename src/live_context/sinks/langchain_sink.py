import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp

from ..core.event_bus import EventBus

logger = logging.getLogger(__name__)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class LangChainSinkConfig:
    backend_url: str = _env("FACTICITY_BACKEND_URL", "https://ecsbackend.facticity.ai")
    access_token: Optional[str] = _env("FACTICITY_ACCESS_TOKEN")
    location: str = _env("FACTICITY_LOCATION", "Global")
    user_email: str = _env("FACTICITY_USER_EMAIL", "live-context@facticity.ai")
    validator: str = _env("FACTICITY_VALIDATOR", "machine")
    frontend: str = _env("FACTICITY_FRONTEND", "web3")
    max_parallel: int = int(_env("FACTICITY_MAX_PARALLEL_FACTCHECK", "3"))
    fact_check_path: str = "/fact-check-sync"


class LangChainSink:
    """
    Watches EventBus for final ASR utterances and forwards them to
    LangChainAgents' fact-check endpoint. Emits fact_check_result
    events with the verdict once ECS responds.
    """

    def __init__(self, event_bus: EventBus, config: Optional[LangChainSinkConfig] = None):
        self.event_bus = event_bus
        self.config = config or LangChainSinkConfig()
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max(1, self.config.max_parallel))
        self._enabled = bool(self.config.backend_url and self.config.access_token)

        if not self._enabled:
            logger.warning(
                "[LangChainSink] Disabled – missing FACTICITY_BACKEND_URL or FACTICITY_ACCESS_TOKEN env vars."
            )
            return

        backend = self.config.backend_url.rstrip("/")
        self._fact_check_url = f"{backend}{self.config.fact_check_path}"
        event_bus.subscribe(self._handle_event)
        logger.info("[LangChainSink] Initialized; streaming results to %s", self._fact_check_url)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_event(self, event: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        if event.get("type") != "final_utterance":
            return

        text = (event.get("raw_text") or "").strip()
        if not text:
            return

        asyncio.create_task(self._process_utterance(event))

    # ------------------------------------------------------------------
    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            timeout = aiohttp.ClientTimeout(total=None)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def close(self) -> None:
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    # ------------------------------------------------------------------
    async def _process_utterance(self, event: Dict[str, Any]) -> None:
        async with self._semaphore:
            utterance_id = event.get("utterance_id")
            task_id = f"livectx_{uuid.uuid4().hex}"
            payload = self._build_payload(event, task_id)

            headers = {
                "Authorization": f"Bearer {self.config.access_token}",
                "Content-Type": "application/json",
                "Validator": self.config.validator,
                "Frontend": self.config.frontend,
            }

            session = await self._get_http_session()
            try:
                async with session.post(
                    self._fact_check_url,
                    json=payload,
                    headers=headers,
                ) as resp:
                    result = await self._consume_stream(resp)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "[LangChainSink] fact-check request failed (utterance=%s task_id=%s): %s",
                    utterance_id,
                    task_id,
                    exc,
                )
                return

            if result is None:
                logger.warning(
                    "[LangChainSink] fact-check produced no result (utterance=%s task_id=%s)",
                    utterance_id,
                    task_id,
                )
                return

            self._emit_fact_check_event(event, task_id, result)

    def _build_payload(self, event: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        timestamp = event.get("t_end")
        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc).isoformat()

        return {
            "query": event.get("raw_text", ""),
            "location": self.config.location,
            "timestamp": timestamp,
            "userEmail": self.config.user_email,
            "speaker": event.get("speaker", ""),
            "source": "live-context-service",
            "add": "",
            "client_api_key": "",
            "version": "pro",
            "context": "",
            "deployment_mode": "frontend2",
            "task_id": task_id,
            "requester_url": None,
        }

    async def _consume_stream(self, response: aiohttp.ClientResponse) -> Optional[Dict[str, Any]]:
        if response.status != 200:
            body = await response.text()
            logger.error(
                "[LangChainSink] fact-check HTTP %s: %s",
                response.status,
                body[:500],
            )
            return None

        buffer = ""
        async for chunk in response.content.iter_chunked(2048):
            if not chunk:
                continue
            chunk_text = chunk.decode("utf-8", errors="ignore")
            buffer += chunk_text

            try:
                maybe_json = json.loads(buffer.strip())
            except json.JSONDecodeError:
                continue

            if isinstance(maybe_json, dict) and "final" in maybe_json:
                try:
                    final_payload = json.loads(maybe_json["final"])
                except json.JSONDecodeError:
                    logger.warning("[LangChainSink] Unable to decode final payload: %s", maybe_json["final"])
                    return None
                return final_payload

        return None

    def _emit_fact_check_event(self, event: Dict[str, Any], task_id: str, payload: Dict[str, Any]) -> None:
        classification = payload.get("Classification")
        overall_assessment = payload.get("overall_assessment")

        result_event = {
            "type": "fact_check_result",
            "source": "langchain_sink",
            "session_id": event.get("session_id"),
            "utterance_id": event.get("utterance_id"),
            "task_id": task_id,
            "query": event.get("raw_text"),
            "speaker": event.get("speaker"),
            "classification": classification,
            "overall_assessment": overall_assessment,
            "payload": payload,
        }

        logger.info(
            "[LangChainSink] Fact-check complete (utterance=%s task_id=%s classification=%s)",
            event.get("utterance_id"),
            task_id,
            classification,
        )
        self.event_bus.emit(result_event)











