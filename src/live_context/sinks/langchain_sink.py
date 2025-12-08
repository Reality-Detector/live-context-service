import asyncio
import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..core.event_bus import EventBus

logger = logging.getLogger(__name__)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


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
    extract_claim_path: str = "/extract-claim"
    snapshot_enabled: bool = _env_bool("FACTICITY_SNAPSHOT_ENABLED", True)
    snapshot_interval_sec: float = float(_env("FACTICITY_SNAPSHOT_INTERVAL", "30"))
    snapshot_max_utterances: int = int(_env("FACTICITY_SNAPSHOT_MAX_UTTERANCES", "12"))


class LangChainSink:
    """
    Watches EventBus for ASR events, builds a rolling transcript, and
    periodically sends snapshots to the ECS fact-check endpoint.
    Emits fact_check_result events with the verdict once ECS responds.
    """

    def __init__(self, event_bus: EventBus, config: Optional[LangChainSinkConfig] = None):
        self.event_bus = event_bus
        self.config = config or LangChainSinkConfig()
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max(1, self.config.max_parallel))
        self._enabled = bool(self.config.backend_url and self.config.access_token)
        self._utterances: Dict[str, Dict[str, Any]] = {}
        self._last_session_id: Optional[str] = None
        self._last_snapshot_hash: Optional[int] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        self._seen_claims: set[str] = set()

        if not self._enabled:
            logger.warning(
                "[LangChainSink] Disabled – missing FACTICITY_BACKEND_URL or FACTICITY_ACCESS_TOKEN env vars."
            )
            return

        backend = self.config.backend_url.rstrip("/")
        self._fact_check_url = f"{backend}{self.config.fact_check_path}"
        self._extract_url = f"{backend}{self.config.extract_claim_path}"
        event_bus.subscribe(self._handle_event)
        logger.info("[LangChainSink] Initialized; streaming results to %s", self._fact_check_url)

        if self.config.snapshot_enabled:
            self._snapshot_task = asyncio.create_task(self._snapshot_loop())

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_event(self, event: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        ev_type = event.get("type")
        if ev_type not in ("partial_utterance", "final_utterance"):
            return

        # Update rolling transcript for snapshots; we no longer push finals directly.
        self._update_transcript(event)

    # ------------------------------------------------------------------
    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            timeout = aiohttp.ClientTimeout(total=None)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def close(self) -> None:
        if self._snapshot_task:
            self._snapshot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._snapshot_task
            self._snapshot_task = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    # ------------------------------------------------------------------
    # Transcript aggregation + snapshots
    # ------------------------------------------------------------------
    def _update_transcript(self, event: Dict[str, Any]) -> None:
        utt_id = event.get("utterance_id")
        if not utt_id:
            return

        existing = self._utterances.get(
            utt_id,
            {
                "id": utt_id,
                "speaker": event.get("speaker") or "",
                "t_start": event.get("t_start"),
                "t_end": event.get("t_end"),
                "latest": "",
                "final": None,
                "is_final": False,
            },
        )

        ev_type = event.get("type")
        if ev_type == "final_utterance":
            txt = (event.get("raw_text") or "").strip()
            if txt:
                existing["final"] = txt
                existing["latest"] = txt
            existing["is_final"] = True
            existing["t_end"] = event.get("t_end", existing["t_end"])
        else:  # partial
            txt = (event.get("text") or event.get("raw_text") or "").strip()
            if txt:
                existing["latest"] = txt
            existing["is_final"] = False
            existing["t_end"] = event.get("t_end", existing["t_end"])

        existing["speaker"] = event.get("speaker") or existing["speaker"] or ""
        existing["t_start"] = event.get("t_start", existing["t_start"])

        self._utterances[utt_id] = existing
        self._last_session_id = event.get("session_id") or self._last_session_id
        self._last_snapshot_hash = None  # mark as changed

    def _build_snapshot_text(self) -> str:
        items: List[Dict[str, Any]] = list(self._utterances.values())
        items.sort(key=lambda u: (u.get("t_start") or 0))

        if self.config.snapshot_max_utterances > 0:
            items = items[-self.config.snapshot_max_utterances :]

        chunks: List[str] = []
        for u in items:
            txt = u.get("final") or u.get("latest") or ""
            if txt:
                chunks.append(txt)
        return " ".join(chunks).strip()

    async def _snapshot_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.snapshot_interval_sec)
            snapshot_text = self._build_snapshot_text()
            if not snapshot_text:
                continue

            snap_hash = hash(snapshot_text)
            if self._last_snapshot_hash is not None and snap_hash == self._last_snapshot_hash:
                continue

            self._last_snapshot_hash = snap_hash
            claims = await self._extract_claims(snapshot_text)
            if not claims:
                continue

            for claim in claims:
                if not claim or claim in self._seen_claims:
                    continue
                self._seen_claims.add(claim)
                synthetic_event = {
                    "type": "claim",
                    "session_id": self._last_session_id,
                    "utterance_id": f"claim_{uuid.uuid4().hex}",
                    "raw_text": claim,
                    "speaker": "",
                    "t_end": datetime.now(tz=timezone.utc).isoformat(),
                }
                asyncio.create_task(self._process_utterance(synthetic_event))

    async def _extract_claims(self, text: str) -> List[str]:
        """
        Send snapshot text to extract-claim endpoint and return a list of claim strings.
        """
        headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
            "Validator": self.config.validator,
            "Frontend": self.config.frontend,
        }

        session = await self._get_http_session()
        try:
            async with session.post(
                self._extract_url,
                json={
                    "query": text,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "claim_extraction": True,
                },
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "[LangChainSink] extract-claim HTTP %s: %s",
                        resp.status,
                        body[:500],
                    )
                    return []
                try:
                    payload = await resp.json()
                except Exception:
                    logger.warning("[LangChainSink] extract-claim: unable to parse JSON response")
                    return []
        except Exception as exc:
            logger.exception("[LangChainSink] extract-claim request failed: %s", exc)
            return []

        claims_field = payload.get("claims") or payload.get("data") or []
        claims: List[str] = []
        for item in claims_field:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    claims.append(text)
            elif isinstance(item, dict):
                value = item.get("claim")
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        claims.append(text)
        if claims:
            preview = claims[:3]
            logger.info("[LangChainSink] Extracted %d claim(s): %s%s",
                        len(claims),
                        preview,
                        " ..." if len(claims) > len(preview) else "")
        return claims

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

        # Read full body (covers both JSON and streamed text in practice)
        body_text = await response.text()
        body_stripped = body_text.strip()

        # Try whole-body JSON
        if body_stripped:
            try:
                direct_json = json.loads(body_stripped)
                if isinstance(direct_json, dict):
                    if "final" in direct_json and isinstance(direct_json["final"], str):
                        try:
                            return json.loads(direct_json["final"])
                        except json.JSONDecodeError:
                            logger.warning("[LangChainSink] Unable to decode final payload (direct JSON): %s", direct_json["final"])
                            return None
                    if "Classification" in direct_json or "overall_assessment" in direct_json:
                        return direct_json
            except json.JSONDecodeError:
                # fall through to line-by-line parsing
                pass

        # Try line-by-line (NDJSON / progress)
        for line in body_stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                maybe_json = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(maybe_json, dict):
                if "final" in maybe_json and isinstance(maybe_json["final"], str):
                    try:
                        return json.loads(maybe_json["final"])
                    except json.JSONDecodeError:
                        logger.warning("[LangChainSink] Unable to decode final payload (line): %s", maybe_json["final"])
                        return None
                if "Classification" in maybe_json or "overall_assessment" in maybe_json:
                    return maybe_json

        logger.warning("[LangChainSink] fact-check ended without parsable result; body=%s", body_text[:500])
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











