# src/live_context/server/ws_server.py
import asyncio
import json
from typing import Optional, Set

import websockets

from ..adapters.audio_frame import AudioFrame
from ..core.session import Session


class WSServer:
    """
    Minimal WebSocket server:

    - Receives binary audio chunks from browser
    - Wraps them as AudioFrame and pushes into Session's ASR pipeline
    - Subscribes to Session.event_bus and forwards events to all clients as JSON
    """

    def __init__(self, session: Session, host: str = "0.0.0.0", port: int = 8765):
        self.session = session
        self.host = host
        self.port = port

        self._server: Optional[websockets.server.Serve] = None
        self._clients: Set[websockets.WebSocketServerProtocol] = set()

        # Subscribe to events from the session
        self.session.event_bus.subscribe(self._on_event)

    async def _handler(self, websocket, path):
        self._clients.add(websocket)
        print("Client connected:", path)
        try:
            async for message in websocket:
                # Only expect binary audio frames from client for now
                if isinstance(message, (bytes, bytearray)):
                    frame = AudioFrame.now(
                        sample_rate=48000,   # matches Deepgram URL
                        num_channels=1,
                        data=bytes(message),
                    )
                    await self.session.asr_pipeline.push_frame(frame)
                else:
                    # Ignore text messages for now (could use later for commands)
                    pass
        except Exception as e:
            print("Client error:", e)
        finally:
            self._clients.discard(websocket)
            print("Client disconnected")

            # If no more clients, you *could* stop the session/Deepgram here.
            # For now we just leave it running until process exit.

    def _on_event(self, event: dict) -> None:
        """
        Called whenever the Session emits an event.
        We forward it as JSON to all connected WS clients.
        """
        if not self._clients:
            return

        msg = json.dumps(event)

        async def _send(ws):
            try:
                await ws.send(msg)
            except Exception:
                # On failure, remove the client silently
                self._clients.discard(ws)

        # Schedule sends for all clients
        for ws in list(self._clients):
            asyncio.create_task(_send(ws))

    async def start(self):
        self._server = await websockets.serve(self._handler, self.host, self.port)
        print("WebSocket server listening on ws://{}:{}".format(self.host, self.port))

    async def wait_forever(self):
        # Keep the server running forever
        await asyncio.Future()
