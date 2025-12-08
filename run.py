# run.py
import asyncio
from dotenv import load_dotenv
load_dotenv(".env.local")


from src.live_context.core.session import Session
from src.live_context.providers.deepgram_asr import DeepgramASRProvider
from src.live_context.server.ws_server import WSServer
from src.live_context.sinks import LangChainSink


async def main():

    config = {
        "sensors": {
            "asr": {"enabled": True},
            "ocr": {"enabled": False},
        },
        "asr": {
            "provider": "deepgram",
        },
    }

    session = Session(session_id="sess_1", config=config)

    # Print all events for now
    def print_event(ev):
        print(ev)

    session.event_bus.subscribe(print_event)

    # Attach Deepgram provider
    provider = DeepgramASRProvider(session_id=session.session_id)
    session.attach_asr_provider(provider)

    # Forward final utterances to LangChainAgents
    langchain_sink = LangChainSink(session.event_bus)

    # Start session (which starts Deepgram stream)
    await session.start()

    # Start WebSocket server
    ws_server = WSServer(session)
    await ws_server.start()

    try:
        # Run forever
        await ws_server.wait_forever()
    finally:
        await langchain_sink.close()


if __name__ == "__main__":
    asyncio.run(main())
