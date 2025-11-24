# live-context-service
ASR + OCR ingestion layer that feeds claims into LangChainAgents.




live-context-service/
├── src/
│   └── live_context/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── session.py              # session lifecycle + config
│       │   ├── event_bus.py            # pub/sub + fan-out
│       │   ├── clock.py                # monotonic timing + alignment
│       │   ├── buffers.py              # audio ring buffer, frame queue
│       │   ├── normalizer.py           # text normalization hooks
│       │   ├── quality.py              # confidence + low-quality flags
│       │   └── types.py                # internal dataclasses/types
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base_asr.py             # ASR interface
│       │   ├── deepgram_asr.py         # first real provider
│       │   └── base_ocr.py             # OCR interface (stub for now)
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── asr_pipeline.py         # orchestrates ASR stream → utterances
│       │   └── ocr_pipeline.py         # stub; later screen frames → text deltas
│       │
│       ├── protocol/
│       │   ├── __init__.py
│       │   ├── events_v1.py            # canonical event schema v1
│       │   └── validate.py             # lightweight schema validation
│       │
│       ├── server/
│       │   ├── __init__.py
│       │   └── ws_server.py            # WebSocket: audio in, events out
│       │
│       └── adapters/
│           ├── __init__.py
│           └── audio_frame.py          # defines audio frame input format
│
├── demo/
│   ├── web/
│   │   └── index.html                  # tiny WS listener demo
│   └── cli.py                          # optional CLI demo (good for testing)
│
├── tests/
│   ├── test_event_bus.py
│   ├── test_session.py
│   └── test_protocol_v1.py
│
├── run.py                               # dev entrypoint
├── pyproject.toml (or requirements.txt)
├── README.md
└── .gitignore
