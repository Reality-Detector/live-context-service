import os
import asyncio
import websockets
from dotenv import load_dotenv

# Load .env.local
load_dotenv(".env.local")

async def main():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("❌ No DEEPGRAM_API_KEY found in .env.local")
        return

    url = "wss://api.deepgram.com/v1/listen?punctuate=true"
    headers = {"Authorization": f"Token {api_key}"}

    print("Connecting to Deepgram...")
    try:
        ws = await websockets.connect(url, extra_headers=headers)
        print("✅ Connected — API key is valid.")
        await ws.close()
    except Exception as e:
        print("❌ Failed:", e)

asyncio.run(main())
