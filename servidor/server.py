"""
Servidor intermedio: ESP32 <-> Gemini Live API.

La ESP32 se conecta por WebSocket a /ws y envía audio del micrófono
(PCM 16-bit, 16 kHz, mono). Este servidor lo reenvía a Gemini Live y
devuelve el audio de respuesta (PCM 16-bit, 24 kHz, mono) a la ESP32.

Mensajes de texto hacia la ESP32 (JSON):
  {"state": "speaking"}   -> empieza a hablar (la ESP32 deja de escuchar)
  {"state": "listening"}  -> terminó de hablar (la ESP32 vuelve a escuchar)
"""

import asyncio
import array
import logging
import math
import os
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robot")

# ---------------- Configuración (variables de entorno) ----------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]          # obligatoria
ROBOT_KEY = os.environ.get("ROBOT_KEY", "")            # clave compartida con la ESP32
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-native-audio-latest")
VOICE = os.environ.get("GEMINI_VOICE", "Puck")         # Puck, Kore, Charon, Aoede, Fenrir...
# Filtro de silencio: 0 = desactivado (enviar todo el audio a Gemini).
# Si lo activas (ej. 300), solo se envía audio cuando hay voz, para gastar menos cuota.
GATE_RMS = int(os.environ.get("GATE_RMS", "0"))
GATE_HANGOVER_S = 1.5   # segundos de audio que se siguen enviando tras la última voz
PREROLL_CHUNKS = 10     # ~320 ms de audio previo que se envían al abrir el filtro

OUT_RATE = 24000        # Gemini devuelve 24 kHz
LEAD_S = 0.4            # cuánto audio adelantado se permite en el buffer de la ESP32
SLICE_BYTES = 4096      # tamaño máximo de cada trozo enviado a la ESP32

BASE_DIR = Path(__file__).parent
client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI()


def build_system_prompt() -> str:
    """Se vuelve a leer negocio.md en cada conexión: editarlo no requiere tocar el código."""
    negocio_path = BASE_DIR / "negocio.md"
    negocio = negocio_path.read_text(encoding="utf-8") if negocio_path.exists() else ""
    return (
        "Eres un robot asistente con voz, amable y cercano. Hablas siempre en español "
        "(chileno neutro), con frases cortas y naturales, como en una conversación real. "
        "Responde en 1 a 3 frases salvo que te pidan más detalle. No uses listas, símbolos "
        "ni formato: tu respuesta se va a oír, no a leer.\n\n"
        "Cuando atiendas a clientes del negocio, usa SOLO la información de abajo para "
        "precios, servicios y horarios. Si te preguntan algo que no está ahí, di que no "
        "tienes ese dato y ofrece avisarle al dueño. Nunca inventes precios.\n\n"
        "=== INFORMACIÓN DEL NEGOCIO ===\n" + negocio
    )


def build_config() -> types.LiveConnectConfig:
    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=build_system_prompt(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # Permite conversaciones largas sin que la sesión expire.
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
    )
    return types.LiveConnectConfig(**kwargs)


def rms_int16(data: bytes) -> float:
    n = len(data) // 2
    if n == 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(data[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


@app.get("/")
async def health():
    return {"ok": True}


# ---------------- Tareas de una conexión ----------------
async def robot_to_gemini(ws: WebSocket, session):
    """Audio del micrófono (ESP32) -> Gemini."""
    preroll = deque(maxlen=PREROLL_CHUNKS)
    last_voice = 0.0
    gate_open = False

    async def send(data: bytes):
        await session.send_realtime_input(
            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
        )

    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            return
        data = msg.get("bytes")
        if not data:
            continue

        if GATE_RMS <= 0:
            await send(data)
            continue

        now = time.monotonic()
        level = rms_int16(data)
        if level >= GATE_RMS:
            last_voice = now
        if now - last_voice <= GATE_HANGOVER_S:
            if not gate_open:
                gate_open = True
                for old in preroll:
                    await send(old)
                preroll.clear()
            await send(data)
        else:
            gate_open = False
            preroll.append(data)


async def gemini_to_queue(session, out_q: asyncio.Queue):
    """Respuestas de Gemini -> cola de audio hacia la ESP32."""
    while True:
        async for resp in session.receive():
            sc = resp.server_content
            if resp.data:
                await out_q.put(resp.data)
            if sc:
                it = getattr(sc, "input_transcription", None)
                if it and getattr(it, "text", None):
                    log.info("Tú: %s", it.text)
                ot = getattr(sc, "output_transcription", None)
                if ot and getattr(ot, "text", None):
                    log.info("Robot: %s", ot.text)
                if getattr(sc, "interrupted", False):
                    while not out_q.empty():
                        out_q.get_nowait()
                    await out_q.put(None)
                if getattr(sc, "turn_complete", False):
                    await out_q.put(None)  # marca de fin de turno


async def queue_to_robot(ws: WebSocket, out_q: asyncio.Queue):
    """Cola -> ESP32, al ritmo de reproducción para no desbordar su memoria."""
    speaking = False
    t0 = 0.0
    sent_s = 0.0
    while True:
        chunk = await out_q.get()
        if chunk is None:
            if speaking:
                # Esperar a que la ESP32 termine de reproducir lo ya enviado.
                remaining = sent_s - (time.monotonic() - t0)
                await asyncio.sleep(max(0.0, remaining) + 0.3)
                await ws.send_text('{"state":"listening"}')
                speaking = False
            continue

        if not speaking:
            speaking = True
            t0 = time.monotonic()
            sent_s = 0.0
            await ws.send_text('{"state":"speaking"}')

        for i in range(0, len(chunk), SLICE_BYTES):
            piece = chunk[i : i + SLICE_BYTES]
            ahead = sent_s - (time.monotonic() - t0)
            if ahead > LEAD_S:
                await asyncio.sleep(ahead - LEAD_S)
            await ws.send_bytes(piece)
            sent_s += len(piece) / 2 / OUT_RATE


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if ROBOT_KEY and ws.query_params.get("key") != ROBOT_KEY:
        await ws.close(code=1008)
        return
    await ws.accept()
    log.info("ESP32 conectada")

    try:
        async with client.aio.live.connect(model=MODEL, config=build_config()) as session:
            log.info("Sesión Gemini Live abierta (%s)", MODEL)
            out_q: asyncio.Queue = asyncio.Queue()
            tasks = [
                asyncio.create_task(robot_to_gemini(ws, session)),
                asyncio.create_task(gemini_to_queue(session, out_q)),
                asyncio.create_task(queue_to_robot(ws, out_q)),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                if t.exception():
                    log.error("Error en tarea: %r", t.exception())
    except Exception as e:  # noqa: BLE001
        log.error("Error con Gemini: %r", e)
    finally:
        log.info("ESP32 desconectada")
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
