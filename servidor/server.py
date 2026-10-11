"""
Servidor intermedio: ESP32 <-> Gemini Live API.

La ESP32 se conecta por WebSocket a /ws y envía audio del micrófono
(PCM 16-bit, 16 kHz, mono). Este servidor lo reenvía a Gemini Live y
devuelve el audio de respuesta (PCM 16-bit, 24 kHz, mono) a la ESP32.

Mensajes de texto hacia la ESP32 (JSON):
  {"state": "speaking"}   -> empieza a hablar (la ESP32 deja de escuchar)
  {"state": "listening"}  -> terminó de hablar (la ESP32 vuelve a escuchar)

Google corta cada conexión de Live API a los ~10 minutos. Este servidor usa
"reanudación de sesión" para abrir una conexión nueva sin perder la conversación,
sin que la ESP32 se entere.
"""

import asyncio
import array
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robot")

# ---------------- Configuración (variables de entorno) ----------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]          # obligatoria
ROBOT_KEY = os.environ.get("ROBOT_KEY", "")            # clave compartida con la ESP32
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-live")
VOICE = os.environ.get("GEMINI_VOICE", "Puck")         # Puck, Kore, Charon, Aoede, Fenrir...
# Si el modelo rechaza alguno de estos extras, ponlos en 0 desde Render.
USE_TRANSCRIPTS = os.environ.get("GEMINI_TRANSCRIPTS", "1") == "1"   # muestra "Tú:" y "Robot:" en los logs
USE_COMPRESSION = os.environ.get("GEMINI_COMPRESSION", "1") == "1"   # conversaciones largas sin límite
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


def build_config(handle: Optional[str]) -> types.LiveConnectConfig:
    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=build_system_prompt(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
        # Permite retomar la conversación al abrir una conexión nueva.
        session_resumption=types.SessionResumptionConfig(handle=handle),
    )
    if USE_TRANSCRIPTS:
        kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
    if USE_COMPRESSION:
        kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
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
    return {"ok": True, "model": MODEL}


# ---------------- Tareas ----------------
async def read_robot(ws: WebSocket, in_q: asyncio.Queue, closed: asyncio.Event):
    """Lee el audio de la ESP32 durante toda la conexión, aunque Gemini se esté renovando."""
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if not data:
                continue
            if in_q.full():  # si Gemini se está reconectando, se descarta lo más viejo
                try:
                    in_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            in_q.put_nowait(data)
    except Exception as e:  # noqa: BLE001
        log.info("Lectura de la ESP32 terminó: %r", e)
    finally:
        closed.set()


async def send_to_gemini(session, in_q: asyncio.Queue):
    """Audio del micrófono -> Gemini (con filtro de silencio opcional)."""
    preroll = deque(maxlen=PREROLL_CHUNKS)
    last_voice = 0.0
    gate_open = False

    async def send(data: bytes):
        await session.send_realtime_input(
            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
        )

    while True:
        data = await in_q.get()

        if GATE_RMS <= 0:
            await send(data)
            continue

        now = time.monotonic()
        if rms_int16(data) >= GATE_RMS:
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


async def gemini_to_queue(session, out_q: asyncio.Queue, holder: dict):
    """Respuestas de Gemini -> cola de audio hacia la ESP32."""
    while True:
        async for resp in session.receive():
            # Guardar el identificador que permite retomar la conversación.
            sru = getattr(resp, "session_resumption_update", None)
            if sru is not None and getattr(sru, "new_handle", None):
                holder["handle"] = sru.new_handle
            ga = getattr(resp, "go_away", None)
            if ga is not None:
                log.info("Google avisa que va a cerrar la conexión (quedan %s)", getattr(ga, "time_left", "?"))

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

    in_q: asyncio.Queue = asyncio.Queue(maxsize=200)   # ~6 s de audio de la ESP32
    closed = asyncio.Event()
    reader = asyncio.create_task(read_robot(ws, in_q, closed))
    holder = {"handle": None}
    quick_failures = 0

    try:
        while not closed.is_set():
            started = time.monotonic()
            try:
                async with client.aio.live.connect(
                    model=MODEL, config=build_config(holder["handle"])
                ) as session:
                    log.info(
                        "Sesión Gemini Live abierta (%s, %s)",
                        MODEL,
                        "retomada" if holder["handle"] else "nueva",
                    )
                    out_q: asyncio.Queue = asyncio.Queue()
                    await ws.send_text('{"state":"listening"}')
                    tasks = [
                        asyncio.create_task(send_to_gemini(session, in_q)),
                        asyncio.create_task(gemini_to_queue(session, out_q, holder)),
                        asyncio.create_task(queue_to_robot(ws, out_q)),
                        asyncio.create_task(closed.wait()),
                    ]
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for t in done:
                        if not t.cancelled() and t.exception():
                            log.warning("La conexión con Gemini terminó: %r", t.exception())
            except Exception as e:  # noqa: BLE001
                log.error("Error con Gemini: %r", e)

            if closed.is_set():
                break

            quick_failures = quick_failures + 1 if time.monotonic() - started < 5 else 0
            if quick_failures >= 3:
                log.error("Gemini rechaza la conexión repetidamente; se cierra la conexión con la ESP32")
                await asyncio.sleep(10)  # evita que la ESP32 reintente de inmediato
                break
            log.info("Renovando la conexión con Gemini...")
            await asyncio.sleep(0.3)
    finally:
        reader.cancel()
        log.info("ESP32 desconectada")
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
