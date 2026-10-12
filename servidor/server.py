"""
Servidor intermedio, modo "por piezas":

  ESP32 (micrófono) --> detecta que terminaste de hablar
        --> Groq Whisper (oír) --> Groq LLM (pensar, en streaming)
        --> Edge TTS (hablar, oración por oración) --> ESP32 (parlante)

La ESP32 envía audio PCM 16-bit, 16 kHz, mono por el WebSocket /ws y recibe PCM 16-bit,
24 kHz, mono. Mensajes de texto hacia la ESP32 (JSON):
  {"state": "speaking"}   -> empieza a hablar (la ESP32 deja de escuchar)
  {"state": "listening"}  -> terminó de hablar (la ESP32 vuelve a escuchar)

Para que la espera sea corta, la respuesta se va diciendo apenas hay la primera oración,
mientras el modelo sigue escribiendo el resto.

Variables de entorno (Render -> Environment):
  GROQ_API_KEY        obligatoria (console.groq.com, gratis)
  ROBOT_KEY           clave compartida con la ESP32
  GROQ_LLM_MODEL      por defecto openai/gpt-oss-120b
  GROQ_STT_MODEL      por defecto whisper-large-v3-turbo
  TTS_VOICE           por defecto es-CL-CatalinaNeural (otra: es-CL-LorenzoNeural)
  TTS_RATE            velocidad de la voz, por defecto +8%
  END_SILENCE_MS      silencio para dar por terminada tu frase, por defecto 600
  VAD_MIN_RMS         sensibilidad: sube si se activa con ruido, baja si no te detecta (350)
"""

import array
import asyncio
import io
import logging
import math
import os
import re
import time
import wave
from collections import deque
from pathlib import Path
from typing import Optional

import edge_tts
import miniaudio
from fastapi import FastAPI, WebSocket
from groq import AsyncGroq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robot")

# ---------------- Configuración ----------------
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ROBOT_KEY = os.environ.get("ROBOT_KEY", "")
STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
TTS_VOICE = os.environ.get("TTS_VOICE", "es-CL-CatalinaNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+8%")

VAD_DEBUG = os.environ.get("VAD_DEBUG", "1") == "1"        # muestra el nivel del micrófono en reposo
VAD_MIN_RMS = int(os.environ.get("VAD_MIN_RMS", "350"))
VAD_MULT = float(os.environ.get("VAD_MULT", "3.0"))        # umbral = ruido de fondo x esto
END_SILENCE_MS = int(os.environ.get("END_SILENCE_MS", "600"))
# Frases que Whisper inventa cuando recibe silencio o ruido
HALLUCINATIONS = (
    "gracias por ver", "suscríbete", "suscribete", "subtítulos realizados",
    "subtitulado por", "amara.org",
)
MIN_SPEECH_MS = 300          # frases más cortas se descartan (clics, golpes)
MAX_UTTERANCE_S = 15          # corta frases larguísimas
MAX_HISTORY = 12              # mensajes de memoria de la conversación

IN_RATE = 16000               # lo que envía la ESP32
OUT_RATE = 24000              # lo que reproduce la ESP32
FRAME_BYTES = 960             # 30 ms de audio a 16 kHz
LEAD_S = 0.4                  # audio adelantado permitido en el buffer de la ESP32
SLICE_BYTES = 1024            # tamaño de cada trozo enviado a la ESP32 (chico = no se fragmenta)

BASE_DIR = Path(__file__).parent
client = AsyncGroq(api_key=GROQ_API_KEY)
app = FastAPI()


def build_system_prompt() -> str:
    """Se vuelve a leer negocio.md en cada conexión: editarlo no requiere tocar el código."""
    negocio_path = BASE_DIR / "negocio.md"
    negocio = negocio_path.read_text(encoding="utf-8") if negocio_path.exists() else ""
    return (
        "Eres un robot asistente con voz, amable y cercano. Hablas siempre en español "
        "(chileno neutro), con frases cortas y naturales, como en una conversación real. "
        "Responde en 1 a 3 frases salvo que te pidan más detalle. No uses listas, símbolos, "
        "emojis ni formato: tu respuesta se va a oír, no a leer. Escribe los números y precios "
        "con palabras, por ejemplo 'cinco mil pesos'.\n\n"
        "Cuando atiendas a clientes del negocio, usa SOLO la información de abajo para "
        "precios, servicios y horarios. Si te preguntan algo que no está ahí, di que no "
        "tienes ese dato y ofrece avisarle al dueño. Nunca inventes precios.\n\n"
        "=== INFORMACIÓN DEL NEGOCIO ===\n" + negocio
    )


# ---------------- Utilidades de audio ----------------
def rms_int16(data: bytes) -> float:
    """Volumen de un trozo de audio, ignorando el desfase constante (DC) que traen algunos micrófonos."""
    n = len(data) // 2
    if n == 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(data[: n * 2])
    mean = sum(samples) / n
    return math.sqrt(sum((s - mean) ** 2 for s in samples) / n)


def remove_dc(pcm: bytes) -> bytes:
    """Quita el desfase constante del audio antes de mandarlo a Whisper."""
    n = len(pcm) // 2
    if n == 0:
        return pcm
    s = array.array("h")
    s.frombytes(pcm[: n * 2])
    mean = int(sum(s) / n)
    if abs(mean) < 8:
        return pcm
    for i in range(n):
        v = s[i] - mean
        s[i] = 32767 if v > 32767 else (-32768 if v < -32768 else v)
    return s.tobytes()


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class Segmenter:
    """Detecta cuándo empiezas y cuándo terminas de hablar (por volumen, con ruido de fondo adaptativo)."""

    def __init__(self):
        self.pending = bytearray()
        self.noise = 150.0
        self.idle_frames = 0
        self._clear()

    def _clear(self):
        self.pre = deque(maxlen=10)          # ~300 ms previos al inicio de la voz
        self.speech = bytearray()
        self.in_speech = False
        self.run_voiced = 0
        self.voiced_frames = 0
        self.silence_frames = 0
        self.peak = 0.0

    def reset(self):
        self.pending.clear()
        self._clear()

    def feed(self, data: bytes) -> Optional[bytes]:
        """Recibe audio; devuelve la frase completa (PCM) cuando detecta que terminaste."""
        self.pending += data
        while len(self.pending) >= FRAME_BYTES:
            frame = bytes(self.pending[:FRAME_BYTES])
            del self.pending[:FRAME_BYTES]
            out = self._frame(frame)
            if out is not None:
                self.pending.clear()
                return out
        return None

    def _frame(self, frame: bytes) -> Optional[bytes]:
        level = rms_int16(frame)
        threshold = max(VAD_MIN_RMS, self.noise * VAD_MULT)
        voiced = level >= threshold

        if not self.in_speech:
            self.pre.append(frame)
            if VAD_DEBUG:
                self.idle_frames += 1
                if self.idle_frames % 200 == 0:      # cada ~6 s
                    log.info(
                        "Micrófono en reposo: nivel %.0f, ruido de fondo %.0f, umbral %.0f",
                        level, self.noise, threshold,
                    )
            if not voiced:
                self.noise = 0.97 * self.noise + 0.03 * level
                self.run_voiced = 0
                return None
            self.run_voiced += 1
            if self.run_voiced >= 3:  # ~90 ms de voz seguida
                self.in_speech = True
                self.speech = bytearray(b"".join(self.pre))
                self.voiced_frames = self.run_voiced
                self.silence_frames = 0
                self.peak = level
            return None

        self.speech += frame
        self.peak = max(self.peak, level)
        if voiced:
            self.voiced_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        too_long = len(self.speech) >= MAX_UTTERANCE_S * IN_RATE * 2
        if self.silence_frames * 30 >= END_SILENCE_MS or too_long:
            speech_ms = self.voiced_frames * 30
            out = bytes(self.speech) if (speech_ms >= MIN_SPEECH_MS and not too_long) else None
            if too_long:
                log.warning(
                    "Ruido continuo de %d s (nivel pico %.0f): se descarta. Revisa el micrófono.",
                    MAX_UTTERANCE_S, self.peak,
                )
            if out:
                log.info(
                    "Voz detectada: %.1f s (pico %.0f, ruido de fondo %.0f)",
                    len(out) / 2 / IN_RATE, self.peak, self.noise,
                )
            self._clear()
            return out
        return None


# ---------------- Oír, pensar, hablar ----------------
async def transcribe(pcm: bytes, rate: int = IN_RATE) -> str:
    wav = pcm_to_wav(remove_dc(pcm), rate)
    r = await client.audio.transcriptions.create(
        file=("audio.wav", wav), model=STT_MODEL, language="es", temperature=0.0
    )
    return (getattr(r, "text", "") or "").strip()


async def llm_stream(messages: list):
    kwargs = dict(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        max_completion_tokens=600,
        temperature=0.5,
    )
    if "gpt-oss" in LLM_MODEL:
        kwargs["reasoning_effort"] = "low"   # menos "pensar" interno = menos espera
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if not chunk.choices:
            continue
        text = getattr(chunk.choices[0].delta, "content", None)
        if text:
            yield text


_SPLIT = re.compile(r"(?<=[.!?…:;])\s+|\n+")


def clean_for_speech(t: str) -> str:
    t = re.sub(r"[*_#`>|~\[\]]", "", t)
    return re.sub(r"\s+", " ", t).strip()


async def sentences(text_stream):
    """Corta el texto que llega en trocitos en oraciones, para empezar a hablar cuanto antes."""
    buf = ""
    first = True
    async for piece in text_stream:
        buf += piece
        while True:
            m = _SPLIT.search(buf)
            if m:
                s, buf = buf[: m.start()], buf[m.end():]
            elif first and len(buf) >= 45 and buf.find(",", 20) != -1:
                i = buf.find(",", 20)           # la primera oración puede salir en una coma
                s, buf = buf[: i + 1], buf[i + 1:]
            else:
                break
            s = clean_for_speech(s)
            if s:
                first = False
                yield s
    s = clean_for_speech(buf)
    if s:
        yield s


def _decode_mp3(mp3: bytes) -> bytes:
    dec = miniaudio.decode(
        mp3,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=OUT_RATE,
    )
    return dec.samples.tobytes()


_edge_bad_until = 0.0     # si Edge TTS falla, se usa la voz de respaldo un rato sin reintentar


async def _edge_mp3(text: str) -> bytes:
    mp3 = bytearray()

    async def collect():
        com = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
        async for ch in com.stream():
            if ch["type"] == "audio":
                mp3.extend(ch["data"])

    await asyncio.wait_for(collect(), timeout=6)
    if not mp3:
        raise RuntimeError("Edge TTS no devolvió audio")
    return bytes(mp3)


def _gtts_mp3_sync(text: str) -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang="es", tld="com.mx").write_to_fp(buf)
    return buf.getvalue()


async def _gtts_mp3(text: str) -> bytes:
    data = await asyncio.wait_for(asyncio.to_thread(_gtts_mp3_sync, text), timeout=8)
    if not data:
        raise RuntimeError("gTTS no devolvió audio")
    return data


async def tts_pcm(text: str, engine: str = "auto") -> bytes:
    """Convierte texto a audio. 'auto': Edge TTS (voz chilena) y, si falla, gTTS (voz de respaldo)."""
    global _edge_bad_until
    mp3 = None
    if engine in ("auto", "edge") and (engine == "edge" or time.monotonic() >= _edge_bad_until):
        try:
            mp3 = await _edge_mp3(text)
        except Exception as e:  # noqa: BLE001
            if engine == "edge":
                raise
            _edge_bad_until = time.monotonic() + 120
            log.warning("Edge TTS falló (%s); uso la voz de respaldo durante 2 min", type(e).__name__)
    if mp3 is None:
        mp3 = await _gtts_mp3(text)
    return await asyncio.to_thread(_decode_mp3, mp3)


class Speaker:
    """Envía audio a la ESP32 al ritmo de reproducción para no desbordar su memoria."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.speaking = False
        self.t0 = 0.0
        self.sent_s = 0.0

    async def play(self, pcm: bytes):
        if not self.speaking:
            self.speaking = True
            self.t0 = time.monotonic()
            self.sent_s = 0.0
            await self.ws.send_text('{"state":"speaking"}')
        for i in range(0, len(pcm), SLICE_BYTES):
            piece = pcm[i : i + SLICE_BYTES]
            now = time.monotonic()
            if self.sent_s < now - self.t0:          # el buffer de la ESP32 se vació: reiniciar la cuenta
                self.t0 = now - self.sent_s
            ahead = self.sent_s - (now - self.t0)
            if ahead > LEAD_S:
                await asyncio.sleep(ahead - LEAD_S)
            await self.ws.send_bytes(piece)
            self.sent_s += len(piece) / 2 / OUT_RATE

    async def finish(self):
        if self.speaking:
            remaining = self.sent_s - (time.monotonic() - self.t0)
            await asyncio.sleep(max(0.0, remaining) + 0.3)
            await self.ws.send_text('{"state":"listening"}')
            self.speaking = False


async def run_turn(ws: WebSocket, session: dict, pcm: bytes):
    """Un turno completo: oír -> pensar -> hablar."""
    spk = Speaker(ws)
    t_start = time.monotonic()
    try:
        text = await transcribe(pcm)
        t_stt = time.monotonic() - t_start
        if len(re.sub(r"[\W_]+", "", text)) < 2:     # vacío o solo "..." / signos
            log.info("(no se entendió nada, se ignora: %r)", text)
            return
        if any(h in text.lower() for h in HALLUCINATIONS):
            log.info("(Whisper inventó texto por ruido, se ignora: %r)", text)
            return
        log.info("Tú: %s   [oír: %.1f s]", text, t_stt)

        history = session["history"]
        history.append({"role": "user", "content": text})
        del history[:-MAX_HISTORY]
        messages = [{"role": "system", "content": session["system"]}] + history

        sent_q: asyncio.Queue = asyncio.Queue()
        pcm_q: asyncio.Queue = asyncio.Queue()
        reply_parts: list = []

        async def produce():          # el modelo escribe, oración por oración
            try:
                async for s in sentences(llm_stream(messages)):
                    reply_parts.append(s)
                    await sent_q.put(s)
            finally:
                await sent_q.put(None)

        async def synth():            # cada oración se convierte en voz
            try:
                while True:
                    s = await sent_q.get()
                    if s is None:
                        break
                    try:
                        await pcm_q.put(await tts_pcm(s))
                    except Exception as e:  # noqa: BLE001
                        log.error("La voz falló en una oración: %r", e)
            finally:
                await pcm_q.put(None)

        async def play():             # y se envía a la ESP32 sin esperar a que termine la respuesta
            first = True
            while True:
                p = await pcm_q.get()
                if p is None:
                    break
                if first:
                    first = False
                    log.info("Espera total hasta empezar a hablar: %.1f s", time.monotonic() - t_start)
                await spk.play(p)

        await asyncio.gather(produce(), synth(), play())
        await spk.finish()

        reply = " ".join(reply_parts)
        log.info("Robot: %s   [turno completo: %.1f s]", reply, time.monotonic() - t_start)
        history.append({"role": "assistant", "content": reply})

    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        log.error("Error en el turno: %r", e)
        try:
            await spk.play(await tts_pcm("Perdón, tuve un problema. ¿Puedes repetirlo?"))
            await spk.finish()
        except Exception as e2:  # noqa: BLE001
            log.error("Tampoco se pudo avisar por voz: %r", e2)
            try:
                await ws.send_text('{"state":"listening"}')
            except Exception:  # noqa: BLE001
                pass


# ---------------- Páginas y WebSocket ----------------
@app.get("/")
async def health():
    return {"ok": True, "modo": "por piezas", "cerebro": LLM_MODEL, "oido": STT_MODEL, "voz": TTS_VOICE}


@app.get("/diag")
async def diag(key: str = ""):
    """Prueba cada pieza sin micrófono: abre /diag?key=TU_ROBOT_KEY en el navegador."""
    if ROBOT_KEY and key != ROBOT_KEY:
        return {"error": "falta ?key=TU_ROBOT_KEY"}
    out: dict = {"oido": STT_MODEL, "cerebro": LLM_MODEL, "voz": TTS_VOICE}

    pcm = None
    for label, eng in (("1a_voz_edge", "edge"), ("1b_voz_respaldo", "gtts")):
        t = time.monotonic()
        try:
            if eng == "edge":
                mp3 = await _edge_mp3("Hola, soy tu robot. Estoy probando mi voz.")
            else:
                mp3 = await _gtts_mp3("Hola, soy tu robot. Estoy probando mi voz.")
            p = await asyncio.to_thread(_decode_mp3, mp3)
            pcm = pcm or p
            out[label] = f"OK ({time.monotonic() - t:.1f} s, {len(p) / 2 / OUT_RATE:.1f} s de audio)"
        except Exception as e:  # noqa: BLE001
            out[label] = f"FALLA: {type(e).__name__}: {str(e)[:120]}"
    try:
        from importlib.metadata import version as _v
        out["edge_tts_version"] = _v("edge-tts")
    except Exception:  # noqa: BLE001
        out["edge_tts_version"] = "?"

    if pcm:
        t = time.monotonic()
        try:
            text = await transcribe(pcm, OUT_RATE)
            out["2_oido"] = f"OK ({time.monotonic() - t:.1f} s): «{text}»"
        except Exception as e:  # noqa: BLE001
            out["2_oido"] = "FALLA: " + repr(e)[:300]
    else:
        out["2_oido"] = "omitido (primero falló la voz)"

    t = time.monotonic()
    try:
        parts = []
        async for p in llm_stream([{"role": "user", "content": "Responde solo con la palabra: listo"}]):
            parts.append(p)
        out["3_cerebro"] = f"OK ({time.monotonic() - t:.1f} s): {''.join(parts)[:60]}"
    except Exception as e:  # noqa: BLE001
        out["3_cerebro"] = "FALLA: " + repr(e)[:300]
    return out


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if ROBOT_KEY and ws.query_params.get("key") != ROBOT_KEY:
        await ws.close(code=1008)
        return
    await ws.accept()
    log.info("ESP32 conectada")

    session = {"history": [], "system": build_system_prompt()}
    seg = Segmenter()
    turn_task: Optional[asyncio.Task] = None

    try:
        await ws.send_text('{"state":"listening"}')
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if not data:
                continue
            if turn_task is not None and not turn_task.done():
                seg.reset()          # el robot está pensando o hablando: se ignora el micrófono
                continue
            pcm = seg.feed(data)
            if pcm is not None:
                turn_task = asyncio.create_task(run_turn(ws, session, pcm))
    except Exception as e:  # noqa: BLE001
        log.info("Conexión con la ESP32 terminó: %r", e)
    finally:
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
        log.info("ESP32 desconectada")
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
