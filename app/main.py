import os, json, base64, logging, re
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from openai import OpenAI

APP_VERSION = "0.1.1"
ACCESS_KEY = os.getenv("ACCESS_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-5-mini").strip()
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe").strip()
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "alloy").strip()
DEBUG_PROTOCOL = os.getenv("DEBUG_PROTOCOL", "0").lower() in {"1", "true", "yes", "on"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coretalk-relay")
app = FastAPI(title="CoreTalk Relay Starter", version=APP_VERSION)


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OPENAI_API_KEY is not configured on the server")
    return OpenAI(api_key=OPENAI_API_KEY)


def _all_candidate_values(headers, query_params):
    values = []
    for _, v in headers.items():
        if v:
            values.append(v.strip())
            if v.lower().startswith("bearer "):
                values.append(v[7:].strip())
    for _, v in query_params.items():
        if v:
            values.append(str(v).strip())
    return values


def auth_ok(headers, query_params) -> bool:
    if not ACCESS_KEY:
        return True
    return ACCESS_KEY in _all_candidate_values(headers, query_params)


def require_auth(request: Request):
    if not auth_ok(request.headers, request.query_params):
        log.warning("AUTH_FAIL %s %s header_names=%s", request.method, request.url.path, list(request.headers.keys()))
        raise HTTPException(401, "Invalid access key")


def redact_text(s: str) -> str:
    s = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***REDACTED***", s)
    s = re.sub(
        r'("?(?:api[_-]?key|access[_-]?key|apiKey|accessKey|token)"?\s*[:=]\s*")([^"]+)(")',
        r'\1***REDACTED***\3',
        s,
        flags=re.I,
    )
    return s[:4000]


async def log_unknown_request(request: Request):
    content_type = request.headers.get("content-type", "")
    summary = {
        "method": request.method,
        "path": request.url.path,
        "query_keys": list(request.query_params.keys()),
        "header_names": list(request.headers.keys()),
        "content_type": content_type,
        "content_length": request.headers.get("content-length"),
        "auth_match": auth_ok(request.headers, request.query_params),
    }
    if DEBUG_PROTOCOL and ("json" in content_type or content_type.startswith("text/")):
        try:
            body = await request.body()
            summary["body_preview"] = redact_text(body.decode("utf-8", "replace"))
        except Exception as e:
            summary["body_preview_error"] = str(e)
    log.info("PROTOCOL_HTTP %s", json.dumps(summary, ensure_ascii=False))


@app.get("/")
@app.get("/health")
@app.get("/healthz")
@app.get("/status")
@app.get("/api/health")
@app.get("/api/v1/health")
async def health(request: Request):
    return {
        "ok": True,
        "status": "ok",
        "service": "coretalk-relay",
        "version": APP_VERSION,
        "openai_configured": bool(OPENAI_API_KEY),
    }


@app.post("/config")
@app.post("/api/config")
@app.post("/api/v1/config")
@app.post("/keys")
@app.post("/api/keys")
async def config_alias(request: Request):
    require_auth(request)
    await log_unknown_request(request)
    return {
        "ok": True,
        "status": "ok",
        "saved": False,
        "server_managed_openai_key": bool(OPENAI_API_KEY),
    }


@app.post("/v1/transcribe")
@app.post("/api/transcribe")
@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    require_auth(request)
    client = get_client()
    data = await file.read()
    kwargs = {
        "model": STT_MODEL,
        "file": (file.filename or "audio.wav", data, file.content_type or "application/octet-stream"),
    }
    if language:
        kwargs["language"] = language
    result = client.audio.transcriptions.create(**kwargs)
    text = getattr(result, "text", None) or str(result)
    return {"ok": True, "text": text}


@app.post("/v1/translate")
@app.post("/api/translate")
@app.post("/translate")
async def translate(request: Request):
    require_auth(request)
    body = await request.json()
    text = str(body.get("text", "")).strip()
    source_lang = str(body.get("source_lang") or body.get("source") or "auto")
    target_lang = str(body.get("target_lang") or body.get("target") or "English")
    if not text:
        raise HTTPException(400, "text is required")
    client = get_client()
    prompt = (
        f"Translate the following speech from {source_lang} to {target_lang}. "
        "Return only the natural spoken translation; do not explain.\n\n" + text
    )
    result = client.responses.create(model=TEXT_MODEL, input=prompt)
    return {"ok": True, "text": result.output_text, "translation": result.output_text}


@app.post("/v1/speak")
@app.post("/api/speak")
@app.post("/api/tts")
@app.post("/tts")
async def speak(request: Request):
    require_auth(request)
    body = await request.json()
    text = str(body.get("text", "")).strip()
    voice = str(body.get("voice") or TTS_VOICE)
    if not text:
        raise HTTPException(400, "text is required")
    client = get_client()
    speech = client.audio.speech.create(model=TTS_MODEL, voice=voice, input=text, response_format="mp3")
    audio = speech.read() if hasattr(speech, "read") else bytes(speech.content)
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/v1/interpret")
@app.post("/api/interpret")
@app.post("/interpret")
async def interpret(
    request: Request,
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("English"),
    voice: str = Form(TTS_VOICE),
):
    require_auth(request)
    client = get_client()
    data = await file.read()
    tr = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=(file.filename or "audio.wav", data, file.content_type or "application/octet-stream"),
    )
    transcript = getattr(tr, "text", None) or str(tr)
    prompt = (
        f"Translate the following speech from {source_lang} to {target_lang}. "
        "Return only the natural spoken translation; do not explain.\n\n" + transcript
    )
    rr = client.responses.create(model=TEXT_MODEL, input=prompt)
    translated = rr.output_text
    speech = client.audio.speech.create(model=TTS_MODEL, voice=voice, input=translated, response_format="mp3")
    audio = speech.read() if hasattr(speech, "read") else bytes(speech.content)
    return {
        "ok": True,
        "transcript": transcript,
        "translation": translated,
        "audio_mime": "audio/mpeg",
        "audio_base64": base64.b64encode(audio).decode("ascii"),
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def protocol_probe(path: str, request: Request):
    await log_unknown_request(request)
    require_auth(request)
    return JSONResponse({
        "ok": True,
        "status": "ok",
        "success": True,
        "message": "coretalk relay protocol probe",
        "path": "/" + path,
    })


@app.websocket("/{path:path}")
async def protocol_probe_ws(websocket: WebSocket, path: str):
    if not auth_ok(websocket.headers, websocket.query_params):
        log.warning("WS_AUTH_FAIL /%s header_names=%s", path, list(websocket.headers.keys()))
        await websocket.close(code=4401)
        return
    await websocket.accept()
    log.info(
        "PROTOCOL_WS_OPEN %s",
        json.dumps({
            "path": "/" + path,
            "query_keys": list(websocket.query_params.keys()),
            "header_names": list(websocket.headers.keys()),
        }, ensure_ascii=False),
    )
    try:
        while True:
            msg = await websocket.receive()
            entry = {"path": "/" + path, "type": msg.get("type")}
            if msg.get("text") is not None:
                text = msg["text"]
                entry["text_length"] = len(text)
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        entry["json_keys"] = list(obj.keys())
                        entry["event_type"] = obj.get("type")
                    if DEBUG_PROTOCOL:
                        entry["text_preview"] = redact_text(text)
                except Exception:
                    if DEBUG_PROTOCOL:
                        entry["text_preview"] = redact_text(text)
            if msg.get("bytes") is not None:
                entry["bytes_length"] = len(msg["bytes"])
            log.info("PROTOCOL_WS_MSG %s", json.dumps(entry, ensure_ascii=False))
            if msg.get("text") is not None:
                await websocket.send_json({"ok": True, "type": "ack"})
    except WebSocketDisconnect:
        log.info("PROTOCOL_WS_CLOSE /%s", path)
    except Exception as e:
        log.exception("PROTOCOL_WS_ERROR /%s: %s", path, e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
