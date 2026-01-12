#!/usr/bin/env python3
"""
Pipecat Twilio Outbound Caller Server

This server handles outbound phone calls using:
- Twilio for telephony (your own number/account)
- ElevenLabs for TTS voices
- OpenAI for LLM
- Pipecat for the real-time pipeline

Endpoints:
- POST /call - Initiate an outbound call
- POST /twilio/voice - TwiML webhook for Twilio
- WebSocket /ws/{call_id} - Media stream for the call
"""

import asyncio
import os
import json
import uuid
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from loguru import logger
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.frames.frames import LLMRunFrame, EndFrame

from agents import get_agent_with_custom_prompt, AGENTS

load_dotenv()

# Config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+14129608589")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://91.98.137.199:4141/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8765")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8765"))

# Callback number for voicemails
CALLBACK_NUMBER = os.getenv("CALLBACK_NUMBER", "+14127062783")

# Discord webhook for call logs
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
DISCORD_CALL_LOG_CHANNEL = os.getenv("DISCORD_CALL_LOG_CHANNEL", "1460101081148428470")

# Active calls tracking
active_calls: dict = {}

# Twilio client
twilio_client = None


class CallRequest(BaseModel):
    """Request to initiate an outbound call"""
    to_number: str
    agent_id: str = "larry"
    custom_prompt: Optional[str] = None


class CallResponse(BaseModel):
    """Response from call initiation"""
    call_id: str
    call_sid: str
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan - initialize/cleanup"""
    global twilio_client
    
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client initialized")
    else:
        logger.warning("Twilio credentials not configured!")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(title="Pipecat Twilio Caller", lifespan=lifespan)


@app.post("/call", response_model=CallResponse)
async def initiate_call(request: CallRequest):
    """Initiate an outbound call with answering machine detection"""
    if not twilio_client:
        raise HTTPException(status_code=500, detail="Twilio not configured")
    
    call_id = str(uuid.uuid4())[:8]
    
    # Get agent config
    agent = get_agent_with_custom_prompt(request.agent_id, request.custom_prompt)
    
    # Store call info
    active_calls[call_id] = {
        "agent": agent,
        "to_number": request.to_number,
        "custom_prompt": request.custom_prompt,
        "status": "initiating",
        "call_sid": None,
        "conversation": [],
        "answered_by": None  # Will be "human" or "machine"
    }
    
    try:
        # Make the call via Twilio with Answering Machine Detection
        call = twilio_client.calls.create(
            to=request.to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{PUBLIC_URL}/twilio/voice?call_id={call_id}",
            status_callback=f"{PUBLIC_URL}/twilio/status?call_id={call_id}",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            record=True,
            recording_status_callback=f"{PUBLIC_URL}/twilio/recording?call_id={call_id}",
            recording_status_callback_event=["completed"],
            # Answering Machine Detection
            machine_detection="DetectMessageEnd",  # Wait for beep before proceeding
            machine_detection_timeout=30,
            async_amd=True,  # Non-blocking AMD
            async_amd_status_callback=f"{PUBLIC_URL}/twilio/amd?call_id={call_id}",
            async_amd_status_callback_method="POST",
        )
        
        active_calls[call_id]["call_sid"] = call.sid
        active_calls[call_id]["status"] = "initiated"
        
        logger.info(f"Call initiated: {call_id} -> {request.to_number} (SID: {call.sid})")
        
        return CallResponse(
            call_id=call_id,
            call_sid=call.sid,
            status="initiated"
        )
        
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        del active_calls[call_id]
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/twilio/amd")
async def twilio_amd_webhook(request: Request, call_id: str):
    """
    Twilio AMD (Answering Machine Detection) callback.
    Called when AMD determines if human or machine answered.
    """
    form = await request.form()
    answered_by = form.get("AnsweredBy", "unknown")
    call_sid = form.get("CallSid", "")
    
    logger.info(f"AMD result for {call_id}: {answered_by}")
    
    if call_id in active_calls:
        active_calls[call_id]["answered_by"] = answered_by
        
        # If voicemail detected, update the call to leave a message
        if answered_by in ["machine_end_beep", "machine_end_silence", "machine_end_other"]:
            logger.info(f"Voicemail detected for {call_id}, will leave message")
            active_calls[call_id]["is_voicemail"] = True
        elif answered_by == "human":
            logger.info(f"Human answered {call_id}, proceeding with live conversation")
            active_calls[call_id]["is_voicemail"] = False
    
    return {"ok": True}


@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request, call_id: str):
    """
    Twilio voice webhook - returns TwiML to connect media stream or leave voicemail.
    This is called by Twilio when the call is answered.
    """
    if call_id not in active_calls:
        logger.error(f"Unknown call_id: {call_id}")
        response = VoiceResponse()
        response.say("Sorry, there was an error with this call.")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")
    
    call_info = active_calls[call_id]
    response = VoiceResponse()
    
    # Check if this is a voicemail (AMD detected machine)
    if call_info.get("is_voicemail"):
        logger.info(f"Leaving voicemail for {call_id}")
        
        # Build voicemail message from custom prompt
        custom_prompt = call_info.get("custom_prompt", "")
        agent_name = call_info.get("agent", {}).get("name", "Larry")
        
        if custom_prompt:
            voicemail_text = f"Hi, this is {agent_name} calling on behalf of Jonah from Nebaura. {custom_prompt}. Please call us back at {CALLBACK_NUMBER}. That's {format_phone_for_speech(CALLBACK_NUMBER)}. Thank you, and have a great day!"
        else:
            voicemail_text = f"Hi, this is {agent_name} calling on behalf of Jonah from Nebaura. We were trying to reach you. Please call us back at {CALLBACK_NUMBER}. That's {format_phone_for_speech(CALLBACK_NUMBER)}. Thank you!"
        
        response.say(voicemail_text, voice="Polly.Matthew")
        response.hangup()
        
        return Response(content=str(response), media_type="application/xml")
    
    # Live conversation - connect media stream to WebSocket
    connect = Connect()
    
    # WebSocket URL for media stream
    ws_url = PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://")
    stream = Stream(url=f"{ws_url}/ws/{call_id}")
    stream.parameter(name="call_id", value=call_id)
    
    connect.append(stream)
    response.append(connect)
    
    logger.info(f"TwiML response for {call_id}: connecting media stream")
    
    return Response(content=str(response), media_type="application/xml")


def format_phone_for_speech(phone: str) -> str:
    """Format phone number for clear speech (digit by digit)"""
    digits = ''.join(c for c in phone if c.isdigit())
    # Format as: 4 1 2, 7 0 6, 2 7 8 3
    if len(digits) == 11 and digits[0] == '1':
        digits = digits[1:]  # Remove leading 1
    if len(digits) == 10:
        return f"{digits[0]} {digits[1]} {digits[2]}, {digits[3]} {digits[4]} {digits[5]}, {digits[6]} {digits[7]} {digits[8]} {digits[9]}"
    return ', '.join(digits)


@app.post("/twilio/status")
async def twilio_status_webhook(request: Request, call_id: str):
    """Twilio status callback - track call status changes"""
    form = await request.form()
    status = form.get("CallStatus", "unknown")
    
    if call_id in active_calls:
        active_calls[call_id]["status"] = status
        logger.info(f"Call {call_id} status: {status}")
        
        if status == "completed":
            asyncio.create_task(send_discord_call_log_delayed(call_id))
    
    return {"ok": True}


@app.post("/twilio/recording")
async def twilio_recording_webhook(request: Request, call_id: str):
    """Twilio recording callback - receive recording URL when ready"""
    form = await request.form()
    recording_url = form.get("RecordingUrl", "")
    recording_sid = form.get("RecordingSid", "")
    duration = form.get("RecordingDuration", "0")
    
    if call_id in active_calls:
        active_calls[call_id]["recording_url"] = f"{recording_url}.mp3"
        active_calls[call_id]["recording_sid"] = recording_sid
        active_calls[call_id]["recording_duration"] = duration
        logger.info(f"Recording ready for {call_id}: {recording_url}.mp3 ({duration}s)")
        
        # Send to Discord
        await send_discord_call_log(call_id, include_recording=True)
    
    return {"ok": True}


async def send_discord_call_log(call_id: str, include_recording: bool = False):
    """Send call log to Discord channel"""
    if not DISCORD_WEBHOOK_URL and not DISCORD_CALL_LOG_CHANNEL:
        return
    
    if call_id not in active_calls:
        return
    
    info = active_calls[call_id]
    to_number = info.get("to_number", "Unknown")
    agent_name = info.get("agent", {}).get("name", "Unknown")
    status = info.get("status", "unknown")
    duration = info.get("recording_duration", "0")
    recording_url = info.get("recording_url", "")
    answered_by = info.get("answered_by", "unknown")
    is_voicemail = info.get("is_voicemail", False)
    
    # Build embed
    title = "📞 Voicemail Left" if is_voicemail else "📞 Call Completed"
    color = 0xffaa00 if is_voicemail else 0x00ff00
    
    embed = {
        "title": title if status == "completed"e f"📞 Call {status.title()}",
        "color": color,
        "fields": [
            {"name": "To", "value": to_number, "inline": True},
            {"name": "Agent", "value": agent_name, "inline": True},
            {"name": "Duration", "value": f"{duration}s", "inline": True},
            {"name": "Answered By", "value": answered_by, "inline": True},
            {"name": "Call ID", "value": call_id, "inline": True},
        ],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if recording_url:
        embed["fields"].append({"name": "Recording", "value": f"[Download MP3]({recording_url})", "inline": False})
    
    # Send via Discord webhook
    try:
        async with httpx.AsyncClient() as client:
            if DISCORD_WEBHOOK_URL:
                await client.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]})
                logger.info(f"Sent call log to Discord webhook for {call_id}")
    except Exception as e:
        logger.error(f"Failed to send Discord call log: {e}")


async def send_discord_call_log_delayed(call_id: str, delay: float = 10.0):
    """Send log after delay if recording hasn't arrived"""
    await asyncio.sleep(delay)
    if call_id in active_calls and not active_calls[call_id].get("recording_url"):
        await send_discord_call_log(call_id, include_recording=False)


@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for Twilio media stream.
    This runs the Pipecat pipeline for real-time conversation.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for call {call_id}")
    
    if call_id not in active_calls:
        logger.error(f"Unknown call_id in WebSocket: {call_id}")
        await websocket.close()
        return
    
    call_info = active_calls[call_id]
    agent = call_info["agent"]
    
    # Wait for Twilio's start message to get stream_sid
    stream_sid = None
    call_sid = call_info.get("call_sid")
    
    try:
        while not stream_sid:
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            data = json.loads(msg)
            event = data.get("event")
            logger.info(f"Received Twilio event: {event}")
            
            if event == "connected":
                logger.info("Twilio WebSocket connected")
                continue
            elif event == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                call_sid = data.get("start", {}).get("callSid") or call_sid
                logger.info(f"Stream started: {stream_sid}, call: {call_sid}")
                break
            elif event == "stop":
                logger.info("Stream stopped before start")
                await websocket.close()
                return
        
        if not stream_sid:
            logger.error("No stream_sid received")
            await websocket.close()
            return
        
        # Set up the Pipecat pipeline
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN,
            params=TwilioFrameSerializer.InputParams(
                auto_hang_up=True,
                sample_rate=8000,
            )
        )
        
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
                serializer=serializer,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
            )
        )
        
        # ElevenLabs TTS with agent's voice
        tts = ElevenLabsTTSService(
            api_key=ELEVENLABS_API_KEY,
            voice_id=agent["voice_id"],
        )
        
        # Deepgram STT for speech recognition
        from deepgram import LiveOptions
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            sample_rate=8000,
            live_options=LiveOptions(
                model="nova-2",
                language="en-US",
                encoding="linear16",
                sample_rate=8000,
                channels=1,
                punctuate=True,
                interim_results=True,
                endpointing=300,
            )
        )
        
        # OpenAI LLM (via copilot-api proxy)
        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL,
        )
        
        # Conversation context
        messages = [
            {"role": "system", "content": agent["prompt"]},
        ]
        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        
        # Build pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
                allow_interruptions=True,
            )
        )
        
        # Start with agent greeting
        @transport.event_handler("on_client_connected")
        async def on_connected(transport, client):
            logger.in(f"Client connected for {call_id}")
            messages.append({
                "role": "system", 
                "content": "Start the conversation by introducing yourself briefly."
            })
            await task.queue_frames([LLMRunFrame()])
        
        @transport.event_handler("on_client_disconnected")
        async def on_disconnected(transport, client):
            logger.info(f"Client disconnected for {call_id}")
            await task.cancel()
        
        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)
        
        logger.info(f"Call {call_id} pipeline completed")
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for stream start: {call_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for {call_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if call_id in active_calls:
            active_calls[call_id]["status"] = "completed"
        logger.info(f"WebSocket closed for {call_id}")


@app.get("/calls")
async def list_calls():
    """List all active/recent calls"""
    return {
        "calls": [
            {
                "call_id": cid,
                "to_number": info["to_number"],
                "status": info["status"],
                "agent": info["agent"]["name"],
                "answered_by": info.get("answered_by", "unknown")
            }
            for cid, info in active_calls.items()
        ]
    }


@app.get("/calls/{call_id}")
async def get_call(call_id: str):
    """Get call details"""
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
    
    info = active_calls[call_id]
    return {
        "call_id": call_id,
        "to_number": info["to_number"],
        "status": info["status"],
        "agent": info["agent"]["name"],
        "call_sid": info.get("call_sid"),
        "answered_by": info.get("answered_by", "unknown"),
        "is_voicemail": info.get("is_voicemail", False)
    }


@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {"id": aid, "name": a["name"], "description": a["description"]}
            for aid, a in AGENTS.items()
        ]
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "twilio_configured": twilio_client is not None,
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
