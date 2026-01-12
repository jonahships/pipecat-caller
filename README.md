# Pipecat Twilio Caller

Self-hosted phone calling system using:
- **Twilio** - Your own phone number and account for telephony
- **ElevenLabs** - Voice synthesis (using your API key)
- **OpenAI** - LLM for conversation intelligence
- **Pipecat** - Real-time voice pipeline

## Features

- 🎯 **Custom prompts per call** - Full control over what the agent says
- 🔊 **Same ElevenLabs voices** - Larry, Liam, Jasmine
- 📞 **Your Twilio number** - Complete ownership
- 💬 **Real-time conversation** - Low latency voice interaction
- 🎭 **Multiple agents** - Larry, Liam Sales, Jasmine Support, etc.

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

2. Get your credentials:
   - Twilio: Account SID, Auth Token from console.twilio.com
   - ElevenLabs: API key from elevenlabs.io
   - OpenAI: API key from platform.openai.com

3. Set up ngrok or similar for the public webhook URL:
```bash
ngrok http 8765
# Update PUBLIC_URL in .env with the ngrok URL
```

4. Run the server:
```bash
./venv/bin/python server.py
```

## API Usage

### Initiate a call
```bash
curl -X POST http://localhost:8765/call \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15551234567",
    "agent_id": "larry",
    "custom_prompt": "Ask them what they are doing this weekend and have a friendly chat"
  }'
```

### List available agents
```bash
curl http://localhost:8765/agents
```

### Check call status
```bash
curl http://localhost:8765/calls/{call_id}
```

### Health check
```bash
curl http://localhost:8765/health
```

## Agents

| ID | Name | Description |
|----|------|-------------|
| larry | Larry | Personal assistant to Master Jonah |
| liam_sales | Liam | Sales agent |
| jasmine_support | Jasmine | Customer support |
| jasmine_appointments | Jasmine | Appointment scheduling |
| receptionist | Jasmine | Inbound call handling |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Twilio    │────▶│  Pipecat     │────▶│  ElevenLabs │
│  (Phone)    │◀────│   Server     │◀────│    (TTS)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   OpenAI     │
                    │   (LLM)      │
                    └──────────────┘
```

1. Twilio receives/makes the call
2. Twilio connects a media stream (WebSocket) to our server
3. Pipecat pipeline processes audio in real-time:
   - User speech → OpenAI → ElevenLabs → User hears response
4. Full control over prompts and behavior
