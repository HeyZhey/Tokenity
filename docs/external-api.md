# Tokenity External API

Tokenity exposes the loaded distributed model through an OpenAI-compatible
HTTP API on the coordinator Mac.

## Client Configuration

For the default Mango/Kiwi cluster:

- Provider type: Custom OpenAI / OpenAI Compatible
- Base URL: `http://192.168.5.23:8000/v1`
- Model: the identifier returned by `GET /v1/models`
- API key: not required; use `tokenity-local` if the client requires text

This configuration can be used by Cherry Studio, Msty, or any client that can
connect to a custom OpenAI endpoint.

## Endpoints

```text
GET  /health
GET  /v1/tokenity/info
GET  /v1/models
POST /v1/chat/completions
```

Both streaming and non-streaming chat completion requests are supported.
Tokenity enables CORS for local browser-based clients. The service currently
has no authentication layer, so it should only be exposed on a trusted local
network.

## Example

```bash
curl http://192.168.5.23:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-122B-A10B-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```
