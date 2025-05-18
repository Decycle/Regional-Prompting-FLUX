import torch
from pipeline_flux_regional import RegionalFluxPipeline, RegionalFluxAttnProcessor2_0
from pipeline_flux_controlnet_regional import RegionalFluxControlNetPipeline
from diffusers import FluxControlNetModel, FluxMultiControlNetModel
from typing import Annotated
from pydantic import BaseModel, Field, field_validator, model_validator, PositiveInt, PositiveFloat
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from functools import wraps
import uuid
import json
from tqdm import trange

class Lora(BaseModel):
    name: str = Field(..., description="The name of the LoRA model.")
    path: str = Field(..., description="The path to the LoRA model file.")

class PromptMaskPair(BaseModel):
    prompt: str = Field(..., description="Description of the region.")
    mask: list[int] = Field(..., description="Coordinates of the mask in the format [x1, y1, x2, y2].")
    adapter: list[str] = Field([], description="List of LoRA adapters to be used for this region.")
    weight: list[float] = Field([], description="List of weights for the LoRA adapters.")

    @field_validator('mask')
    def check_mask_coordinates(mask):
        if len(mask) != 4:
            raise ValueError('Mask must be a list of four integers [x1, y1, x2, y2].')
        x1, y1, x2, y2 = mask
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid mask coordinates.')
        return mask

    @field_validator('weight')
    def check_weight(weight):
        for w in weight:
            assert w > 0, "Weight must be a positive number."
        return weight

    @model_validator(mode='after')
    def check_adapter_weight(self):
        if len(self.adapter) != len(self.weight):
            raise ValueError('Adapter and weight lists must have the same length.')
        return self

class ImageGenerationRequest(BaseModel):
    # loras: list[Lora] = Field(..., description="List of LoRA models to be used.")
    image_width: PositiveInt = Field(1024, description="Width of the generated image.")
    image_height: PositiveInt = Field(768, description="Height of the generated image.")
    num_samples: PositiveInt = Field(1, description="Number of images to generate.")
    num_inference_steps: PositiveInt = Field(24, description="Number of inference steps.")
    guidance_scale: PositiveFloat = Field(3.5, description="Guidance scale for generation.")
    seed: int = Field(-1, description="Random seed for generation.")

    mask_inject_steps: PositiveInt = Field(16, description="Number of steps for mask injection.")
    double_inject_blocks_interval: PositiveInt = Field(1, description="Interval for double injection blocks.")
    single_inject_blocks_interval: PositiveInt = Field(1, description="Interval for single injection blocks.")
    base_ratio: PositiveFloat = Field(0.1, description="Base ratio for blending.")

    base_prompt: str = Field(..., description="Base prompt for image generation.")

    base_adapter: list[str] = Field([], description="List of base adapters to be used.")
    base_adapter_weight: list[float] = Field([], description="List of weights for the base adapters.")

    normal_adapter: list[str] = Field([], description="List of normal adapters to be used.")
    normal_adapter_weight: list[float] = Field([], description="List of weights for the normal adapters.")

    regional_prompt_mask_pairs: list[PromptMaskPair] = Field([], description="Dictionary of regional prompt and mask pairs.")

    @classmethod
    @field_validator('image_width', 'image_height')
    def check_divisible_by_8(cls, v):
        if v % 8 != 0:
            raise ValueError('Width and height must be divisible by 8.')
        return v

    @model_validator(mode='after')
    def check_loras(self):
        assert len(self.base_adapter) == len(self.base_adapter_weight), "Base adapter and weight lists must have the same length."
        assert len(self.normal_adapter) == len(self.normal_adapter_weight), "Normal adapter and weight lists must have the same length."
        assert self.mask_inject_steps <= self.num_inference_steps, "Mask inject steps must be less than or equal to the number of inference steps."
        return self

model_path = "black-forest-labs/FLUX.1-dev"

pipeline = RegionalFluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")


emperor_lora = Lora(name="emperor", path="/workspace/emperor.safetensors")
chinese_maid_lora = Lora(name="chinese-maid", path="/workspace/chinese-maid.safetensors")
ami_lora = Lora(name="ami", path="/workspace/ami.safetensors")

for lora in [emperor_lora, chinese_maid_lora, ami_lora]:
    pipeline.load_lora_weights(lora.path, adapter_name=lora.name)

attn_procs = {}
for name in pipeline.transformer.attn_processors.keys():
    if 'transformer_blocks' in name and name.endswith("attn.processor"):
        attn_procs[name] = RegionalFluxAttnProcessor2_0()
    else:
        attn_procs[name] = pipeline.transformer.attn_processors[name]
pipeline.transformer.set_attn_processor(attn_procs)

def generate_images(img_generation_request: ImageGenerationRequest):
    ## generation settings

    # example regional prompt and mask pairs
    image_width = img_generation_request.image_width
    image_height = img_generation_request.image_height
    num_samples = img_generation_request.num_samples
    num_inference_steps = img_generation_request.num_inference_steps
    guidance_scale = img_generation_request.guidance_scale
    seed = img_generation_request.seed

    base_prompt = img_generation_request.base_prompt
    base_adapter = img_generation_request.base_adapter
    base_adapter_weight = img_generation_request.base_adapter_weight

    normal_adapter = img_generation_request.normal_adapter
    normal_adapter_weight = img_generation_request.normal_adapter_weight
    # background_prompt = img_generation_request.base_prompt

    regional_prompt_mask_pairs = img_generation_request.regional_prompt_mask_pairs
    mask_inject_steps = img_generation_request.mask_inject_steps
    double_inject_blocks_interval = img_generation_request.double_inject_blocks_interval
    single_inject_blocks_interval = img_generation_request.single_inject_blocks_interval

    base_ratio = img_generation_request.base_ratio

    image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
    image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor

    regional_prompts = []
    regional_masks = []
    regional_adapters = []
    regional_adapter_weights = []
    # background_mask = torch.ones((image_height, image_width))

    for region in regional_prompt_mask_pairs:
        prompt = region.prompt
        mask = region.mask
        adapter = region.adapter
        weight = region.weight
        x1, y1, x2, y2 = mask

        mask = torch.zeros((image_height, image_width))
        mask[y1:y2, x1:x2] = 1.0

        # background_mask -= mask

        regional_prompts.append(prompt)
        regional_masks.append(mask)
        regional_adapters.append(adapter)
        regional_adapter_weights.append(weight)


    # if regional masks don't cover the whole image, append background prompt and mask
    # if background_mask.sum() > 0:
    #     regional_prompts.append(background_prompt)
    #     regional_masks.append(background_mask)

    # setup regional kwargs that pass to the pipeline
    joint_attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'regional_adapters': regional_adapters,
        'regional_adapter_weights': regional_adapter_weights,
        'double_inject_blocks_interval': double_inject_blocks_interval,
        'single_inject_blocks_interval': single_inject_blocks_interval,
        'base_ratio': base_ratio,
        'base_adapter': base_adapter,
        'base_adapter_weight': base_adapter_weight,
        'normal_adapter': normal_adapter,
        'normal_adapter_weight': normal_adapter_weight,
    }


    images = pipeline(
        prompt=base_prompt,
        num_samples=num_samples,
        width=image_width, height=image_height,
        mask_inject_steps=mask_inject_steps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        joint_attention_kwargs=joint_attention_kwargs,
    ).images

    return images

base_prompt = "a picture of a chinese emperor and a maid, facing the camera. The light shines on their faces, highlighting their features. The emperor is wearing a traditional Chinese robe with intricate patterns and a golden crown. The maid is wearing a traditional Chinese dress with intricate patterns and a golden crown. The Chinese emperor cnstll is standing on the left while the maid trtcrd is standing on the right. The background features a beautiful cliff with lush greenery."
# base_prompt = "a picture of a mixed-asian girl and a maid, facing the camera. The light shines on their faces, highlighting their features. The mixed-aisan girl p3rs0n is standing on the left while the maid trtcrd is standing on the right. The maid is wearing a traditional Chinese dress with intricate patterns, and girl is wearing a black dress."
base_adapter = ["emperor", "chinese-maid"]
base_adapter_weight = [0.05, 0.05]

normal_adapter = ["emperor", "chinese-maid"]
normal_adapter_weight = [0.2, 0.2]

regional_prompt_mask_pairs = [
    PromptMaskPair(
        prompt="a picture of cnstll, a Chinese emperor, facing the camera, the light shines on his face, highlighting his features. He is wearing a traditional Chinese robe with intricate patterns and a golden crown",
        mask=[0, 0, 512, 768],
        adapter=["emperor"],
        weight=[1.2]
    ),
    PromptMaskPair(
        prompt="a picture of trtcrd, a Chinese maid, facing the camera, the light shines on her face, highlighting her features. She is wearing a traditional red Chinese dress with intricate patterns",
        mask=[512, 0, 1024, 768],
        adapter=["chinese-maid"],
        weight=[1.2]
    )
]

# seed = torch.randint(0, 2**32 - 1, (1,)).item()
# with open("seed.txt", "w") as f:
#     f.write(str(seed))
# print(f"seed: {seed}")

img_generation_request = ImageGenerationRequest(
    # loras=[emperor_lora, chinese_maid_lora],
    image_width=1024,
    image_height=768,
    num_samples=1,
    num_inference_steps=24,
    mask_inject_steps=24,
    guidance_scale=2,
    seed=-1,
    double_inject_blocks_interval=1,
    single_inject_blocks_interval=2,
    base_ratio=0.05,
    base_prompt=base_prompt,
    base_adapter=base_adapter,
    base_adapter_weight=base_adapter_weight,
    normal_adapter=normal_adapter,
    normal_adapter_weight=normal_adapter_weight,
    regional_prompt_mask_pairs=regional_prompt_mask_pairs
)

# generate image
for i in trange(300):
    seed = torch.randint(0, 2**32 - 1, (1,)).item()
    img_generation_request.seed = seed
    image = generate_images(img_generation_request)[0]
    image.save(f"output_benchmark/regional_{i}.png")


def lazy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def roll():
            return func(*args, **kwargs)
        return roll
    return wrapper

@lazy
def randint(start, end):
    """Generate a random integer between start and end."""
    return np.random.randint(start, end+1)

@lazy
def uniform(start, end):
    """Generate a random float between start and end."""
    return round(np.random.uniform(start, end), 2)

# keep generating random images
# test_args = {
#     "num_inference_steps": randint(20, 50),
#     "mask_inject_steps_ratio": uniform(0.2, 1.0),
#     "guidance_scale": uniform(1, 5),
#     "double_inject_blocks_interval": randint(1, 3),
#     "single_inject_blocks_interval": randint(1, 3),
#     "base_ratio": uniform(0, 0.1),
#     "base_adapter_weight": uniform(0, 0.2),
#     "normal_adapter_weight": uniform(0, 0.2),
#     "lora_weight": uniform(1, 2),
# }

# total_iters = 1000
# for i in trange(total_iters):
#     # generate random values for the test args
#     random_args = {k: v() for k, v in test_args.items()}
#     random_args["seed"] = np.random.randint(0, 2**32 - 1)
#     img_generation_request_copy = img_generation_request.model_copy(deep=True)

#     # update the image generation request with the random values
#     img_generation_request_copy.num_inference_steps = random_args["num_inference_steps"]
#     img_generation_request_copy.mask_inject_steps = int(random_args["mask_inject_steps_ratio"] * random_args["num_inference_steps"])
#     img_generation_request_copy.guidance_scale = random_args["guidance_scale"]
#     img_generation_request_copy.double_inject_blocks_interval = random_args["double_inject_blocks_interval"]
#     img_generation_request_copy.single_inject_blocks_interval = random_args["single_inject_blocks_interval"]
#     img_generation_request_copy.base_ratio = random_args["base_ratio"]
#     img_generation_request_copy.base_adapter_weight = random_args["base_adapter_weight"]
#     img_generation_request_copy.normal_adapter_weight = random_args["normal_adapter_weight"]
#     for i in range(len(img_generation_request_copy.regional_prompt_mask_pairs)):
#         regional_prompt_mask_pair = img_generation_request_copy.regional_prompt_mask_pairs[i]
#         for j in range(len(regional_prompt_mask_pair.weight)):
#             regional_prompt_mask_pair.weight[j] = random_args["lora_weight"]
#     img_generation_request_copy.seed = random_args["seed"]

#     # print("COPY", img_generation_request_copy)
#     # generate the image
#     image = generate_images(img_generation_request_copy)[0]
#     # save the image
#     uuid_str = str(uuid.uuid4())
#     image.save(f"outputs/random_image_{uuid_str}.png")

#     with open("output_config.jsonl", 'a') as f:
#         # data = {
#         #     uuid_str: random_args
#         # }
#         random_args["uuid"] = uuid_str
#         json.dump(random_args, f)
#         f.write('\n')






# want to see base ratio vs

# images = []



# for weight in torch.arange(1.0, 3.0, 0.3):
#     weight = round(weight.item(), 2)
#     for i in range(len(img_generation_request.regional_prompt_mask_pairs)):
#         regional_prompt_mask_pair = img_generation_request.regional_prompt_mask_pairs[i]
#         for j in range(len(regional_prompt_mask_pair.weight)):
#             regional_prompt_mask_pair.weight[j] = weight

#     print(f"lora weight: {weight}")
#     image = generate_images(img_generation_request)[0]
#     images.append({
#         "lora": weight,
#         "image": image
#     })
#     image.save(f"lora_weight_{weight}.png")

# combine the images into a grid

# def plot_images(images, base_figsize=(8,8)):
#     now = datetime.now()
#     formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S")

#     keys = list(images[0].keys())
#     keys.remove("image")
#     if len(keys) == 0:
#         plt.imshow(images[0]["image"])
#         plt.axis('off')
#         plt.savefig(f"grid_{formatted_date}.png", bbox_inches='tight', pad_inches=0)
#     elif len(keys) == 1:
#         num_images = len(images)
#         fig, axes = plt.subplots(1, num_images, figsize=(num_images * base_figsize[0], base_figsize[1]))
#         for i, img in enumerate(images):
#             axes[i].imshow(img["image"])
#             axes[i].set_title(f"{keys[0]}: {img[keys[0]]}")
#             axes[i].axis('off')
#         plt.tight_layout()
#         plt.savefig(f"grid_{formatted_date}.png", bbox_inches='tight', pad_inches=0)
#     elif len(keys) == 2:
#         raise NotImplementedError("2D grid not implemented yet.")
#     else:
#         raise NotImplementedError("More than 2D grid not implemented yet.")

# plot_images(images)