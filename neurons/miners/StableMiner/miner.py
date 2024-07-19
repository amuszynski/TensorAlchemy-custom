import torch
from torch.quantization import quantize_dynamic
from base import BaseMiner
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from utils import colored_log, warm_up
from functools import lru_cache
import bittensor as bt  # Dodane, aby użyć bt.Tensor

class StableMiner(BaseMiner):
    def __init__(self):
        super().__init__()
        self.request_count = 0
        self.memory_clear_threshold = 10  # Czyść pamięć co 10 zapytań

        # Load the model
        self.load_models()

        # Optimize model
        self.optimize_models()

        # Serve the axon
        self.start_axon()

        # Start the miner loop
        self.loop()

    def clear_gpu_memory(self):
        torch.cuda.empty_cache()

    def load_models(self):
        # Load the text-to-image model
        self.t2i_model = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)

        self.t2i_model = quantize_dynamic(
            self.t2i_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        
        self.t2i_model.set_progress_bar_config(disable=True)
        self.t2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.t2i_model.scheduler.config
        )

        # Load the image to image model using the same pipeline (efficient)
        self.i2i_model = AutoPipelineForImage2Image.from_pipe(self.t2i_model).to(
            self.config.miner.device,
        )
        self.i2i_model.set_progress_bar_config(disable=True)
        self.i2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.i2i_model.scheduler.config
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        self.processor = CLIPImageProcessor()

        # Set up mapping for the different synapse types
        self.mapping = {
            "text_to_image": {"args": self.t2i_args, "model": self.t2i_model},
            "image_to_image": {"args": self.i2i_args, "model": self.i2i_model},
        }
        
    @lru_cache(maxsize=100)
    def cached_generate_image(self, prompt, width, height, seed):
        generator = torch.Generator(device=self.config.miner.device).manual_seed(seed)
        return self.t2i_model(
            prompt=prompt,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        
    def optimize_models(self):
        if self.config.miner.optimize:
            self.t2i_model.unet = torch.compile(
                self.t2i_model.unet, mode="reduce-overhead", fullgraph=True
            )

            colored_log(
                ">>> Warming up model with compile... "
                + "this takes roughly two minutes...",
                color="yellow",
            )
            warm_up(self.t2i_model, self.t2i_args)

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        try:
            if synapse.generation_type == "text_to_image":
                seed = synapse.seed if synapse.seed != -1 else self.config.miner.seed
                image = self.cached_generate_image(
                    synapse.prompt, synapse.width, synapse.height, seed
                )
                synapse.images = [bt.Tensor.serialize(self.transform(image))]
            else:
                # Istniejąca logika dla image_to_image
                model = self.mapping[synapse.generation_type]["model"]
                args = self.mapping[synapse.generation_type]["args"].copy()
                args.update({
                    "prompt": synapse.prompt,
                    "image": synapse.image,
                    "strength": synapse.strength,
                    "guidance_scale": synapse.guidance_scale,
                })
                images = model(**args).images
                synapse.images = [bt.Tensor.serialize(self.transform(image)) for image in images]
        finally:
            self.request_count += 1
            if self.request_count >= self.memory_clear_threshold:
                self.clear_gpu_memory()
                self.request_count = 0
        
        return synapse
