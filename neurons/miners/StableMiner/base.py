import asyncio
from functools import lru_cache
import bittensor as bt
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
from abc import ABC, abstractmethod
import random
import argparse
import os
import torch
from torchvision import transforms
import logging
from typing import Dict, Tuple
from bittensor.utils.tokenizer_utils import get_coldkey_for_hotkey

logger = logging.getLogger(__name__)

VPERMIT_TAO = 1.0  # Stała do użycia w _base_blacklist

class BackgroundTimer:
    def __init__(self, interval, function, args=None):
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.timer = None

    def start(self):
        self.timer = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            await asyncio.sleep(self.interval)
            await self.function(*self.args)

    def stop(self):
        if self.timer:
            self.timer.cancel()

class BaseMiner(ABC):
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_initial_state()
        self.setup_wallet_and_subtensor()
        self.setup_metagraph()
        self.setup_axon()
        self.setup_background_tasks()

    def setup_logging(self):
        if self.config.logging.debug:
            bt.debug()
            logger.info("Enabling debug mode...")
        
        self.colored_log("Outputting miner config:", color="green")
        self.colored_log(f"{self.config}", color="green")

    def setup_initial_state(self):
        self.wandb = None
        self.t2i_args, self.i2i_args = self.get_args()
        self.hotkey_blacklist = set()
        self.coldkey_blacklist = set()
        self.coldkey_whitelist = set(["5F1FFTkJYyceVGE4DCVN5SxfEQQGJNJQ9CVFVZ3KpihXLxYo"])
        self.hotkey_whitelist = set(["5C5PXHeYLV5fAx31HkosfCkv8ark3QjbABbjEusiD3HXH2Ta"])
        self.storage_client = None
        self.event = {}
        self.stats = self.get_defaults()
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.request_dict = {}

    def setup_wallet_and_subtensor(self):
        logger.info("Establishing subtensor connection")
        self.subtensor = bt.subtensor(config=self.config)
        self.wallet = bt.wallet(config=self.config)

    def setup_metagraph(self):
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.loop_until_registered()

    def setup_axon(self):
        self.colored_log(f"Serving axon on port {self.config.axon.port}.", color="green")
        self.axon = bt.axon(
            wallet=self.wallet,
            ip=bt.utils.networking.get_external_ip(),
            external_ip=self.config.axon.get("external_ip") or bt.utils.networking.get_external_ip(),
            config=self.config,
        )
        self.axon.attach(
            forward_fn=self.is_alive,
            blacklist_fn=self.blacklist_is_alive,
            priority_fn=self.priority_is_alive,
        ).attach(
            forward_fn=self.generate_image,
            blacklist_fn=self.blacklist_image_generation,
            priority_fn=self.priority_image_generation,
        ).start()
        
        self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)
        self.colored_log(f"Axon created: {self.axon}", color="green")

    def setup_background_tasks(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.background_timer = BackgroundTimer(300, self.background_loop)
        self.background_timer.start()

    async def background_loop(self):
        self.executor.submit(self.update_metagraph)
        self.executor.submit(self.log_stats)

    def update_metagraph(self):
        self.metagraph.sync(subtensor=self.subtensor)

    def get_config(self) -> bt.config:
        argp = argparse.ArgumentParser(description="Miner Configs")
        self.add_args(argp)
        argp.add_argument("--netuid", type=int, default=1)
        argp.add_argument("--wandb.project", type=str, default="")
        argp.add_argument("--wandb.entity", type=str, default="")
        argp.add_argument("--wandb.api_key", type=str, default="")
        argp.add_argument("--miner.device", type=str, default="cuda:0")
        argp.add_argument("--miner.optimize", action="store_true")
        seed = random.randint(0, 100_000_000_000)
        argp.add_argument("--miner.seed", type=int, default=seed)
        argp.add_argument("--miner.model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
        bt.subtensor.add_args(argp)
        bt.logging.add_args(argp)
        bt.wallet.add_args(argp)
        bt.axon.add_args(argp)
        config = bt.config(argp)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "miner",
            )
        )
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path, exist_ok=True)
        return config

    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser):
        pass

    def get_args(self):
        return {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
        }, {"guidance_scale": 5, "strength": 0.6}

    def loop_until_registered(self):
        while True:
            index = self.get_miner_index()
            if index is not None:
                self.miner_index = index
                logger.info(f"Miner {self.config.wallet.hotkey} is registered with uid {self.metagraph.uids[self.miner_index]}")
                break
            logger.warning(f"Miner {self.config.wallet.hotkey} is not registered. Sleeping for 120 seconds...")
            time.sleep(120)
            self.metagraph.sync(subtensor=self.subtensor)

    def get_miner_index(self):
        try:
            return self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            return None

    @lru_cache(maxsize=1000)
    def get_caller_stake(self, hotkey: str) -> float:
        try:
            return float(self.metagraph.S[self.metagraph.hotkeys.index(hotkey)])
        except ValueError:
            return 0.0

    async def generate_image(self, synapse: bt.ImageGeneration) -> bt.ImageGeneration:
        start_time = time.perf_counter()
        self.stats.total_requests += 1

        try:
            local_args = self.prepare_generation_args(synapse)
            model = self.get_model(synapse.generation_type)

            for attempt in range(3):
                try:
                    images = await self.execute_image_generation(model, local_args, synapse)
                    synapse.images = [bt.Tensor.serialize(self.transform(image)) for image in images]
                    self.colored_log(f"Generating -> Successful image generation after {attempt+1} attempt(s).", color="cyan")
                    break
                except Exception as e:
                    logger.error(f"Error in attempt number {attempt+1} to generate an image: {e}... sleeping for 5 seconds...")
                    await asyncio.sleep(5)
                    if attempt == 2:
                        synapse.images = []
                        logger.info(f"Failed to generate any images after {attempt+1} attempts.")

            if self.is_nsfw(images):
                logger.info("An image was flagged as NSFW: discarding image.")
                self.stats.nsfw_count += 1
                synapse.images = []

            await self.log_to_wandb(synapse)

        except Exception as e:
            logger.error(f"Unexpected error in generate_image: {traceback.format_exc()}")
            synapse.images = []

        generation_time = time.perf_counter() - start_time
        self.update_stats(generation_time)

        return synapse

    def prepare_generation_args(self, synapse):
        local_args = self.t2i_args.copy() if synapse.generation_type == "text_to_image" else self.i2i_args.copy()
        local_args["prompt"] = [self.clean_nsfw_from_prompt(synapse.prompt)]
        local_args["width"] = synapse.width
        local_args["height"] = synapse.height
        local_args["num_images_per_prompt"] = synapse.num_images_per_prompt
        
        try:
            local_args["guidance_scale"] = synapse.guidance_scale
            if synapse.negative_prompt:
                local_args["negative_prompt"] = [synapse.negative_prompt]
        except AttributeError:
            logger.info("Values for guidance_scale or negative_prompt were not provided.")

        try:
            local_args["num_inference_steps"] = synapse.steps
        except AttributeError:
            logger.info("Values for steps were not provided.")

        return local_args

    @abstractmethod
    def get_model(self, generation_type: str):
        pass

    async def execute_image_generation(self, model, local_args, synapse):
        seed = synapse.seed if synapse.seed != -1 else self.config.miner.seed
        local_args["generator"] = [torch.Generator(device=self.config.miner.device).manual_seed(seed)]
        
        if synapse.generation_type == "image_to_image":
            local_args["image"] = transforms.ToPILImage()(bt.Tensor.deserialize(synapse.prompt_image))

        return await asyncio.to_thread(model, **local_args)

    @abstractmethod
    def is_nsfw(self, images):
        pass

    @abstractmethod
    def clean_nsfw_from_prompt(self, prompt: str) -> str:
        pass

    async def log_to_wandb(self, synapse):
        if self.wandb:
            try:
                await asyncio.to_thread(self.wandb._add_images, synapse)
                await asyncio.to_thread(self.wandb._log)
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}")

    def update_stats(self, generation_time):
        self.stats.generation_time += generation_time
        avg_time = self.stats.generation_time / self.stats.total_requests
        self.colored_log(f"Time -> {generation_time:.2f}s | Average: {avg_time:.2f}s", color="yellow")

    def _base_blacklist(self, synapse, vpermit_tao_limit=VPERMIT_TAO, rate_limit=1) -> Tuple[bool, str]:
        caller_hotkey = synapse.dendrite.hotkey
        caller_coldkey = get_coldkey_for_hotkey(self, caller_hotkey)
        caller_stake = self.get_caller_stake(caller_hotkey)

        if caller_coldkey in self.coldkey_whitelist or caller_hotkey in self.hotkey_whitelist:
            return False, "Whitelisted key recognized."

        if self.is_rate_limited(caller_hotkey, rate_limit):
            return True, f"Rate limit ({rate_limit:.2f}) exceeded."

        if caller_stake is None:
            return True, "Non-registered hotkey."

        if caller_stake < vpermit_tao_limit:
            return True, f"Insufficient stake: {caller_stake:.2f} < {vpermit_tao_limit}"

        return False, "Hotkey recognized"

    def is_rate_limited(self, caller_hotkey, rate_limit):
        now = time.perf_counter()
        if caller_hotkey in self.request_dict:
            last_request = self.request_dict[caller_hotkey]["history"][-1]
            if now - last_request < rate_limit:
                self.request_dict[caller_hotkey]["rate_limited_count"] += 1
                return True
        self.request_dict.setdefault(caller_hotkey, {"history": [], "count": 0, "rate_limited_count": 0})
        self.request_dict[caller_hotkey]["history"].append(now)
        self.request_dict[caller_hotkey]["count"] += 1
        return False

    def log_stats(self):
        log = (
            f"Block: {self.metagraph.block.item()} | "
            f"Stake: {self.metagraph.S[self.miner_index]:.2f} | "
            f"Rank: {self.metagraph.R[self.miner_index]:.2f} | "
            f"Trust: {self.metagraph.T[self.miner_index]:.2f} | "
            f"Consensus: {self.metagraph.C[self.miner_index]:.2f} | "
            f"Incentive: {self.metagraph.I[self.miner_index]:.2f} | "
            f"Emission: {self.metagraph.E[self.miner_index]:.2f}"
        )
        self.colored_log(log, color="green")

        top_requestors = sorted(
            [(k, v["count"], sum(v["history"]) / len(v["history"]) if v["history"] else 0, v["rate_limited_count"])
             for k, v in self.request_dict.items()],
            key=lambda x: x[1], reverse=True
        )[:10]

        total_requests = sum(r[1] for r
