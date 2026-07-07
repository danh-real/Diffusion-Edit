import time
import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
# import cv2
import torch
import os
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate

class TimingCallback(L.Callback):
    def __init__(self, print_every_n_steps: int = 10):
        self.print_every_n_steps = print_every_n_steps
        self._step_start: float = 0.0
        self._batch_end: float | None = None
        self._total_steps: int = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.time()
        self._step_start = now
        # first step has no previous batch_end, so skip data-load timing
        if self._batch_end is not None:
            self._data_load_time = now - self._batch_end

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        self._step_time = now - self._step_start
        self._batch_end = now
        self._total_steps += 1

        if self._total_steps % self.print_every_n_steps == 0:
            data_load_str = (
                f"{self._data_load_time * 1000:.1f} ms"
                if hasattr(self, "_data_load_time")
                else "N/A"
            )
            # print(
            #     f"[Timing] Step {self._total_steps}: "
            #     f"step={self._step_time * 1000:.1f} ms, "
            #     f"data_load={data_load_str}"
            # )


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None
        )

        self.total_steps = 0

        self._step_start: float = 0.0
        self._batch_end: float | None = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Getting time
        now = time.time()
        self._step_start = now
        # first step has no previous batch_end, so skip data-load timing
        if self._batch_end is not None:
            self._data_load_time = now - self._batch_end

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Getting time
        now = time.time()
        self._step_time = now - self._step_start
        self._batch_end = now

        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            data_load_str = (
                f"{self._data_load_time * 1000:.1f} ms"
                if hasattr(self, "_data_load_time")
                else "N/A"
            )
            print(
                f"Epoch: {trainer.current_epoch}, "
                f"Steps: {self.total_steps}, "
                f"Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, "
                f"Gradient size: {gradient_size:.4f}, "
                f"Max gradient size: {max_gradient_size:.4f}, "
                f"Training time: {self._step_time * 1000:.1f} ms, "
                f"Data time: {data_load_str}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type,
    ):

        file_name = [
            "assets/coffee.png",
            "assets/coffee.png",
            "assets/coffee.png",
            "assets/coffee.png",
            "assets/clock.jpg",
            "assets/book.jpg",
            "assets/monalisa.jpg",
            "assets/oranges.jpg",
            "assets/penguin.jpg",
            "assets/vase.jpg",
            "assets/room_corner.jpg",
        ]

        test_instruction = [
            "Make the image look like it's from an ancient Egyptian mural.",
            'get rid of the coffee bean.',
            'remove the cup.',
            "Change it to look like it's in the style of an impasto painting.",
            "Make this photo look like a comic book",
            "Give this the look of a traditional Japanese woodblock print.",
            'delete the woman',
            "Change the image into a watercolor painting.",
            "Make it black and white.",
            "Make it pop art.",
            'the sofa is leather, and the wall is black',
        ]

        pl_module.flux_kontext_pipe.transformer.eval()
        for i, name in enumerate(file_name):
            test_image = Image.open(name)

            if pl_module.use_sequence_conditioning:
                # Kontext conditions on the reference image directly (no mask/diptych) —
                # mirrors infer.py's inference call and the training data in data.py.
                image = pl_module.flux_kontext_pipe(
                    prompt=test_instruction[i],
                    image=test_image,
                    height=test_image.size[1],
                    width=test_image.size[0],
                    guidance_scale=2.5,
                    num_inference_steps=28,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]
                image.save(os.path.join(save_path, f'flux-kontext-test-{self.total_steps}-{i}-{condition_type}.jpg'))
            else:
                combined_image = Image.new('RGB', (test_image.size[0] * 2, test_image.size[1]))
                combined_image.paste(test_image, (0, 0))
                combined_image.paste(test_image, (test_image.size[0], 0))

                mask = Image.new('L', combined_image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([test_image.size[0], 0, test_image.size[0] * 2, test_image.size[1]], fill=255)
                if condition_type == 'edit_n':
                    prompt_ = "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left. \n " + test_instruction[i]
                else:
                    prompt_ = "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but " + test_instruction[i]

                image = pl_module.flux_kontext_pipe(
                    prompt=prompt_,
                    image=combined_image,
                    height=512,
                    width=1024,
                    mask_image=mask,
                    guidance_scale=50,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]
                image.save(os.path.join(save_path, f'flux-fill-test-{self.total_steps}-{i}-{condition_type}.jpg'))

        pl_module.flux_kontext_pipe.transformer.train()
