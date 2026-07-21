"""
EditScore reward client, used by router-level GRPO (see rl_utils.py / router_grpo.py).

Wraps EditScore/EditScore-Qwen3-VL-8B-Instruct (https://github.com/VectorSpaceLab/EditScore)
via the `editscore` package's own `EditScore` class (same usage as infer.py), which already
reproduces the paper's VIEScore-style scoring protocol (arXiv:2509.23909, Section 4.1.1 and
Appendix M) internally:

  - Semantic Consistency (SC): given (input image, output image, instruction), scores
    editing success and degree-of-overediting, each in [0, score_range], collapsed to a
    single scalar via min() (the stricter, harder-to-game choice).
  - Perceptual Quality (PQ): given the output image alone, scores naturalness and
    absence-of-artifacts, each in [0, score_range], collapsed via min() the same way.
  - Final reward = sqrt(S_SC * S_PQ), the geometric mean the paper uses to fine-tune
    OmniGen2 via RL (Section 6.3) -- returned as `overall` by `EditScore.evaluate()`.

`num_pass` (`k_ensemble` here) reproduces the paper's Avg@K self-ensembling by averaging
independent stochastic judge passes -- also handled inside `EditScore.evaluate()`.
"""
from __future__ import annotations

import contextlib
from typing import List

import torch
from editscore import EditScore
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration


@contextlib.contextmanager
def _pin_qwen3vl_device(device: torch.device):
    """editscore's Qwen3VL backend hardcodes device_map="auto" internally (see
    editscore/mllm_tools/qwen3vl.py's Qwen3VL.__init__), which balances the model across
    every CUDA device with free memory at load time -- including whichever GPU the policy
    model is about to load onto, since "auto" has no notion of "reserved for later". Monkeypatch
    Qwen3VLForConditionalGeneration.from_pretrained for the duration of EditScore(...)
    construction to force one explicit device instead of "auto". Scoped narrowly (context
    manager, always restored) rather than patched permanently, since this is reaching into a
    third-party package's internals.
    """
    original = Qwen3VLForConditionalGeneration.from_pretrained.__func__

    def _pinned_from_pretrained(cls, *args, **kwargs):
        kwargs["device_map"] = {"": device}
        return original(cls, *args, **kwargs)

    Qwen3VLForConditionalGeneration.from_pretrained = classmethod(_pinned_from_pretrained)
    try:
        yield
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained = classmethod(original)


class EditScoreReward:
    """Frozen EditScore-8B judge, queried in-process. Not trainable -- always no_grad."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        lora_id: str = "EditScore/EditScore-Qwen3-VL-8B-Instruct",
        device: str = "cuda:1",
        dtype: torch.dtype = torch.bfloat16,
        k_ensemble: int = 1,
        temperature: float = 0.7,
        aggregation: str = "min",
        score_scale: float = 25.0,
    ):
        # NOTE: bfloat16 is still hardcoded inside editscore's own from_pretrained call
        # (matches this repo's default `dtype: "bfloat16"` everywhere, so left alone) --
        # `device` IS honored, via _pin_qwen3vl_device below.
        if aggregation != "min":
            raise ValueError(
                f"aggregation={aggregation!r} is not supported: the editscore package's "
                "evaluate() always collapses SC/PQ sub-scores via min()."
            )
        self.aggregation = aggregation
        self.device = torch.device(device)

        with _pin_qwen3vl_device(self.device):
            self.model = EditScore(
                backbone="qwen3vl", # set to "qwen3vl_vllm" for faster inference
                model_name_or_path=model_id,
                lora_path=lora_id,
                score_range=score_scale,
                temperature=temperature,
                num_pass=max(1, k_ensemble), # Avg@K self-ensembling
            )

    @torch.no_grad()
    def score(self, cond_img: Image.Image, out_img: Image.Image, instruction: str) -> float:
        """Sfinal = sqrt(S_SC * S_PQ) for one (input, output, instruction) triple,
        averaged over `k_ensemble` independent stochastic judge passes (Avg@K, paper eq. 1)."""
        # editscore's Qwen3VL.prepare_input hardcodes `inputs.to("cuda")` -- the ambient
        # "current device", not the model's actual device -- so the current device must match
        # self.device for the duration of this call, regardless of what the training loop (on a
        # different GPU) left it set to. torch.cuda.device() saves/restores automatically.
        device_ctx = (
            torch.cuda.device(self.device) if self.device.index is not None else contextlib.nullcontext()
        )
        with device_ctx:
            result = self.model.evaluate([cond_img, out_img], instruction)
        return float(result["overall"])

    def score_batch(
        self,
        cond_imgs: List[Image.Image],
        out_imgs: List[Image.Image],
        instructions: List[str],
    ) -> torch.Tensor:
        """Sequential scoring (one VLM call at a time) -- simplicity over throughput, per
        the in-process HF-transformers deployment choice. Returns a CPU float32 [N] tensor."""
        assert len(cond_imgs) == len(out_imgs) == len(instructions)
        scores = [
            self.score(c, o, p) for c, o, p in zip(cond_imgs, out_imgs, instructions)
        ]
        return torch.tensor(scores, dtype=torch.float32)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[0, 1]-range CHW tensor -> PIL.Image, matching data.py's T.ToTensor() dataset convention
    (batch["condition"] / batch["image"] are NOT the [-1, 1] range the VAE/image_processor use
    internally -- that remapping happens inside encode_images's image_processor.preprocess call)."""
    arr = (t.detach().float().clamp(0, 1) * 255.0).round().to(torch.uint8)
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)
