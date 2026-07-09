# Use the modified diffusers & peft library
import sys
import os
import pandas as pd
from glob import glob
import json
from tqdm import tqdm
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./custom"))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
    
from diffusers import FluxKontextPipeline

# Below is the original library
import torch
from PIL import Image
import argparse
import random
    
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the output image")
parser.add_argument("--flux-path", type=str, default='ckpt/FLUX.1-Kontext-dev', help="Path to the model")
parser.add_argument("--lora-path", type=str, default=None, help="Path to the LoRA weights")
parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
parser.add_argument("--edit-file", type=str, help="edit.json file containing benchmark data")
parser.add_argument("--data-root", type=str, help="Root path to input data")
parser.add_argument("--saved-suffix-model", type=str, help="Suffix to model name in saving path")
parser.add_argument("--top-k", type=int, default=1, help="Override top_k for MoE routing (must be <= num_experts)")
parser.add_argument("--save-every", type=int, default=1, help="Write ablation CSV every N images; -1 to save only at the end")
parser.add_argument("--token-stride", type=int, default=1, help="Sample every N tokens when collecting ablation data")
parser.add_argument("--save-tensors", action="store_true",
                     help="Also save per-image route_weight/route_path/token_type as a .pth file "
                          "(one file per image, alongside the CSV)")
parser.add_argument("--save-token-inputs", action="store_true",
                     help="Also save the raw per-token MoE-layer input (the hidden state fed into "
                          "the router) in the .pth file. Large — D is the full hidden dim, not the "
                          "expert count. Implies --save-tensors.")

args = parser.parse_args()

# _CSV_PATH = f"/data/repos/models/ICEdit/ablation_{args.data_root.split('/')[-1]}_token_specific.csv"
# _GLOBAL_CSV_PATH = f"/data/repos/models/ICEdit/ablation_{args.data_root.split('/')[-1]}_global.csv"
# for _path in (_CSV_PATH, _GLOBAL_CSV_PATH):
#     if os.path.exists(_path):
#         raise FileExistsError(
#             f"{_path} already exists — refusing to append to data from a previous run. "
#             "Delete/rename it first, or pass a different --data-root."
#         )

pipe = FluxKontextPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
# pipe.load_lora_weights(args.lora_path)

# if args.top_k is not None:
#     for module in pipe.transformer.modules():
#         if getattr(module, 'moe_lora', False):
#             module.top_k = args.top_k

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload() 
else:
    pipe = pipe.to("cuda")

with open(args.edit_file, "r") as f:
    edits = json.load(f)
    
# with open("decomposed_instruction_2pass.json", "r") as f:
#     resize2remove = json.load(f)

# _CSV_COLS = [
#     "image_path", "edit_type", "timestep", "layer", "token_idx", "token_type",
#     "route_weight", "route_path", "n_txt_tokens", "n_img_tokens",
# ]
# _GLOBAL_CSV_COLS = [
#     "image_path", "edit_type", "timestep", "layer", "n_tokens",
#     "mean_route_weight", "expert_counts", "n_txt_tokens", "n_img_tokens",
# ]
# _csv_first_write = True
# _global_csv_first_write = True
# _pending_rows = []
# _pending_global_rows = []

for index, edit in enumerate(tqdm(edits)):
    
    # if edit["edit_type"] != "resize":
    #     continue

    image_path = os.path.join(args.data_root, edit["input_image"])
    instruction = edit["instruction"]
    # instruction = resize2remove[instruction]
    task = edit["edit_type"] if "edit_type" in edit.keys() else ""
    save_dir = os.path.join(args.output_dir, task, "FLUX.1-Kontext-dev" + args.saved_suffix_model, str(edit["id"]))
    
    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(image_path)
    image = image.convert("RGB")
    image.save(f"{save_dir}/input.jpg")

    if image.size[0] != 512:
        print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
        new_width = 512
        scale = new_width / image.size[0]
        new_height = int(image.size[1] * scale)
        new_height = (new_height // 8) * 8  
        image = image.resize((new_width, new_height))
        print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")

    print(f"Instruction: {instruction}")

    width, height = image.size

    pipe_output = pipe(
        prompt=instruction,
        image=image,
        height=height,
        width=width,
        num_inference_steps=28,
        generator=torch.Generator("cpu").manual_seed(args.seed) if args.seed is not None else None,
    )

    result_image = pipe_output.images[0]

    result_image.save(os.path.join(save_dir, f"output.jpg"))
    print(f"\033[92mResult saved as {os.path.join(save_dir, 'output.jpg')}\033[0m")

    with open(os.path.join(save_dir, "instruction.txt"), "w") as f:
        print(instruction, file=f)

    # # NOTE: ABLATION STUDY — flush to CSV every --save-every images
    # _pending_rows.extend((image_path, task, *data) for data in pipe_output.ablation_data)
    # _pending_global_rows.extend((image_path, task, *data) for data in pipe_output.ablation_global)
    # del pipe_output

    # is_last = (index == len(edits) - 1)
    # if is_last or (args.save_every > 0 and (index + 1) % args.save_every == 0):
    #     df = pd.DataFrame(_pending_rows, columns=_CSV_COLS)
    #     df.to_csv(_CSV_PATH, mode="a", header=_csv_first_write, index=False)
    #     _csv_first_write = False
    #     _pending_rows.clear()
    #     del df

    #     global_df = pd.DataFrame(_pending_global_rows, columns=_GLOBAL_CSV_COLS)
    #     global_df.to_csv(_GLOBAL_CSV_PATH, mode="a", header=_global_csv_first_write, index=False)
    #     _global_csv_first_write = False
    #     _pending_global_rows.clear()
    #     del global_df