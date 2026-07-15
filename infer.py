from PIL import Image
from editscore import EditScore
from diffusers.utils import load_image

# Load the EditScore model. It will be downloaded automatically.
# Replace with the specific model version you want to use.
model_path = "ckpt/Qwen3-VL-8B-Instruct"
lora_path = "ckpt/EditScore-Qwen3-VL-8B-Instruct"

scorer = EditScore(
    backbone="qwen3vl", # set to "qwen3vl_vllm" for faster inference
    model_name_or_path=model_path,
    lora_path=lora_path,
    score_range=25,
    num_pass=1, # Increase for better performance via self-ensembling
)

# Below is Qwen2.5-VL version

# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# lora_path = "EditScore/EditScore-7B"

# scorer = EditScore(
#     backbone="qwen25vl", # set to "qwen25vl_vllm" for faster inference
#     model_name_or_path=model_path,
#     lora_path=lora_path,
#     score_range=25,
#     num_pass=1, # Increase for better performance via self-ensembling
# )

input_image = load_image("https://raw.githubusercontent.com/VectorSpaceLab/EditScore/main/example_images/input.png")
output_image = load_image("https://raw.githubusercontent.com/VectorSpaceLab/EditScore/main/example_images/output.png")
instruction = "Adjust the background to a glass wall."

result = scorer.evaluate([input_image, output_image], instruction)
print(f"Edit Score: {result['overall']}")
# Expected output: A dictionary containing the final score and other details.
