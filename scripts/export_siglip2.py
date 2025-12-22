from pathlib import Path

import torch
from transformers import AutoModel

model_ckpt = "google/siglip2-base-patch32-256"
output_file = Path(f"{Path(__file__).parent.parent}/models/siglip2/siglip2_base_patch32_256.onnx")

print(f"Loading {model_ckpt}...")
# Load the full model
full_model = AutoModel.from_pretrained(model_ckpt)

# Load visual model
vision_model = full_model.vision_model
vision_model.eval()

img_size = full_model.config.vision_config.image_size

dummy_input = torch.randn(1, 3, img_size, img_size)
output_file.parent.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(vision_model, dummy_input, output_file)

print(f"Exported to {output_file}...")