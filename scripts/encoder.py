import os
from pathlib import Path

import onnx
import torch
import clip
import tensorrt as trt

def onnx_to_trt(onnx_path):
    # https://github.com/easydiffusion/sdkit/blob/e94b1ffd0a914e5b1907898662c91f252aae260a/sdkit/utils/convert_model_utils.py
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    TIMING_CACHE = "timing.cache"

    TRT_BUILDER = trt.Builder(TRT_LOGGER)
    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    parse_success = onnx_parser.parse_from_file(onnx_path)

    if parse_success is False:
        raise RuntimeError(f"Unable to parse onnx : {onnx_path}")
    for idx in range(onnx_parser.num_errors):
        raise RuntimeError(f"ONNX model parsing failed : {onnx_parser.get_error(idx)}")

    trt_path = Path(str(onnx_path).replace('onnx', 'trt'))

    print(f"Making TRT engine")
    config = TRT_BUILDER.create_builder_config()
    profile = TRT_BUILDER.create_optimization_profile()

    if os.path.exists(TIMING_CACHE):
        with open(TIMING_CACHE, "rb") as f:
            timing_cache = config.create_timing_cache(f.read())
    else:
        timing_cache = config.create_timing_cache(b"")
    config.set_timing_cache(timing_cache, ignore_mismatch=True)

    config.add_optimization_profile(profile)

    # config.max_workspace_size = 4096 * (1 << 20)

    input_name = network.get_input(0).name
    profile.set_shape(input_name,
                      min=(1, 3, 224, 224),
                      opt=(1, 3, 224, 224),
                      max=(4, 3, 224, 224))
    config.add_optimization_profile(profile)
    
    config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = TRT_BUILDER.build_serialized_network(network, config)

    ## save TRT engine
    Path(trt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(trt_path, "wb") as f:
        f.write(serialized_engine)

    # save the timing cache
    timing_cache = config.get_timing_cache()
    with timing_cache.serialize() as buffer:
        with open(TIMING_CACHE, "wb") as f:
            f.write(buffer)
            f.flush()
            os.fsync(f)
            print(f"Wrote TRT timing cache to {TIMING_CACHE}")

    print(f"TRT Engine saved to {trt_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

model = model.visual.float().to("cuda")     # ensure float32 weights
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

onnx_path = "vit.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

if os.path.exists(onnx_path):
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model check passed!")

onnx_to_trt(onnx_path)