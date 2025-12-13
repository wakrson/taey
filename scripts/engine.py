import os
from pathlib import Path
from typing import Optional, Any, List, Tuple

import tensorrt as trt
import numpy as np
import numpy.typing as npt
import cupy as cp
from skimage.transform import resize

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class Engine:
    def __init__(self, model_path: Path, channel_first: Optional[bool] = True):
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        self.stream = None
        self.logger = trt.Logger(trt.Logger.INFO)

        if model_path.suffix == '.onnx':
            engine_path = Path(str(model_path).replace('onnx', 'engine'))
            if engine_path.exists() is False:
                self.build_tensorrt(model_path)
            model_path = engine_path
        
        self.engine_path = model_path
        self.engine = self.load()
        self.buffers_allocated = False
        self.context = self.engine.create_execution_context()
        self.channel_first = channel_first

    @property
    def input_names(self):
        inputs = []
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
        return inputs

    @property
    def output_names(self):
        outputs = []
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        return outputs
    
    @property
    def input_shapes(self):
        shapes = [ self.engine.get_tensor_shape(name) for name in self.input_names ]
        return shapes

    @property
    def input_dtypes(self):
        shapes = [ self.engine.get_tensor_dtype(name) for name in self.input_names ]
        return shapes

    def load(self):
        with open(self.engine_path, 'rb') as f:
            # Create a runtime object
            runtime = trt.Runtime(self.logger)
            # Deserialize the engine data
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine

    def create_builder(
        self,
        model_path: Path
    ) -> Tuple[trt.Builder, trt.INetworkDefinition, trt.OnnxParser, trt.IOptimizationProfile]:
        # Initialize builder
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, self.logger)
        
        parse_success = onnx_parser.parse_from_file(str(model_path))
        if parse_success is False:
            raise RuntimeError(f"Unable to parse onnx : {model_path}")
        for idx in range(onnx_parser.num_errors):
            raise RuntimeError(f"ONNX model parsing failed : {onnx_parser.get_error(idx)}")
        
        return builder, network, onnx_parser
        
    def build_tensorrt(
        self,
        model_path: Path,
        builder: Optional[trt.Builder] = None,
        network: Optional[trt.INetworkDefinition] = None,
        profile: Optional[trt.IOptimizationProfile] = None
    ) -> None:
        if builder is None:
            builder, network, _ = self.create_builder(model_path)
        
        print(f"Making TRT engine")
        engine_path = Path(str(model_path).replace('onnx', 'engine'))

        # Add profile
        config = builder.create_builder_config()
        #config.set_flag(trt.BuilderFlag.FP16)
        if profile is not None:
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)

        # Save TensorRT engine
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"TRT Engine saved to {engine_path}")

    def allocate_buffers(self):
        self.stream = cp.cuda.Stream()
        self.buffers_allocated = True
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)

            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Allocate host and device buffers
            device_mem = cp.zeros(shape, dtype=dtype)

            self.context.set_tensor_address(tensor_name, int(device_mem.data.ptr))
            
            # Append to device
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs[tensor_name] = HostDeviceMem(None, device_mem)
            else:
                self.outputs[tensor_name] = HostDeviceMem(None, device_mem)

    def resize(self, imgs: npt.NDArray[Any], imgsz: Tuple[int, int]) -> npt.NDArray[Any]:
        output = []
        for b in range(imgs.shape[0]):
            img = resize(np.transpose(imgs[b], axes=(1, 2, 0)), imgsz)
            img = np.transpose(img, axes=(2, 0, 1))
            output.append(img)
        return np.asarray(output)

    def normalize(self, imgs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return imgs