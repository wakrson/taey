import os
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Dict, List
import logging

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import numpy.typing as npt
import tensorrt as trt

from engine import Engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIP(Engine):
    def __init__(self, model_path: Path, channel_first: Optional[bool] = True):
        super().__init__(Path(model_path), channel_first)

    def __call__(self, img):
        x = img.copy()
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        
        # Make channel first
        if self.channel_first is False:
            img = np.transpose(img, axes=(0, -1, 1, 2))
        
        imgsz = self.input_shapes[0][2:]
        img = self.resize(img, imgsz).copy()

        processed_img = self.normalize(img).copy()
        processed_img = processed_img.astype(trt.nptype(self.input_dtypes[0]))

        # Allocate buffers
        if not self.buffers_allocated:
            self.allocate_buffers()

        # Move to GPU
        self.inputs['input'].device.set(processed_img, self.stream)
    
        self.context.execute_async_v3(stream_handle=self.stream.ptr)
        self.stream.synchronize()

        logger.info(self.outputs['output'].device[0].shape)
        return x

    def build_tensorrt(self, model_path: Path) -> None:
        builder, network, parser = self.create_builder(model_path)
        profile = builder.create_optimization_profile()
        profile.set_shape('input', min=(1, 3, 224, 224), opt=(4, 3, 224, 224), max=(8, 3, 224, 224))
        super().build_tensorrt(model_path, builder, network, parser)

def main():
    clip = CLIP('models/clip.onnx')

    dataset = torchvision.datasets.CocoDetection(
        root='datasets/coco/images/val2017',
        annFile='datasets/coco/annotations/instances_val2017.json',
        transform=ToTensor()
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for img, _ in dataloader:
        img = img.cpu().numpy()
        predictions = clip(img)
        print(predictions.shape)
        
if __name__ == '__main__':
    main()