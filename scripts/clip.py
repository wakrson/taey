from pathlib import Path
from typing import Optional, Any
import logging

import numpy as np
import numpy.typing as npt
import tensorrt as trt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from scripts.engine import Engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIP(Engine):
    def __init__(self, model_path: Path, channel_first: Optional[bool] = True):
        super().__init__(Path(model_path), channel_first)

    def __call__(self, imgs: npt.NDArray[Any]):
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, axis=0)
        
        # Make channel first
        if self.channel_first is False:
            imgs = np.transpose(imgs, axes=(0, -1, 1, 2))
        
        # Allocate buffers
        if not self.buffers_allocated:
            batch_size = imgs.shape[0]
            self.allocate_buffers(batch_size=batch_size)

        x = np.empty_like(imgs.astype(trt.nptype(self.input_dtypes[0])))
        for b in range(batch_size):
            x[b] = self.normalize(self.resize(imgs[b]))
        
        # Move to GPU
        self.inputs["input"].device.set(x, self.stream)  
        self.context.set_input_shape("input", (batch_size,) + (self.input_shapes[0][1:]))
        self.context.execute_async_v3(stream_handle=self.stream.ptr)
        self.stream.synchronize()
        return self.outputs["output"].device[0]

    def build_tensorrt(self, model_path: Path) -> None:
        builder, network, _ = self.create_builder(model_path)
        profile = builder.create_optimization_profile()
        profile.set_shape('input', min=(1, 3, 224, 224), opt=(4, 3, 224, 224), max=(8, 3, 224, 224))
        super().build_tensorrt(model_path, builder, network, profile)

def main():
    rootdir = Path(__file__).parent.parent
    clip = CLIP(f'{rootdir}/models/clip.onnx')

    dataset = datasets.CocoDetection(
        root=f'{rootdir}/datasets/coco/images/val2017',
        annFile=f'{rootdir}/datasets/coco/annotations/instances_val2017.json',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    img = next(iter(dataloader))[0].cpu().numpy()
    predictions = clip(img)
    logger.info(predictions.shape)
        
if __name__ == '__main__':
    main()