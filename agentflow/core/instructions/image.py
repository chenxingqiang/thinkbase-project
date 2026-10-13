"""Image processing instruction backed by a Vision Transformer (ViT).

The instruction is exposed as a Ray actor (``ImageProcessingInstruction``)
so image workloads can be scheduled distributedly.  The plain
implementation class is kept separate (``_ImageProcessingInstruction``) so
that its logic can also be exercised without a running Ray cluster.

When ``use_mock=True`` the actor avoids downloading any weights and uses
lightweight in-process stand-ins for the processor/model pair, which keeps
tests hermetic and fast.  The real path loads
``ViTImageProcessor``/``ViTForImageClassification`` from the transformers
hub.
"""

from typing import Any, Dict, List, Optional, Tuple

import logging

import numpy as np
import ray
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor

try:  # pragma: no cover - PIL is optional at import time
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None

logger = logging.getLogger(__name__)

# Default ViT checkpoint and input geometry.
DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"
DEFAULT_TARGET_SIZE = (224, 224)
NUM_MOCK_CLASSES = 1000


def _to_tensor(image: Any) -> torch.Tensor:
    """Convert PIL/ndarray/tensor input to a ``float`` CHW tensor."""
    if torch.is_tensor(image):
        return image.float()
    if isinstance(image, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(image)).float()
    if PILImage is not None and isinstance(image, PILImage.Image):
        array = np.array(image.convert("RGB"))
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
    raise TypeError(f"Unsupported image type: {type(image)!r}")


class _ModelOutput:
    """Minimal output object exposing ``logits`` (like HF model outputs)."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _MockViTProcessor:
    """Lightweight stand-in for ``ViTImageProcessor`` (no network access)."""

    def __call__(self, images: Any, return_tensors: str = "pt", **kwargs: Any) -> Dict[str, Any]:
        if isinstance(images, (list, tuple)):
            pixel_values = torch.stack([self._normalize(image) for image in images])
        else:
            pixel_values = self._normalize(images).unsqueeze(0)
        return {"pixel_values": pixel_values}

    @staticmethod
    def _normalize(image: Any) -> torch.Tensor:
        tensor = _to_tensor(image)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        return tensor


class _MockViTModel:
    """Lightweight stand-in for ``ViTForImageClassification``."""

    def __call__(self, pixel_values: Optional[torch.Tensor] = None, **kwargs: Any) -> _ModelOutput:
        if pixel_values is None:
            batch_size = 1
        else:
            batch_size = int(pixel_values.shape[0]) if pixel_values.dim() >= 1 else 1
        logits = torch.randn(batch_size, NUM_MOCK_CLASSES)
        return _ModelOutput(logits=logits)


class _ImageProcessingInstruction:
    """Classify images with a ViT model (plain, non-actor implementation)."""

    def __init__(
        self,
        use_mock: bool = False,
        model_name: str = DEFAULT_MODEL_NAME,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        batch_size: int = 32,
        min_batch_size: int = 2,
    ):
        self.name = "process_image"
        self.use_mock = bool(use_mock)
        self.model_name = model_name
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.batch_size = max(1, int(batch_size))
        self.min_batch_size = max(1, int(min_batch_size))

        if self.use_mock:
            self.processor = _MockViTProcessor()
            self.model = _MockViTModel()
        else:
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(model_name)
            self.model.eval()

    # -- introspection -------------------------------------------------

    def get_name(self) -> str:
        """Return the instruction name."""
        return self.name

    def get_processor(self) -> Any:
        """Return the underlying image processor."""
        return self.processor

    def get_model(self) -> Any:
        """Return the underlying classification model."""
        return self.model

    # -- preprocessing helpers ------------------------------------------

    @staticmethod
    def _spatial_size(image: Any) -> Tuple[int, int]:
        """Return the (height, width) of an image-like input."""
        if torch.is_tensor(image) or isinstance(image, np.ndarray):
            shape = tuple(image.shape)
        elif PILImage is not None and isinstance(image, PILImage.Image):
            width, height = image.size
            return int(height), int(width)
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")
        if len(shape) < 2:
            raise ValueError(f"Cannot infer spatial size from shape {shape}")
        return int(shape[-2]), int(shape[-1])

    def should_resize_image(self, context: Dict[str, Any]) -> bool:
        """Check whether the context image exceeds the target geometry."""
        if not isinstance(context, dict):
            return False
        image = context.get("image")
        if image is None:
            return False
        try:
            height, width = self._spatial_size(image)
        except (TypeError, ValueError):
            return False
        max_height, max_width = self.target_size
        return height > max_height or width > max_width

    def should_batch_process(self, context: Dict[str, Any]) -> bool:
        """Check whether the context contains enough images to batch."""
        if not isinstance(context, dict):
            return False
        images = context.get("images")
        if not isinstance(images, (list, tuple)) or not images:
            return False
        return len(images) >= self.min_batch_size

    def resize_image(self, image: Any, size: Tuple[int, int]) -> torch.Tensor:
        """Resize an image tensor to ``(height, width)`` via bilinear interp."""
        tensor = _to_tensor(image)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        height, width = int(size[0]), int(size[1])
        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return an optimized copy of the context (downscaling large images)."""
        optimized = dict(context)
        if "image" in optimized:
            if self.should_resize_image({"image": optimized["image"]}):
                optimized["image"] = self.resize_image(optimized["image"], self.target_size)
        if "images" in optimized and isinstance(optimized["images"], (list, tuple)):
            optimized["images"] = [
                self.resize_image(image, self.target_size)
                if self.should_resize_image({"image": image})
                else image
                for image in optimized["images"]
            ]
        return optimized

    # -- execution -------------------------------------------------------

    def _to_numpy_hwc(self, image: Any) -> np.ndarray:
        """Convert to the HWC uint8/float numpy arrays HF processors expect."""
        tensor = _to_tensor(image)
        if tensor.dim() == 3:
            array = tensor.detach().cpu().permute(1, 2, 0).numpy()
        else:
            array = tensor.detach().cpu().numpy()
        return np.ascontiguousarray(array)

    @staticmethod
    def _logits_to_result(logits: torch.Tensor) -> Dict[str, Any]:
        """Convert one row of logits into a prediction record."""
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        return {
            "logits": logits.detach().cpu(),
            "predicted_class": int(predicted.item()),
            "confidence": float(confidence.item()),
        }

    def _forward(self, images: List[Any]) -> List[Dict[str, Any]]:
        """Run the processor + model over a list of images."""
        if self.use_mock:
            inputs = self.processor(list(images))
            with torch.no_grad():
                outputs = self.model(pixel_values=inputs["pixel_values"])
        else:
            arrays = [self._to_numpy_hwc(image) for image in images]
            inputs = self.processor(images=arrays, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
        logits = outputs.logits
        return [self._logits_to_result(logits[index]) for index in range(logits.shape[0])]

    def _process_single(self, image: Any) -> Dict[str, Any]:
        """Preprocess and classify a single image."""
        tensor = _to_tensor(image)
        if self.should_resize_image({"image": tensor}):
            tensor = self.resize_image(tensor, self.target_size)
        return self._forward([tensor])[0]

    def _process_batch(self, images: List[Any]) -> List[Dict[str, Any]]:
        """Classify images in batches of ``self.batch_size``."""
        results: List[Dict[str, Any]] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            results.extend(self._forward(chunk))
        return results

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process either a single image or a list of images."""
        if not isinstance(context, dict):
            raise ValueError(f"Context must be a dict, got {type(context)!r}")

        if "images" in context:
            images = context["images"]
            if not isinstance(images, (list, tuple)):
                raise ValueError("Context 'images' must be a list of images")
            if not images:
                raise ValueError("Context 'images' must not be empty")
            if self.should_batch_process(context):
                results = self._process_batch(list(images))
            else:
                results = [self._process_single(image) for image in images]
        elif "image" in context:
            results = [self._process_single(context["image"])]
        else:
            raise ValueError("Context must contain an 'image' or 'images' key")

        return {"results": results}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Public alias for :meth:`_execute_impl`."""
        return self._execute_impl(context)


# Expose the instruction as a Ray actor so it can be scheduled
# distributedly; tests drive it through ``ImageProcessingInstruction.remote()``.
ImageProcessingInstruction = ray.remote(_ImageProcessingInstruction)
