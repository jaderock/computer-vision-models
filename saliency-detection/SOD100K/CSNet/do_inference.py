import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage.transform import resize

image = np.asarray(Image.open("photo.jpg").convert("RGB"))
height, width = image.shape[:2]

image = image.astype(np.float32) / 255.0
image = resize(
    image, (224, 224),
    mode="reflect", anti_aliasing=False,
)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

x = ((image - mean) / std).transpose(2, 0, 1)[None]
x = np.ascontiguousarray(x, dtype=np.float32)

session = ort.InferenceSession(
    "exported/csnet-L-x1.onnx",
    providers=["CPUExecutionProvider"],
)
mask = session.run(None, {"image": x})[0][0, 0]

mask = resize(
    mask, (height, width),
    mode="reflect", anti_aliasing=False,
)
Image.fromarray(
    (np.clip(mask, 0, 1) * 255).astype(np.uint8)
).save("saliency.png")
