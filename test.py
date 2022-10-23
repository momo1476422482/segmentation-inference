from torchvision import models
import torch
import numpy as np
import cv2
from pathlib import Path
import torchvision.transforms as T

from PIL import Image


class Inference_seg_model:
    # ==================================================================================
    def __init__(self, name_model: str):
        self.name_model = name_model
        if self.name_model == "fcn_resnet101":
            self.model = models.segmentation.fcn_resnet101(pretrained=True).eval()
        elif self.name_model == "deeplab3_resnet101":
            self.model = models.segmentation.deeplab3_resnet101(pretrained=True).eval()

    # ==================================================================================
    @staticmethod
    def decode_segmap(image, nc=21):
        label_colors = np.array(
            [
                (0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0),
                (0, 128, 0),
                (128, 128, 0),
                (0, 0, 128),
                (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128),
                (128, 128, 128),
                (64, 0, 0),
                (192, 0, 0),
                (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0),
                (64, 0, 128),
                (192, 0, 128),
                (64, 128, 128),
                (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0),
                (128, 64, 0),
                (0, 192, 0),
                (128, 192, 0),
                (0, 64, 128),
            ]
        )
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        return np.stack([r, g, b], axis=2)

    # =================================================
    def segment(self, path: Path):
        img = cv2.imread(str(path))
        PIL_image = Image.open(str(path))
        # Comment the Resize and CenterCrop for better inference results
        trf = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        inp = trf(PIL_image).unsqueeze(0)
        print(inp.shape)
        out = self.model(inp)["out"]
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb = self.decode_segmap(om)
        cv2.imwrite("result.png", rgb)


# ===============================
if __name__ == "__main__":
    infseg = Inference_seg_model("fcn_resnet101")
    infseg.segment(Path(__file__).parent / "resized_frame_058.png")
