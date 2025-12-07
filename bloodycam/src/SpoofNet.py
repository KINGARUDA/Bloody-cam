import torch
import torch.nn as nn
from PIL import Image
import cv2
from torchvision import transforms
from utills import device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class SpoofNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        dense = torch.hub.load('pytorch/vision:v0.13.1', 'densenet161', pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = torch.sigmoid(dec)
        out = self.linear(out_map.view(-1, 14 * 14))
        out = torch.sigmoid(out)
        out = torch.flatten(out)
        return out_map, out

spoof_model = SpoofNet().to(device)
spoof_model.eval()
state = torch.load(r"DeePixBiS.pth", map_location=device)
spoof_model.load_state_dict(state, strict=False)

def check_spoof(frame, face_box, thr=0.8):
    if face_box is None:
        return False, 0.0
    x1, y1, x2, y2 = map(int, face_box)
    h, w = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return False, 0.0

    face_crop = frame[y1:y2, x1:x2]
    img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, out = spoof_model(img)
        prob_real = float(out.item())
    return (prob_real >= thr), prob_real
