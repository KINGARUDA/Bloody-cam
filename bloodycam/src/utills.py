import torch
import time
import serial
import pyttsx3
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

arduino = serial.Serial('COM3', 9600, timeout=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
yolo_model = YOLO("yolov8l.pt")
engine = pyttsx3.init()

def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"get_embedding: cannot open {img_path}: {e}")
        return None

    face = mtcnn(img)   # [3,160,160] or None
    if face is None:
        print(f"get_embedding: no face found in {img_path}")
        return None

    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(face).cpu().numpy()[0]
    return emb

def cosine_similarity(a, b):
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_body_center(frame):
    results = yolo_model(frame, verbose=False)
    best = None
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best = (x1, y1, x2, y2)
    if best is None:
        return None, None
    x1, y1, x2, y2 = best
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return cx, cy

def send_command(cmd):
    arduino.write(cmd.encode())
    print(f"Sent: {cmd}")
    time.sleep(0.05)

def decision_box(cx, cy, frame_center_x, frame_center_y, tolerance):
    commands = []
    if cx < frame_center_x - tolerance:
        commands.append('L')
    elif cx > frame_center_x + tolerance:
        commands.append('R')

    if cy < frame_center_y - tolerance:
        commands.append('D')
    elif cy > frame_center_y + tolerance:
        commands.append('U')
    
    return commands

