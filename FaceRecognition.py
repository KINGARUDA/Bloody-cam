import cv2
import torch
import time
import numpy as np
from PIL import Image
import utills as ut
import SpoofNet
from SpoofNet import check_spoof as check_spoof
import KnowFaces as kf

mtcnn = ut.mtcnn
resnet = ut.resnet
device = ut.device      

def face_recognition(known_faces: dict = kf.Faces, timeout=10):
    known_faces = known_faces or {}
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("face_recognition: Cannot open camera")
        return False

    start_time = time.time()
    print("face_recognition: started, timeout =", timeout)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ face_recognition: failed to grab frame")
                break

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs = mtcnn.detect(img_pil)

            if boxes is not None and len(boxes) > 0:
                best_idx = int(np.argmax(probs))
                best_box = boxes[best_idx]

                is_real, prob_real = check_spoof(frame, best_box, thr=0.7)

                face_tensor = mtcnn(img_pil)
                if face_tensor is not None:
                    if face_tensor.ndim == 3:
                        face_tensor = face_tensor.unsqueeze(0)
                    face_tensor = face_tensor.to(device)
                    with torch.no_grad():
                        emb = resnet(face_tensor).cpu().numpy()[0]

                    # Compare with known faces
                    for person, ref_emb in known_faces.items():
                        if ref_emb is None:
                            # debug
                            print(f"Skipping {person}: no reference embedding")
                            continue
                        sim = ut.cosine_similarity(emb, ref_emb)
                        print(f"compare -> {person} sim={sim:.3f} spoof_prob={prob_real:.3f}")
                        if sim > 0.70 and is_real:
                            print(f"Known face: {person} (sim={sim:.2f}, spoof={prob_real:.2f})")
                            return True

                # draw box & label
                x1, y1, x2, y2 = map(int, best_box)
                color = (0,255,0) if is_real else (0,0,255)
                label = "REAL" if is_real else f"SPOOF {prob_real:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("face_recognition: user exit")
                break
            if time.time() - start_time > timeout:
                print("face_recognition: timeout reached")
                break

    except Exception as e:
        print(f"face_recognition error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return False