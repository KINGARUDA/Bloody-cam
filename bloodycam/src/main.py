import utills  as ut
from notification import send_telegram_notification as telegram
import cv2
import serial
import time
import threading
from notification import send_telegram_notification as telegram
import FaceRecognition as fc


arduino = ut.arduino
engine = ut.engine
time.sleep(2)

arduino.write(b'I')
print("Servo sweeping started (searching for face)...")

model = ut.yolo_model

frame_center_x, frame_center_y = 320, 240
tolerance = 50  

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

count = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  
                count += 1
                print(f"Person detected!")
                break 

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 2:
        print("Reached 10 detections — stopping.")
        break

cap.release()
cv2.destroyAllWindows()

arduino.write(b'F')
print("Intruder Detected — stopping sweep, switching to tracking mode.")

engine.say("Intruder Alert! Someone is there")
engine.runAndWait()
engine.stop()
time.sleep(0.2)

telegram("Intruder Alert! Someone is there")

ans = fc.face_recognition(kf)
print(ans)

engine.say("Hello, Dear User")
engine.runAndWait()
engine.stop()
time.sleep(0.2)
print("Hello, Dear User")
if ans:
    engine.say("Hello User, welcome")
else :
    engine.say("Hello Unknown Person, you are not authorized to access this area")
    engine.runAndWait()
    time.sleep(0.2)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cx, cy, box = ut.get_body_center(frame)

    if cx is not None and cy is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        commands = fc.decision_box(cx, cy, frame_center_x, frame_center_y, tolerance)
        for cmd in commands:
            fc.send_command(cmd)   

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

engine.stop()
arduino.close()