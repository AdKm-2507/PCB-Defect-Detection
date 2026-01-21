import cv2
from ultralytics import YOLO

model = YOLO("Final_Model.pt")

cap = cv2.VideoCapture(0)

prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

 
    annotated_frame = results[0].plot()

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    unstable = False

    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion_score = diff.mean()

        if motion_score > 5:   
            unstable = True

    prev_gray = gray

    if unstable:
        cv2.putText(
            annotated_frame,
            "Stabilize Camera",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("PCB Fault Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
