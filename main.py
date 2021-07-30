import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("Cannot open webcam")
pTime = 0

mpFaceDetect = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetect.FaceDetection()


while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, -1)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        bboxC = results.detections[0].location_data.relative_bounding_box
        ih, iw, ic = frame.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
            int(bboxC.width * iw), int(bboxC.height * ih)

        # crop into face
        try:
            (x, y, w, h) = bbox
            im = frame[y:y+h, x:x+h]
            imS = cv2.resize(
                im, (480, 800), interpolation=cv2.INTER_LINEAR)
            frame = imS
        except:
            frame = frame

    frm = frame[0:800, 0:480]
    frm = cv2.resize(frm, (480, 800), fx=0.5, fy=0.5,
                     interpolation=cv2.INTER_AREA)

    # FPS counter
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frm, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow('Face Tracking', frm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
