import cv2
import mediapipe as mp

import warnings
warnings.filterwarnings("ignore")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_timestamps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect_for_video(
        mp_image,
        frame_timestamps
    )
    frame_timestamps += 1

    # ----------------------
    # Draw Face Mesh
    # ----------------------

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0,255,0),
                    -1
                )

            for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                start = connection.start
                end =  connection.end

                x1 = int(
                    face_landmarks[start].x * frame.shape[1]
                )
                y1 = int(
                    face_landmarks[start].y * frame.shape[0]
                )

                x2 = int(
                    face_landmarks[end].x * frame.shape[1]
                )

                y2 = int(
                    face_landmarks[end].y * frame.shape[0]
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1
                )


    cv2.imshow(
        "MediaPipe Face Mesh - Task API",
        frame
    )

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()