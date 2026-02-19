import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def process_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    for face_landmarks in results.multi_face_landmarks:
        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face.FACEMESH_TESSELATION
        )

        lm = face_landmarks.landmark
        h, w, _ = frame.shape

        upper_lip_y = int(lm[13].y * h)
        lower_lip_y = int(lm[14].y * h)

        mouth_open = abs(lower_lip_y - upper_lip_y)

        if mouth_open > 28:
            return "tongue_out"

    return None
