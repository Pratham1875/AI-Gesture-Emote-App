import cv2
import time

from hand_tracker import process_hands
from face_tracker import process_face
from video_player import play_video
from gif_player import play_gif
from image_player import show_image


cap = cv2.VideoCapture(0)

last_gesture = None
last_trigger_time = 0
cooldown = 3

active_image = None
image_end_time = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    hand_gesture = process_hands(frame)
    face_gesture = process_face(frame)

    gesture = None

    # Hand priority
    if hand_gesture:
        gesture = hand_gesture
    elif face_gesture:
        gesture = face_gesture

    if gesture:
        last_gesture = gesture

        if current_time - last_trigger_time > cooldown:

            if gesture == "peace":
                play_video("videos/peace.mp4")

            elif gesture == "thumbs_up":
                play_video("videos/thumbs.mp4")

            elif gesture == "namaste":
                play_video("videos/namaste.mp4")

            elif gesture == "double_index":
                play_video("videos/double_index.mp4")
                
            elif gesture == "both_palms_forward":
                play_gif("gifs/both_palms_forward.gif")
                
            elif gesture == "hands_on_head":
                play_gif("gifs/hands_on_head.gif")

            elif gesture == "tongue_out":
                play_video("videos/tongue_out.mp4")
                
            elif gesture == "thinking":
                active_image = cv2.imread("images/thinking.jpeg")
                image_end_time = current_time + 2
                
            elif gesture == "single_index_up":
                active_image = cv2.imread("images/single_index.jpeg")
                image_end_time = current_time + 2

            last_trigger_time = current_time

    if last_gesture:
        cv2.putText(
            frame,
            f"Gesture: {last_gesture}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # ==============================
    # IMAGE OVERLAY (CENTERED)
    # ==============================
    if active_image is not None and current_time < image_end_time:
        overlay = cv2.resize(active_image, (500, 500))

        h, w, _ = frame.shape
        oh, ow, _ = overlay.shape

        x1 = w // 2 - ow // 2
        y1 = h // 2 - oh // 2
        x2 = x1 + ow
        y2 = y1 + oh

        frame[y1:y2, x1:x2] = overlay
    elif current_time >= image_end_time:
        active_image = None

    cv2.imshow("Emote Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
