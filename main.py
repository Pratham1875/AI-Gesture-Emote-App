import cv2
import time

from hand_tracker import process_hands
from face_tracker import process_face
from video_player import play_video
from gif_player import play_gif


cap = cv2.VideoCapture(0)

last_gesture = None
last_trigger_time = 0
cooldown = 3


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

    cv2.imshow("Emote Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
