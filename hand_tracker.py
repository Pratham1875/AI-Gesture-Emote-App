import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def distance(p1, p2, frame_shape):
    h, w, _ = frame_shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def process_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
    
    if len(results.multi_hand_landmarks)==2:
        both_palms_forward=True
        
        for hand_landmarks in results.multi_hand_landmarks:
            lm =hand_landmarks.landmark
            
            all_up =(
                lm[8].y < lm[6].y and
                lm[12].y < lm[10].y and
                lm[16].y < lm[14].y and
                lm[20].y <lm[18].y
            )
            vertical = lm[8].y <lm[0].y
            
            palm_forward=lm[8].z <lm[0].z
            
            if not(all_up and vertical and palm_forward):
                both_palms_forward=False
        if both_palms_forward:
            return "both_palms_forward"

    # ==============================
    # 1️⃣ NAMASTE
    # ==============================
    if len(results.multi_hand_landmarks) == 2:
        h1 = results.multi_hand_landmarks[0].landmark
        h2 = results.multi_hand_landmarks[1].landmark

        index_dist = distance(h1[8], h2[8], frame.shape)
        middle_dist = distance(h1[12], h2[12], frame.shape)
        palm_dist = distance(h1[9], h2[9], frame.shape)

        h1_vertical = h1[8].y < h1[0].y
        h2_vertical = h2[8].y < h2[0].y

        if (
            palm_dist < 100
            and index_dist < 70
            and middle_dist < 70
            and h1_vertical
            and h2_vertical
        ):
            return "namaste"

    # ==============================
    # 2️⃣ DOUBLE INDEX
    # ==============================
    if len(results.multi_hand_landmarks) == 2:
        both_index_only = True

        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            index_up = lm[8].y < lm[6].y
            middle_down = lm[12].y > lm[10].y
            ring_down = lm[16].y > lm[14].y
            pinky_down = lm[20].y > lm[18].y

            if not (index_up and middle_down and ring_down and pinky_down):
                both_index_only = False

        if both_index_only:
            return "double_index"

    # ==============================
    # 3️⃣ SINGLE HAND
    # ==============================
    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark

        # Open palm
        all_up = (
            lm[8].y < lm[6].y
            and lm[12].y < lm[10].y
            and lm[16].y < lm[14].y
            and lm[20].y < lm[18].y
        )

        if all_up:
            return "open_palm"

        # Peace
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_down = lm[16].y > lm[14].y
        pinky_down = lm[20].y > lm[18].y

        if index_up and middle_up and ring_down and pinky_down:
            return "peace"

        # Thumbs up
        if lm[4].y < lm[3].y:
            return "thumbs_up"

    return None
