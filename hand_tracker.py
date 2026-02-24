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

    # Draw landmarks
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    # ===================================================
    # 1️⃣ TWO HAND GESTURES (NO CONFLICT VERSION)
    # ===================================================
    if len(results.multi_hand_landmarks) == 2:

        h1 = results.multi_hand_landmarks[0].landmark
        h2 = results.multi_hand_landmarks[1].landmark

        # ---------------- HANDS ON HEAD ----------------
        # Detect both hands raised high (above upper frame area)
        h, w, _ = frame.shape

        wrist1_y = int(h1[0].y * h)
        wrist2_y = int(h2[0].y * h)

        index1_y = int(h1[8].y * h)
        index2_y = int(h2[8].y * h)

        # Hands must be high in the frame (top 35%)
        hands_high = wrist1_y < int(h * 0.35) and wrist2_y < int(h * 0.35)

        # Fingers roughly above wrists (indicating upward placement)
        fingers_above = index1_y < wrist1_y and index2_y < wrist2_y

        if hands_high and fingers_above:
            return "hands_on_head"

        palm_dist = distance(h1[9], h2[9], frame.shape)

        h1_vertical = h1[8].y < h1[0].y
        h2_vertical = h2[8].y < h2[0].y

        # ---------------- NAMASTE (ROBUST VERSION) ----------------
        palm_dist = distance(h1[9], h2[9], frame.shape)

        # Hands must be vertical
        h1_vertical = h1[8].y < h1[0].y
        h2_vertical = h2[8].y < h2[0].y

        # Palms roughly facing each other (depth check)
        palms_facing = abs(h1[9].z - h2[9].z) < 0.04

        # Hands must be close both horizontally and vertically
        horizontal_close = abs(h1[9].x - h2[9].x) < 0.12
        vertical_close = abs(h1[9].y - h2[9].y) < 0.12

        if (
            palm_dist < 100 and
            h1_vertical and
            h2_vertical and
            palms_facing and
            horizontal_close and
            vertical_close
        ):
            return "namaste"

        # ---------------- BOTH PALMS FORWARD ----------------
        both_all_up = True

        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            all_up = (
                lm[8].y < lm[6].y and
                lm[12].y < lm[10].y and
                lm[16].y < lm[14].y and
                lm[20].y < lm[18].y
            )

            vertical = lm[8].y < lm[0].y

            if not (all_up and vertical):
                both_all_up = False

        if both_all_up and palm_dist > 150:
            return "both_palms_forward"

        # ---------------- DOUBLE INDEX ----------------
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

    # ===================================================
    # 2️⃣ SINGLE HAND GESTURES (ONLY IF 1 HAND)
    # ===================================================
    if len(results.multi_hand_landmarks) == 1:

        lm = results.multi_hand_landmarks[0].landmark


        # ---------- SINGLE INDEX UP ----------
        index_up_only = (
            lm[8].y < lm[6].y and      # index up
            lm[12].y > lm[10].y and    # middle down
            lm[16].y > lm[14].y and    # ring down
            lm[20].y > lm[18].y        # pinky down
        )

        # ---------- THINKING (FACE LEVEL INDEX) ----------
        # Only require index up and near upper half of frame (no bending needed)
        index_tip_y = lm[8].y
        index_high_in_frame = index_tip_y < 0.50

        # ---------- THINKING vs SINGLE INDEX (WITH BUFFER) ----------
        if not hasattr(process_hands, "thinking_counter"):
            process_hands.thinking_counter = 0
        if not hasattr(process_hands, "index_counter"):
            process_hands.index_counter = 0

        if index_up_only and index_high_in_frame:
            process_hands.thinking_counter += 1
            process_hands.index_counter = 0

            # Require a few frames to confirm thinking
            if process_hands.thinking_counter > 4:
                process_hands.thinking_counter = 0
                return "thinking"

        elif index_up_only:
            process_hands.index_counter += 1
            process_hands.thinking_counter = 0

            # Require slight delay before confirming single index
            if process_hands.index_counter > 6:
                process_hands.index_counter = 0
                return "single_index_up"

        else:
            process_hands.thinking_counter = 0
            process_hands.index_counter = 0


        # ---------- OPEN PALM ----------
        all_up = (
            lm[8].y < lm[6].y
            and lm[12].y < lm[10].y
            and lm[16].y < lm[14].y
            and lm[20].y < lm[18].y
        )

        if all_up:
            return "open_palm"

        # ---------- PEACE ----------
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_down = lm[16].y > lm[14].y
        pinky_down = lm[20].y > lm[18].y

        if index_up and middle_up and ring_down and pinky_down:
            return "peace"

       

    return None