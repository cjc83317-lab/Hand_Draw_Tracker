import cv2
import numpy as np
import mediapipe as mp

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Drawing canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Initial color
draw_color = (255, 0, 255)
eraser_mode = False

# Color picker boxes
colors = {
    "Pink": (255, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (0, 255, 255),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255)
}
color_boxes = [(i * 100 + 50, 50) for i in range(len(colors))]

# Previous point
prev_x, prev_y = None, None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw color boxes
    for i, ((x, y), color) in enumerate(zip(color_boxes, colors.values())):
        cv2.rectangle(img, (x, y), (x + 50, y + 50), color, cv2.FILLED)

    # Display shortcut guide
    cv2.putText(img, "Press - (E) Eraser, (C) Clear, (Q) Quit", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    # Show eraser status
    status_text = "Eraser ON" if eraser_mode else "Draw Mode"
    cv2.putText(img, status_text, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[8]  # Index fingertip
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Check color picker
            for i, (x, y) in enumerate(color_boxes):
                if x < cx < x + 50 and y < cy < y + 50:
                    draw_color = list(colors.values())[i]

            # Draw or erase
            if cy > 100:
                if prev_x is None or prev_y is None:
                    prev_x, prev_y = cx, cy
                if eraser_mode:
                    cv2.circle(canvas, (cx, cy), 30, (0, 0, 0), -1)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, 10)
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = None, None

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Combine canvas with webcam feed
    mask = canvas.astype(bool)
    img[mask] = canvas[mask]

    cv2.imshow("Hand Drawing Tracker", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    elif key == ord('e'):
        eraser_mode = not eraser_mode  # Toggle eraser

cap.release()
cv2.destroyAllWindows()