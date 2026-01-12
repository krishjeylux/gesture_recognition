import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("model-2.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

labels = {0: "ZERO", 1: "ONE", 2: "TWO", 3:"THREE"}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([data])[0]
            gesture = labels[prediction]

            cv2.putText(frame, gesture, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
