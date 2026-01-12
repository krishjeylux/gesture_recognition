import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
for i in range(4):
    gesture_label = int(input("Enter gesture label (0-3 ): "))
    samples = 0

    file = open("gestures.csv", "a", newline="")
    writer = csv.writer(file)

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                data.append(gesture_label)
                writer.writerow(data)
                samples += 1

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Samples: {samples}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collecting Data", frame)

        if samples >= 100:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    
    file.close()
cap.release()
cv2.destroyAllWindows()
