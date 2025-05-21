# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# offset = 20
# imgSize = 300

# folder = "data/C"
# counter = 0

# labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
#           "W", "X", "Y", "Z", "hello"]

# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to capture image")
#         break

#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         # Ensure the cropping region is within image bounds
#         y1, y2 = max(0, y - offset), min(y + h + offset, img.shape[0])
#         x1, x2 = max(0, x - offset), min(x + w + offset, img.shape[1])
#         imgCrop = img[y1:y2, x1:x2]

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#         aspectRatio = h / w

#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize

#         prediction, index = classifier.getPrediction(imgWhite, draw=False)

#         cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255),
#                       cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#         cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)

#     cv2.imshow("Image", imgOutput)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore

# Load model and labels
model = load_model("Model/keras_model.h5")
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Webcam
cap = cv2.VideoCapture(0)

img_size = 64  # Same as used during training

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box around hand
            h, w, _ = frame.shape
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            x_vals = [pt[0] for pt in lm_list]
            y_vals = [pt[1] for pt in lm_list]
            x_min, x_max = max(min(x_vals) - 20, 0), min(max(x_vals) + 20, w)
            y_min, y_max = max(min(y_vals) - 20, 0), min(max(y_vals) + 20, h)

            # Crop and preprocess
            hand_img = frame[y_min:y_max, x_min:x_max]
            try:
                hand_img = cv2.resize(hand_img, (img_size, img_size))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # Prediction
                prediction = model.predict(hand_img)[0]
                index = np.argmax(prediction)
                confidence = prediction[index]

                label = f"{labels[index]} ({confidence*100:.1f}%)"

                # Draw box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print("Error during prediction:", e)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model

# # Configuration
# conf_threshold = 0.7  # Only accept predictions with >70% confidence

# # Load model and labels
# model = load_model("Model/keras_model.h5")
# with open("Model/labels.txt", "r") as f:
#     labels = [line.strip() for line in f.readlines()]

# # MediaPipe setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process frame
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
    
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Get bounding box
#             h, w = frame.shape[:2]
#             x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
#             y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            
#             x1, y1 = max(0, min(x_coords) - 20), max(0, min(y_coords) - 20)
#             x2, y2 = min(w, max(x_coords) + 20), min(h, max(y_coords) + 20)
            
#             # Crop and preprocess
#             hand_img = frame[y1:y2, x1:x2]
#             if hand_img.size == 0:
#                 continue
                
#             hand_img = cv2.resize(hand_img, (224, 224))
#             hand_img = hand_img.astype('float32') / 255.0
#             hand_img = np.expand_dims(hand_img, axis=0)
            
#             # Predict
#             preds = model.predict(hand_img)[0]
#             max_idx = np.argmax(preds)
#             confidence = preds[max_idx]
            
#             if confidence > conf_threshold:
#                 label = f"{labels[max_idx]} ({confidence*100:.1f}%)"
#                 color = (0, 255, 0)
#             else:
#                 label = "Unknown"
#                 color = (0, 0, 255)
            
#             # Draw results
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     cv2.imshow("Sign Language Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model

# # Configuration
# conf_threshold = 0.7

# # Load model and labels
# model = load_model("Model/keras_model.h5")
# with open("Model/labels.txt", "r") as f:
#     labels = [line.strip() for line in f.readlines()]

# # MediaPipe setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False,
#                       max_num_hands=1,
#                       min_detection_confidence=0.7)

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
    
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw landmarks
#             mp.solutions.drawing_utils.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
#             # Extract landmarks
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.extend([lm.x, lm.y, lm.z])
            
#             # Predict
#             landmarks = np.array(landmarks).reshape(1, -1)
#             preds = model.predict(landmarks)[0]
#             max_idx = np.argmax(preds)
#             confidence = preds[max_idx]
            
#             if confidence > conf_threshold:
#                 label = f"{labels[max_idx]} ({confidence*100:.1f}%)"
#                 color = (0, 255, 0)
#             else:
#                 label = "Unknown"
#                 color = (0, 0, 255)
            
#             # Display
#             cv2.putText(frame, label, (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     cv2.imshow("Sign Language Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()