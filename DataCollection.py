# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# offset = 20
# imgSize = 300
# folder = "Data/C"
# counter = 0
# while True:
#     success, img = cap.read()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#         imgCropShape = imgCrop.shape
#         aspectRatio = h / w
#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == ord("s"):
#         counter += 1
#         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
#         print(counter)





# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import os

# # === CONFIGURATION ===
# folder = "Data/B"  # Change this for different letters
# os.makedirs(folder, exist_ok=True)
# imgSize = 300
# offset = 20

# # === INIT MEDIA PIPE ===
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(static_image_mode=False,
#                       max_num_hands=1,
#                       min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)
# counter = 0

# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to read from webcam.")
#         break

#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     imgOutput = img.copy()

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             # Get bounding box from landmarks
#             h, w, _ = img.shape
#             cx_min, cy_min = w, h
#             cx_max, cy_max = 0, 0

#             for lm in handLms.landmark:
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cx_min = min(cx_min, cx)
#                 cy_min = min(cy_min, cy)
#                 cx_max = max(cx_max, cx)
#                 cy_max = max(cy_max, cy)

#             # Apply offset and crop
#             x1 = max(0, cx_min - offset)
#             y1 = max(0, cy_min - offset)
#             x2 = min(w, cx_max + offset)
#             y2 = min(h, cy_max + offset)

#             imgCrop = img[y1:y2, x1:x2]
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#             aspectRatio = (y2 - y1) / (x2 - x1)

#             if aspectRatio > 1:
#                 k = imgSize / (y2 - y1)
#                 wCal = int(k * (x2 - x1))
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 wGap = (imgSize - wCal) // 2
#                 imgWhite[:, wGap:wCal + wGap] = imgResize
#             else:
#                 k = imgSize / (x2 - x1)
#                 hCal = int(k * (y2 - y1))
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 hGap = (imgSize - hCal) // 2
#                 imgWhite[hGap:hCal + hGap, :] = imgResize

#   #          mpDraw.draw_landmarks(imgWhite, handLms, mpHands.HAND_CONNECTIONS)    

#             cv2.imshow("Cropped", imgCrop)
#             cv2.imshow("Resized", imgWhite)

#     cv2.imshow("Webcam", imgOutput)

#     key = cv2.waitKey(1)
#     if key == ord('s'):
#         counter += 1
#         file_path = os.path.join(folder, f"Image_{time.time()}.jpg")
#         cv2.imwrite(file_path, imgWhite)
#         print(f"Saved {file_path} ({counter})")

#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Configuration
folder = "Data/B"  # Change for different letters
os.makedirs(folder, exist_ok=True)
imgSize = 224  # Consistent with model input size
offset = 30

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True,  # Changed to True for better single images
                     max_num_hands=1,
                     min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
counter = 0

print(f"Collecting images for {folder}. Press 's' to save, 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    imgOutput = img.copy()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get bounding box
            h, w, _ = img.shape
            x_coords = [int(lm.x * w) for lm in handLms.landmark]
            y_coords = [int(lm.y * h) for lm in handLms.landmark]
            
            x1, x2 = max(0, min(x_coords) - offset), min(w, max(x_coords) + offset)
            y1, y2 = max(0, min(y_coords) - offset), min(h, max(y_coords) + offset)
            
            # Crop and resize
            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:
                continue
                
            aspect_ratio = imgCrop.shape[0] / imgCrop.shape[1]
            
            if aspect_ratio > 1:
                new_height = imgSize
                new_width = int(imgSize / aspect_ratio)
                imgResize = cv2.resize(imgCrop, (new_width, new_height))
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                start = (imgSize - new_width) // 2
                imgWhite[:, start:start+new_width] = imgResize
            else:
                new_width = imgSize
                new_height = int(imgSize * aspect_ratio)
                imgResize = cv2.resize(imgCrop, (new_width, new_height))
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                start = (imgSize - new_height) // 2
                imgWhite[start:start+new_height, :] = imgResize

            cv2.imshow("Hand Image", imgWhite)

    cv2.imshow("Webcam", imgOutput)

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        timestamp = int(time.time())
        file_path = os.path.join(folder, f"{timestamp}.jpg")
        cv2.imwrite(file_path, imgWhite)  # Save without landmarks
        print(f"Saved {file_path} (Total: {counter})")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import time
# import json

# # Configuration
# folder = "Data/B"  # Change for different letters
# os.makedirs(folder, exist_ok=True)

# # MediaPipe setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True,
#                       max_num_hands=1,
#                       min_detection_confidence=0.7)

# cap = cv2.VideoCapture(0)
# counter = 0

# print(f"Collecting landmarks for {folder}. Press 's' to save, 'q' to quit")

# while True:
#     success, img = cap.read()
#     if not success:
#         continue

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     img_output = img.copy()

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw landmarks (optional)
#             mp.solutions.drawing_utils.draw_landmarks(
#                 img_output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
#             # Extract landmark coordinates
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.extend([lm.x, lm.y, lm.z])  # 3D coordinates

#     cv2.imshow("Webcam", img_output)

#     key = cv2.waitKey(1)
#     if key == ord('s') and results.multi_hand_landmarks:
#         counter += 1
#         timestamp = int(time.time())
        
#         # Save landmarks as JSON
#         file_path = os.path.join(folder, f"{timestamp}.json")
#         with open(file_path, 'w') as f:
#             json.dump({
#                 "label": os.path.basename(folder),
#                 "landmarks": landmarks
#             }, f)
        
#         print(f"Saved landmarks {file_path} (Total: {counter})")
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







