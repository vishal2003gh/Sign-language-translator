import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("Model/keras_model.h5", compile=False)
with open("Model/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Text-to-speech setup
engine = pyttsx3.init()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# GUI window
window = tk.Tk()
window.title("Sign Language Translator")
window.geometry("700x600")
label = Label(window, text="Prediction:", font=("Arial", 24))
label.pack(pady=20)

# Video frame
video_label = Label(window)
video_label.pack()

# Initialize webcam
cap = cv2.VideoCapture(0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detect_and_predict():
    ret, frame = cap.read()
    if not ret:
        return

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    prediction_text = "No hand detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            h, w, _ = frame.shape
            x1, y1 = int(x_min * w), int(y_min * h)
            x2, y2 = int(x_max * w), int(y_max * h)

            hand_img = frame[y1:y2, x1:x2]
            hand_img = cv2.resize(hand_img, (224, 224))
            img_array = np.asarray(hand_img, dtype=np.float32).reshape(1, 224, 224, 3)
            img_array = (img_array / 127.5) - 1

            prediction = model.predict(img_array)
            index = np.argmax(prediction)
            prediction_text = labels[index]

    # Show video frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    label.configure(text=f"Prediction: {prediction_text}")

    window.after(10, detect_and_predict)

def speak_prediction():
    text = label.cget("text").replace("Prediction: ", "")
    if text != "No hand detected":
        speak(text)

btn_speak = Button(window, text="Speak", command=speak_prediction, font=("Arial", 16))
btn_speak.pack(pady=20)

# Start detection loop
detect_and_predict()

# GUI loop
window.mainloop()