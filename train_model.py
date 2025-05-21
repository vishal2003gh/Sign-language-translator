# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.optimizers import Adam

# # Constants
# DATA_DIR = "Data"
# IMG_SIZE = 64  # You can also use 128

# # Load images and labels
# images = []
# labels = []
# label_dict = {}

# for i, label in enumerate(sorted(os.listdir(DATA_DIR))):
#     label_dict[i] = label
#     for img_name in os.listdir(os.path.join(DATA_DIR, label)):
#         img_path = os.path.join(DATA_DIR, label, img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         images.append(img)
#         labels.append(i)

# images = np.array(images) / 255.0  # Normalize
# labels = to_categorical(labels)

# # Train-test split
# x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# # Model architecture
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(len(label_dict), activation='softmax')
# ])

# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32)

# # Save model
# model.save("Model/keras_model.h5")

# # Save labels
# with open("Model/labels.txt", "w") as f:
#     for i in range(len(label_dict)):
#         f.write(f"{label_dict[i]}\n")

# print("Model and labels saved successfully.")








import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# === Configuration ===
data_dir = 'Data'
img_size = 64
labels = sorted(os.listdir(data_dir))
num_classes = len(labels)

# Save label names to file
with open("Model/labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

# === Data Loading ===
data = []
target = []

print("Loading images...")
for label_index, label_name in enumerate(labels):
    folder_path = os.path.join(data_dir, label_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            target.append(label_index)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

X = np.array(data) / 255.0
y = to_categorical(np.array(target), num_classes)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Definition ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# === Training ===
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# === Save Model ===
os.makedirs("Model", exist_ok=True)
model.save("Model/keras_model.h5")
print("Model saved to Model/keras_model.h5")







# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # Configuration
# data_dir = 'Data'
# img_size = 224  # Increased size for better feature extraction
# batch_size = 16
# epochs = 10

# # Get labels
# labels = sorted(os.listdir(data_dir))
# num_classes = len(labels)

# # Save labels
# os.makedirs("Model", exist_ok=True)
# with open("Model/labels.txt", "w") as f:
#     f.write("\n".join(labels))

# # Data loading
# print("Loading images...")
# data = []
# target = []

# for label_idx, label in enumerate(labels):
#     label_dir = os.path.join(data_dir, label)
#     for img_name in os.listdir(label_dir):
#         img_path = os.path.join(label_dir, img_name)
#         try:
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (img_size, img_size))
#             data.append(img)
#             target.append(label_idx)
#         except Exception as e:
#             print(f"Error loading {img_path}: {e}")

# X = np.array(data)
# y = np.array(target)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

# # Convert to categorical
# y_train = np.eye(num_classes)[y_train]
# y_test = np.eye(num_classes)[y_test]

# # Data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=False,
#     fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255)

# # Model architecture
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(128, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
    
#     Flatten(),
#     Dropout(0.5),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# # Compile
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Callbacks
# callbacks = [
#     EarlyStopping(patience=5, restore_best_weights=True),
#     ReduceLROnPlateau(factor=0.1, patience=3)
# ]

# # Train
# print("Training model...")
# history = model.fit(
#     train_datagen.flow(X_train, y_train, batch_size=batch_size),
#     steps_per_epoch=len(X_train) // batch_size,
#     epochs=epochs,
#     validation_data=test_datagen.flow(X_test, y_test),
#     callbacks=callbacks
# )

# # Save
# model.save("Model/keras_model.h5")
# print("Model saved to Model/keras_model.h5")





# import os
# import json
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # Configuration
# data_dir = 'Data'
# num_features = 63  # 21 landmarks * 3 coordinates (x,y,z)

# # Get labels and load data
# labels = sorted(os.listdir(data_dir))
# num_classes = len(labels)

# # Save labels
# os.makedirs("Model", exist_ok=True)
# with open("Model/labels.txt", "w") as f:
#     f.write("\n".join(labels))

# # Load and prepare data
# X = []
# y = []

# print("Loading landmark data...")
# for label_idx, label in enumerate(labels):
#     label_dir = os.path.join(data_dir, label)
#     for file_name in os.listdir(label_dir):
#         if file_name.endswith('.json'):
#             file_path = os.path.join(label_dir, file_name)
#             with open(file_path) as f:
#                 data = json.load(f)
#                 X.append(data['landmarks'])
#                 y.append(label_idx)

# X = np.array(X)
# y = np.array(y)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

# # Convert to categorical
# y_train = np.eye(num_classes)[y_train]
# y_test = np.eye(num_classes)[y_test]

# # Model architecture
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(num_features,)),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(num_classes, activation='softmax')
# ])

# # Compile
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Callbacks
# callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

# # Train
# print("Training model...")
# history = model.fit(X_train, y_train,
#                     epochs=100,
#                     batch_size=32,
#                     validation_data=(X_test, y_test),
#                     callbacks=callbacks)

# # Save
# model.save("Model/keras_model.h5")
# print("Model saved to Model/keras_model.h5")