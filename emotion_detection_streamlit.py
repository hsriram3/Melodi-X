import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from torchvision.models import resnet18, mobilenet_v2

# Streamlit page settings
st.set_page_config(page_title="MELODI-X", layout="centered")
st.title("🎧 MELODI-X: Real-time Emotion Detector")

# Emotion categories and recommendations
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
recommendations = {
    "angry": {
        "music": "https://www.youtube.com/watch?v=bg1sT4ILG0w",
        "exercise": "Try a high-intensity kickboxing or shadowboxing session to release tension."
    },
    "disgust": {
        "music": "https://www.youtube.com/watch?v=IQoTiIoEtXc",
        "exercise": "Practice deep breathing or take a short nature walk to reset your senses."
    },
    "fear": {
        "music": "https://www.youtube.com/watch?v=HNbcVKwYarY",
        "exercise": "Do a short guided meditation or grounding exercises to reduce anxiety."
    },
    "happy": {
        "music": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
        "exercise": "Dance freely or go for a fun bike ride to amplify your joy!"
    },
    "sad": {
        "music": "https://www.youtube.com/watch?v=1ZYbU82GVz4",
        "exercise": "Do gentle yoga or go for a peaceful walk in nature to soothe your mood."
    },
    "surprise": {
        "music": "https://www.youtube.com/watch?v=JGwWNGJdvx8",
        "exercise": "Take a moment to stretch or journal what surprised you—it can bring clarity."
    },
    "neutral": {
        "music": "https://www.youtube.com/watch?v=lFcSrYw-ARY",
        "exercise": "Try a short workout or breathing routine to energize and refocus your mind."
    }
}

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Define CNN model
class CNN_Model(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN_Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Instantiate all models (uninitialized)
cnn_model = CNN_Model()
resnet_model = resnet18(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 7)

mobilenet_model = mobilenet_v2(pretrained=True)
mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, 7)

# User chooses the model
model_choice = st.selectbox("Select the Emotion Detection Model", ["CNN", "ResNet18", "MobileNetV2"])

# Load weights for the selected model
if model_choice == "CNN":
    cnn_model.load_state_dict(torch.load("best_model_cnn18.pth", map_location="cpu"))
    model = cnn_model
elif model_choice == "ResNet18":
    resnet_model.load_state_dict(torch.load("best_model_resnet18.pth", map_location="cpu"))
    model = resnet_model
else:  # MobileNetV2
    mobilenet_model.load_state_dict(torch.load("best_model_mobilenet.pth", map_location="cpu"))
    model = mobilenet_model

model.eval()

# Webcam snapshot capture
if st.button("📸 Capture Snapshot"):
    st.info("Accessing webcam...")
    cap = cv2.VideoCapture(1)
    frame = None

    for _ in range(60):
        ret, temp = cap.read()
        if ret:
            frame = temp
    cap.release()

    if frame is not None:
        st.session_state['captured_frame'] = frame
        st.image(frame, channels="BGR", caption="Snapshot Captured")
    else:
        st.error(" Failed to capture image from webcam.")

# Detect mood from the captured snapshot
if st.button("Detect Mood"):
    if 'captured_frame' not in st.session_state:
        st.warning("Please capture a snapshot first.")
    else:
        frame = st.session_state['captured_frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No faces detected!")
        else:
            detected = []
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    tensor = transform(face).unsqueeze(0)
                    with torch.no_grad():
                        output = model(tensor)
                        pred = torch.argmax(output, dim=1).item()
                        emotion = emotion_labels[pred]
                        detected.append(emotion)

                    # Draw face box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            st.image(frame, channels="BGR", caption="Detected Emotions")

            for idx, emo in enumerate(detected, 1):
                rec = recommendations[emo]
                st.markdown(f"---")
                st.subheader(f" Person {idx}: **{emo.upper()}**")
                st.markdown(f"[ Music Recommendation]({rec['music']})")
                st.markdown(f" **Suggested Activity:** _{rec['exercise']}_")
