# 🎭 AI Gesture & Emotion Controlled Emote System

## 🚀 Overview

The **AI Gesture & Emotion Controlled Emote System** is a real-time computer vision application that detects hand gestures and facial expressions to trigger dynamic emotes or multimedia responses.

This project combines **Computer Vision + Machine Learning** to enable interactive, touchless control using gestures and emotions.

---

## 🧠 Features

* ✋ Multi-hand gesture recognition

  * Peace ✌️
  * Namaste 🙏
  * Double index 👉👉
  * Thumbs up 👍

* 😊 Facial expression detection

  * Smile 😄
  * Tongue-out 😛

* 🎬 Real-time emote / multimedia triggering

* ⏱️ Cooldown mechanism to avoid repeated triggers

* 🧩 Modular architecture (easy to extend)

---

## 🛠️ Tech Stack

* **Python**
* **OpenCV** – video processing
* **MediaPipe** – hand & face landmark detection
* **NumPy** – numerical computations

---

## ⚙️ How It Works

1. Captures live video using webcam
2. Detects hand & face landmarks using MediaPipe
3. Identifies gestures using landmark positions
4. Detects facial expressions using facial features
5. Maps gesture/emotion → specific emote/action
6. Applies cooldown logic to prevent repeated triggers

---

## 📂 Project Structure

```
EmoteApp/
│
├── main.py
├── gesture/
├── emotion/
├── utils/
├── assets/
└── README.md
```

---

## ▶️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/Pratham1875/AI-Gesture-Emote-App.git
cd AI-Gesture-Emote-App
```

### 2️⃣ Install Dependencies

```
pip install opencv-python mediapipe numpy
```

### 3️⃣ Run Project

```
python main.py
```

---

## 📊 Applications

* 🎮 Gaming controls
* 📺 Streaming & content creation
* 🧑‍💻 Touchless interfaces
* 🤖 Human-computer interaction

---

## ⚡ Challenges Faced

* Handling continuous gesture detection → solved using cooldown
* Avoiding false positives → improved landmark conditions
* Real-time performance optimization

---

## 🔮 Future Improvements

* Add deep learning-based gesture classification
* Improve facial emotion accuracy
* Web-based deployment (Flask/Streamlit)
* Integration with AR/VR systems

---

## 👨‍💻 Author

**Pratham Chauhan**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
