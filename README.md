----------------------------------SilentBridge AI---------------------------------------------------------

Breaking communication barriers through AI-powered sign recognition.

SilentBridge AI is a Python-based computer vision project designed to recognize static American Sign Language (ASL) alphabet hand signs and display predicted letters on screen.

The project explores how artificial intelligence, computer vision, and machine learning can help make communication more accessible to people who use sign language.

##  Features

- **Real-Time Webcam:** Captures live video using a webcam.
- **Hand Tracking:** Detects and tracks hand landmarks using MediaPipe.
- **Machine Learning:** Uses a Random Forest classifier trained on collected hand-landmark data.
- **Sign Prediction:** Implements a pipeline for predicting hand signs and displaying the results.
- **Custom Dataset:** Collects hand-landmark samples to train the recognition model.
- **Desktop Interface:** Provides a graphical interface using CustomTkinter for webcam-based recognition.

**Development status: Hand detection is implemented, while reliable real-time sign prediction is still being debugged and tested.

##  Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- CustomTkinter
- Pillow
- Joblib

##  Project Structure

SilentBridge/
├── app.py
├── collect_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── landmarks.csv
└── src/
    ├── models/
    │   └── sign_model.joblib
    ├── recognize_sign.py
    └── hand_tracking.py



## Getting Started

### 1. Install Dependencies

Python 3.12 is recommended for the project's computer vision and machine-learning dependencies.

Create a virtual environment:
python -m venv .venv
Activate the environment.

**macOS / Linux:**
source .venv/bin/activate

Install the required packages:
python -m pip install -r requirements.txt

### 2. Collect Training Data

Run the data collection script:
python collect_data.py
Follow the terminal instructions to collect samples for each supported static hand sign.

The collected hand-landmark data is saved to:
dataset/landmarks.csv

Collect varied samples under different hand positions and recording conditions to help the model learn more robust patterns.

### 3. Train the Model

After collecting sufficient training data, run:
python train_model.py
The script trains a Random Forest