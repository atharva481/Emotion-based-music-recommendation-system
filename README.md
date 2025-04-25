🎵 Emotion-Based Music Recommendation System
This project is a real-time Emotion-Based Music Recommendation System that detects users' facial expressions using a webcam and recommends personalized music based on their emotional state. It combines deep learning for emotion recognition with a content-based recommendation engine using Spotify song features.

📌 Features
🎥 Real-time facial emotion detection using ResNet architecture

🎧 Personalized music recommendations using K-Nearest Neighbors (KNN)

🔄 Emotion-to-genre mapping for mood-based music curation

⚡ Fast response: < 5 seconds from detection to song suggestion

🎵 Integration with Spotify via Spotify URIs

💻 Simple web-based interface built with Streamlit

🛠️ Technologies Used
Python

OpenCV

ResNet (for facial emotion detection)

K-Nearest Neighbors (KNN)

Pandas & NumPy

Spotify Tracks Dataset

Streamlit (for GUI)

🚀 How to Run
Clone the Repository
git clone https://github.com/yourusername/emotion-music-recommendation.git
cd emotion-music-recommendation

Install Dependencies
pip install -r requirements.txt

Run the App
streamlit run app.py
🧠 How It Works
User enables webcam.

Facial emotion is detected using a trained ResNet model.

The emotion is mapped to a suitable genre (e.g., Happy → Pop).

Songs are filtered by genre and release year.

KNN identifies songs with similar audio features (danceability, energy, valence, tempo).

Spotify URI is used to suggest tracks that match the user’s mood.

📈 Future Scope
Multimodal emotion detection (voice, body language)

Deep learning-based recommendation engine

Cross-platform support (Apple Music, YouTube Music)

Social sharing and group playlist features

Gamified mood tracking and analytics

Stronger privacy and ethical safeguards
