

##  Emotion-Based Music Recommendation System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0%2B-red)

An intelligent music recommendation engine that bridges the gap between computer vision and audio analytics. By leveraging **ResNet** for facial emotion recognition and **K-Nearest Neighbors (KNN)** for content-based filtering, this system suggests the perfect Spotify tracks tailored to your current mood.

---

## 🚀 Overview
Music is deeply tied to emotion. This project automates the "vibe check" by detecting a user's facial expression via webcam and cross-referencing it with a multi-dimensional Spotify dataset. 

The system analyzes musical features such as danceability, energy, and valence to find the nearest mathematical neighbors to your emotional state.



## ✨ Key Features
*   **Real-time Emotion Detection:** Uses a Deep Learning **ResNet** architecture to classify facial expressions accurately.
*   **Smart Recommendations:** Implements **KNN** (K-Nearest Neighbors) to find songs with similar acoustic profiles.
*   **Extensive Catalog:** Powered by a merged Spotify dataset containing detailed information on Artists, Albums, and Tracks.
*   **Interactive UI:** Built with **Streamlit** for a seamless, browser-based user experience.

## 🛠️ Technologies Used
*   **Deep Learning:** ResNet (Residual Networks) for image classification.
*   **Machine Learning:** Scikit-learn (K-Nearest Neighbors).
*   **Data Manipulation:** Pandas, NumPy.
*   **Frontend:** Streamlit.
*   **API/Data:** Spotify Music Dataset.

---

## 📥 Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/emotion-music-recommendation.git
    cd emotion-music-recommendation
    ```

2.  **Set Up Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
2.  **Grant Camera Access:** When prompted, allow the browser to access your webcam.
3.  **Get Recommendations:** The system will capture your expression and instantly display a curated list of tracks from the `filter_track_df` dataset.

---

## 📊 Dataset Architecture
The engine processes and merges three primary data sources to create a high-fidelity feature set:
*   **Spotify Artists:** Metadata regarding genres and popularity.
*   **Spotify Albums:** Release dates and album-level features.
*   **Spotify Tracks:** Granular audio features (Tempo, Acousticness, Liveness, etc.).



---


## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
