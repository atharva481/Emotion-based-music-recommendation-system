import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
import cv2
import numpy as np
import tensorflow
import time
from tensorflow import keras
from keras.models import load_model

st.set_page_config(page_title="Emotion Based Music Recommendation Engine", layout="wide")

model_path = "facialemotionmodel.h5"
emotion_model = load_model(model_path)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(face, model):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    prediction = model.predict(face)
    return emotions[np.argmax(prediction)]

@st.cache_resource
def load_data():
    df = pd.read_csv("filtered_track_df.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

emotion_presets = {
    'Angry': {'acousticness': 0.2, 'danceability': 0.6, 'energy': 0.8, 'instrumentalness': 0.1, 'valence': 0.3, 'tempo': 140.0},
    'Disgust': {'acousticness': 0.5, 'danceability': 0.3, 'energy': 0.4, 'instrumentalness': 0.2, 'valence': 0.2, 'tempo': 100.0},
    'Fear': {'acousticness': 0.7, 'danceability': 0.3, 'energy': 0.2, 'instrumentalness': 0.5, 'valence': 0.1, 'tempo': 90.0},
    'Happy': {'acousticness': 0.4, 'danceability': 0.8, 'energy': 0.7, 'instrumentalness': 0.1, 'valence': 0.9, 'tempo': 120.0},
    'Sad': {'acousticness': 0.8, 'danceability': 0.2, 'energy': 0.2, 'instrumentalness': 0.3, 'valence': 0.1, 'tempo': 80.0},
    'Surprise': {'acousticness': 0.4, 'danceability': 0.7, 'energy': 0.6, 'instrumentalness': 0.2, 'valence': 0.7, 'tempo': 130.0},
    'Neutral': {'acousticness': 0.5, 'danceability': 0.5, 'energy': 0.5, 'instrumentalness': 0.2, 'valence': 0.5, 'tempo': 110.0},
}

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios

title = "Emotion Based Music Recommendation System"
st.title(title)

st.write("This project recommends emotion-based music by matching detected moods with pre-curated playlists, using machine learning.")
st.markdown("##")

# Add this after the emotion_presets dictionary
emotion_genre_mapping = {
    'Angry': 'Rock',
    'Disgust': 'Jazz',
    'Fear': 'Electronic',
    'Happy': 'Pop',
    'Sad': 'R&B',
    'Surprise': 'Dance Pop',
    'Neutral': 'Pop'
}

# Initialize session state for emotion and genre
if 'selected_emotion' not in st.session_state:
    st.session_state.selected_emotion = 'Neutral'
if 'genre' not in st.session_state:
    st.session_state.genre = 'Pop'

with st.container():
    col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col4:
        st.markdown("***Choose an emotion:***")
        selected_emotion = st.selectbox('', 
            list(emotion_presets.keys()) + ["Start Camera"], 
            index=list(emotion_presets.keys()).index(st.session_state.selected_emotion))
        st.session_state.selected_emotion = selected_emotion
        
        if selected_emotion == "Start Camera":
            placeholder = st.empty()
            cap = cv2.VideoCapture(0)
            detected_emotion = None
            start_time = time.time()
            countdown = 6  # 3 seconds countdown
        
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
        
                current_time = time.time()
                time_left = countdown - int(current_time - start_time)
                
                if time_left > 0:
                    # Show countdown
                    cv2.putText(frame, str(time_left), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                elif time_left == 0:
                    # Capture and process frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        detected_emotion = detect_emotion(face, emotion_model)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        break
        
                placeholder.image(frame, channels="BGR", caption="Detecting Emotion")
        
                if detected_emotion or time_left < -1:  # Exit after detection or 1 second after countdown
                    break
        
            cap.release()
            st.write(f"Detected Emotion: **{detected_emotion}**")
            if detected_emotion in emotion_presets:
                st.session_state.selected_emotion = detected_emotion
                st.session_state.genre = emotion_genre_mapping[detected_emotion]
                selected_emotion = detected_emotion

    with col3:
        st.markdown("***Choose your genre:***")
        default_genre = emotion_genre_mapping[st.session_state.selected_emotion]
        st.session_state.genre = st.radio("", genre_names, index=genre_names.index(default_genre))

    if selected_emotion:
        acousticness = emotion_presets[selected_emotion]['acousticness']
        danceability = emotion_presets[selected_emotion]['danceability']
        energy = emotion_presets[selected_emotion]['energy']
        instrumentalness = emotion_presets[selected_emotion]['instrumentalness']
        valence = emotion_presets[selected_emotion]['valence']
        tempo = emotion_presets[selected_emotion]['tempo']
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider('Select the year range', 1990, 2019, (2015, 2019))
        acousticness = st.slider('Acousticness', 0.0, 1.0, acousticness)
        danceability = st.slider('Danceability', 0.0, 1.0, danceability)
        energy = st.slider('Energy', 0.0, 1.0, energy)
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, instrumentalness)
        valence = st.slider('Valence', 0.0, 1.0, valence)
        tempo = st.slider('Tempo', 0.0, 244.0, tempo)

test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
uris, audios = n_neighbors_uri_audio(st.session_state.genre, start_year, end_year, test_feat)
tracks_per_page = 6
tracks = []
for uri in uris:
    track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
    tracks.append(track)

if 'previous_inputs' not in st.session_state:
    st.session_state['previous_inputs'] = [st.session_state.genre, start_year, end_year] + test_feat

current_inputs = [st.session_state.genre, start_year, end_year] + test_feat
if current_inputs != st.session_state['previous_inputs']:
    if 'start_track_i' in st.session_state:
        st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

if 'start_track_i' not in st.session_state:
    st.session_state['start_track_i'] = 0

with st.container():
    col1, col2, col3 = st.columns([2,1,2])
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page
    current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    if st.session_state['start_track_i'] < len(tracks):
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            if i%2==0:
                with col1:
                    components.html(track, height=400)
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(r=audio[:5], theta=audio_feats[:5]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)
            else:
                with col3:
                    components.html(track, height=400)
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(r=audio[:5], theta=audio_feats[:5]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)
    else:
        st.write("No songs left to recommend")
