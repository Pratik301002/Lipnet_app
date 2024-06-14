import streamlit as st
import os
from pathlib import Path
from moviepy.editor import VideoFileClip
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

def convert_mpg_to_mp4(input_file, output_file):
    try:
        # Load the .mpg file
        clip = VideoFileClip(input_file)

        # Write the result to a .mp4 file
        clip.write_videofile(output_file, codec='libx264')
        return output_file

    except Exception as e:
        st.error(f"Error during conversion: {e}")
        return None

st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Generating a list of options or videos
options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox("Choose video", options)
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data', 's1', selected_video)
        output_path = os.path.join("converted_video.mp4")  # Use raw string

        converted_file = convert_mpg_to_mp4(file_path, output_path)

        if converted_file:
            # Rendering inside of the app
            video = open(output_path, 'rb')
            video_bytes = video.read()
            st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor("/content/data/s1/bbaf4p.mpg"))
        st.write(annotations)
        imageio.mimsave("animation.gif",video,fps=10)
        st.image("animation.gif",width = 500)

        st.info('This is the ouput of our machine learning model as tokens')

        model = load_model()

        y_hat = model.predict(tf.expand_dims(video,axis=0))
        decoder = tf.keras.backend.ctc_decode(y_hat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
