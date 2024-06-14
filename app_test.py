import streamlit as st
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
import io
import numpy as np
from skimage.transform import resize
from deep_translator import GoogleTranslator
from gtts import gTTS

st.title("Braille To Text & Voice Generation")

uploaded_image = st.file_uploader(f"Upload an image of", type=["jpg", "jpeg", "png"])
size = (28, 28)

if uploaded_image is not None:
    img_filename = uploaded_image.name

    braille = Image.open(io.BytesIO(uploaded_image.read()))
    braille = np.array(braille)
    img_reshape = braille / 255.0
    st.image(braille)
    
    model = tf.keras.models.load_model('./braille_letter_classification_model.h5')

    csv_filename = os.path.splitext(img_filename)[0] + '.csv'
    csv_folder = './'  # Path to the folder containing all CSV files
    csv_path = os.path.join(csv_folder, csv_filename)
    
    bb=[]
    if os.path.exists(csv_path):
        # Read the CSV file if it exists
        df = pd.read_csv(csv_path, delimiter=';', header=None, names=['left', 'top', 'right', 'bottom'])

        # Extract bounding boxes
        bb = df[['left', 'top', 'right', 'bottom']].values
        bb = [[int(value) for value in bbox] for bbox in bb]

    else:
        error = "We have encountered an error reading this image !!"
        st.markdown(f"<div style='border: 1px solid grey; padding: 10px; margin-bottom: 10px; font-size: 28px;'>{error}</div>", unsafe_allow_html=True)

    def preprocess(character_image):
        resized_character = resize(character_image, (28, 28, 3))
        normalized_character_with_batch = np.expand_dims(resized_character, axis=0)
        return normalized_character_with_batch

    def post_process(predictions):
        character_labels = [chr(np.argmax(pred) + ord('A')) for pred in predictions]
        final_text = ''.join(character_labels)
        return final_text

    predictions = []

    for bbox in bb:
        x_min, y_min, x_max, y_max = bbox
        character_image = braille[y_min:y_max, x_min:x_max]
        processed_character = preprocess(character_image)
        prediction = model.predict(processed_character)
        predictions.append(prediction)

    final_text = post_process(predictions)

    def calculate_center(bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def calculate_distance(center1, center2):
        x1, y1 = center1
        x2, y2 = center2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def find_dist(bounding_boxes):
        centers = [calculate_center(bbox) for bbox in bounding_boxes]
        distances = [calculate_distance(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
        return distances

    distance = find_dist(bb)

    def detect_spaces(characters, threshold):
        sp = []
        prev_x = None

        for i in range(len(characters)):
            x = characters[i]

            if prev_x is not None and x - prev_x > threshold:
                sp.append(i)
            prev_x = x

        return sp

    spaces = detect_spaces(distance, 40) # change from 50 -> 40

    def add_spaces(input_string, spaces_positions):
        input_list = list(input_string)
        for pos in reversed(spaces):
            input_list.insert(pos + 1, ' ')
        output_string = ''.join(input_list)
        return output_string

    output_string = add_spaces(final_text, spaces)
    st.title("Output:")
    st.markdown(f"<div style='border: 1px solid grey; padding: 10px; margin-bottom: 10px; font-size: 28px;'>{output_string}</div>", unsafe_allow_html=True)
    
    # Translation
    languages = {
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Chinese': 'zh-CN',
        'Hindi': 'hi',
        'Bengali': 'bn',
        'Marathi': 'mr',
        'Tamil': 'ta',
        'Telugu': 'te',
        'Gujarati': 'gu',
        'Kannada': 'kn',
        'Malayalam': 'ml',
        'Punjabi': 'pa',
        'Urdu': 'ur'
    }
    target_language = st.selectbox("Select language for translation:", list(languages.keys()))
    translated_text = GoogleTranslator(source='auto', target=languages[target_language]).translate(output_string)
    st.title("Translated Output:")
    st.markdown(f"<div style='border: 1px solid grey; padding: 10px; margin-bottom: 10px; font-size: 28px;'>{translated_text}</div>", unsafe_allow_html=True)

    # Function to convert text to speech
    def text_to_speech(text, language_code):
        tts = gTTS(text=text, lang=language_code)
        tts.save('output_audio.mp3')

    # Display sound icon
    if st.button('🔊 Play Audio'):
        text_to_speech(translated_text, languages[target_language])
        audio_file = open('output_audio.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')


