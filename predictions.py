import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tempfile
import pickle
st.set_page_config(page_title="Image Captioning")
# Load the pre-trained ResNet-50 model
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Define the prediction function
def predict_caption(photo):
    in_text = "startseq"
    max_len = 29

    # Load word_to_idx dictionary from file
    with open("wordToIndex[1].pkl", "rb") as f:
        word_to_idx = pickle.load(f)

    # Load idx_to_word dictionary from file
    with open("indexToWord[1].pkl", "rb") as f:
        idx_to_word = pickle.load(f)

    # Placeholder for the image captioning model
    model = load_model("final_mod.h5")

    for _ in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        if ypred not in idx_to_word:
            break
        word = idx_to_word[ypred]
        in_text += ' ' + word

        if word == 'endseq':
            break

    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption

# Function to check file size
def check_file_size(file, size_limit):
    file.seek(0, 2)  # Move the file pointer to the end
    file_size = file.tell()
    file.seek(0)  # Reset the file pointer to the beginning
    return file_size <= size_limit

# Streamlit app
def main():
    st.title("Image Captioning App")

    st.markdown("## Upload a video file")
    st.markdown("Limit: 2MB per file • MP4")

    # Upload video file
    video_file = st.file_uploader("Drag and drop file here", type=["mp4"])
    if video_file is not None:
        if not check_file_size(video_file, 2 * 1024 * 1024):  # 2MB limit
            st.error("File size exceeds the limit of 2MB.")
            return

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        # Convert video frames to images
        frames = []
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        while success:
            frames.append(image)
            success, image = vidcap.read()

        # Process each frame and predict captions
        st.write(f"Number of frames: {len(frames)}")
        for i, frame in enumerate(frames):
            # Resize the frame to the input size of the ResNet-50 model
            frame = cv2.resize(frame, (224, 224))

            # Preprocess the image
            img = preprocess_input(frame)

            # Pass the image through the ResNet-50 model
            img_features = resnet_model.predict(np.expand_dims(img, axis=0))

            # Get the predicted caption
            caption = predict_caption(img_features)

            # Display the frame and the predicted caption
            st.image(frame, use_column_width=True)
            st.write(f"Caption for Frame {i+1}: {caption}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
