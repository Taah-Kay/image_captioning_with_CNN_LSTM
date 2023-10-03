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
# loading index_to word, word to index and the model
with open('index_to_word.pickle', 'rb') as handle:
  ixtoword = pickle.load(handle)

with open('word_to_index.pickle', 'rb') as handle:
  wordtoix = pickle.load(handle)
 
 
model = load_model('model_img_caption.h5')
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1
def predict_caption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

# Function to check file size
def check_file_size(file, size_limit):
    file.seek(0, 2)  # Move the file pointer to the end
    file_size = file.tell()
    file.seek(0)  # Reset the file pointer to the beginning
    return file_size <= size_limit

# Streamlit app
def main():
    st.title("Image_captioning_with_CNN_LSTM")

    st.markdown("## Upload your video")
    st.markdown("Limit:Your video should not exceed 2MB • MP4")

    # Upload video file
    video_file = st.file_uploader("Drag and drop your video here", type=["mp4"])
    if video_file is not None:
        if not check_file_size(video_file, 2 * 1024 ):  # 2MB limit
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
            st.write(f"Caption {i+1}: {caption}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
