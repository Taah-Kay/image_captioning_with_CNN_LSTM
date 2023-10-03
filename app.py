import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
import tempfile
import keras.utils as image
import pickle
from keras.models import load_model
st.set_page_config(page_title="Image Captioning")
# Loading InceptionV3
model = InceptionV3(weights='imagenet')
# we do not need to classify the images here, we only need to extract an image vector for our images
model_new = Model(model.input, model.layers[-2].output)  


# loading index_to word, word to index and the model
with open('index_to_word.pickle', 'rb') as handle:
  ixtoword = pickle.load(handle)

with open('word_to_index.pickle', 'rb') as handle:
  wordtoix = pickle.load(handle)
 
 
model = load_model('model_img_caption.h5', compile = False)
max_lenth = 38



#function for pre-processing the image
def preprocess(image):
    img = image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

#function for encoding the image
def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

# Define the prediction function
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
        if not check_file_size(video_file, 2 * 1024 * 1024 ):  # 2MB limit
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
        for img in frames:
          x=img
          image = encode(img).reshape((1,2048))
          plt.imshow(x)
          frame = plt.show()
          caption = predict_caption(image)

            # Display the frame and the predicted caption
    st.image(frame, use_column_width=True)
    st.write(f"Caption {i+1}: {caption}")      




# Process each frame and predict captions



            
            

# Run the Streamlit app
if __name__ == "__main__":
    main()
