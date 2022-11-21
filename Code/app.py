from keras.models import load_model
import os 
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf


def classify_image(img, model):

    """
    Utility function to classify image using model
    """

    classes = {0: 'Cardboard Waste (Biodegradable)', 1: 'Food Waste (Biodegradable)', 2: 'Glass Waste (Non-Biodegradable)', 3:'Metal Waste (Non-Biodegradable)'}

    

    # # read the image
    # img1 = cv2.imread(img.name)
    # print(img1)

    # resize the image to the required dimensions
    resized_img = tf.image.resize(img, (256,256))
    
    # since the model inputs from a batch, so we convert a single image into a batch of single image
    resized_img = np.expand_dims(resized_img/255, 0)
    # predict the image using model
    yhat = model.predict(resized_img)
    return "Predicted class for the image is: {}".format(classes[np.argmax(yhat,axis=1)[0]])



def main():

    """
    Main function to create a streamlit application and classify images
    """

    st.set_option('deprecation.showfileUploaderEncoding',False)

    # create a title for the app
    st.title("Garbage Classifier")

    # style the interface
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white; text-align:center;">Streamlit Garbage Classifier App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # input the image file from the user
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    
    # and create a button for "Classify"
    class_btn = st.button("Classify")

    # Open the uploaded image
    if file_uploaded is not None:    
        # with open(file_uploaded.name,'wb') as f:
        #     f.write(file_uploaded.read())

        image = Image.open(file_uploaded)
        # display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
    predictions = ""
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):

                # Saves
                img = Image.open(file_uploaded)
                img = img.save("img.jpg")

                # OpenCv Read
                img = cv2.imread("img.jpg")
                predictions = classify_image(img, model)

    st.success(predictions)
                

model = load_model(os.path.join('Model', 'garbage_classifier.h5'))


if __name__ == "__main__":
    main()

