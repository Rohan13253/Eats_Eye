import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = Image.open(test_image)
    image = image.resize((64, 64))  # Resize image to match model's input shape
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array(input_arr)
    predictions = model.predict(input_arr.reshape(1, 64, 64, 3))
    return np.argmax(predictions)

st.markdown(
    """
    <style>
    body {
        background-image: url('istockphoto-1319299044-170667a.webp');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ("Home", "About Project", "Prediction"))

if app_mode == "Home":
    st.header("Fruits & Vegetable Recognition System")

elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset") 
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant")
    st.subheader("content")

elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image)
    if st.button("Predict") and test_image is not None:
        predicted_class = model_prediction(test_image)
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i.strip())
        st.write("It's a :", label[predicted_class])
        image = Image.open(test_image)
        st.image(image, caption=f"Its a : {label[predicted_class]}", use_column_width=True)
 



