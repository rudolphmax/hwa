# Loading necessary packages and models
import pickle
import streamlit as st
@st.experimental_singleton
def initial():
    import cv2
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import svm
    import plotly.express as px
    from feature_extractor import Image_fe
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    return cv2, np, pd, svm, px, Image_fe, StandardScaler, train_test_split
cv2, np, pd, svm, px, Image_fe, StandardScaler, train_test_split = initial()

# WebApp
st.subheader('Welcome to HWA')
with st.expander('Additional Information 📝'):
    st.text('1.Please make sure the image is either scanned or captured by a steady hand.')
    st.text('2.Take the picture in good lighting setup')
#     st.text('3.Avoid using flash while taking the picture')
st.session_state['features'] = False
st.select_slider(label = 'Photo Options', options = ['Upload', 'Camera'], key = 'upload')
if st.session_state['upload'] == 'Upload':
    img_file_buffer = st.file_uploader('Upload a picture')
else:
    img_file_buffer = st.camera_input('Take a picture')
labels = ['good', 'bad']
if img_file_buffer:
    if st.session_state['upload'] == 'Upload':
        st.image(img_file_buffer)
    img = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_features = Image_fe(img_name = gray, local = False)
    if not st.session_state['features']:
        st.session_state['features'] = img_features.word_fe()
    display_options = ['Slant Variance', 'Spacing', 'Height Uniformity', 'Area Uniformity']
    imgs = [img_features.slant_img, img_features.space_img, img_features.height_img, img_features.area_img]
    tabs = st.tabs(display_options)
    for i, tab in enumerate(tabs):
        feature, size = display_options[i], 720
        tab.subheader(feature)
        # size = tab.slider('Adjust Plot Size ', min_value = 100, max_value = 1500, value = 500, step = 50, key = feature.split(' ')[0] + '_plotsize')
        tab.plotly_chart(px.imshow(imgs[i], color_continuous_scale = 'magma' if i != 1 else 'gray', height = size, width = size))

    model = joblib.load("model1.pkl")
    scaler = joblib.load("scaler1.joblib")
    prediction = model.predict(scaler.transform(st.session_state['features'].to_numpy().reshape(1, -1)))
#     st.write(prediction)
    if prediction[0]:
        st.success('Your Handwriting is quite good. 👌')
    else:
        st.error('Your Handwriting is bad. 🙅‍♂️')
else:
    st.session_state['features'] = False
