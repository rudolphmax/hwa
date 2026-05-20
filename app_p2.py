# Loading necessary packages and models
import streamlit as st
# @st.cache_resource
def initial():
    import cv2
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import svm
    from hwa1 import Image_fe
    import plotly.express as px
    from skimage.color import label2rgb
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    return cv2, np, pd, svm, px, Image_fe, MinMaxScaler, train_test_split, label2rgb
cv2, np, pd, svm, px, Image_fe, MinMaxScaler, train_test_split, label2rgb = initial()

# WebApp
st.subheader('Welcome to HWA')
with st.expander('Additional Information 📝'):
    st.text('1.Please make sure the image is either scanned or captured by a steady hand.')
    st.text('2.Take the picture in good lighting setup')
st.session_state['features'] = False
ruled = st.checkbox('Check this box if the uploaded document contains ruled lines?')
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
    img_features = Image_fe(img_name = gray, local = False, ruled = ruled)
#     if not st.session_state['features']:
    st.session_state['features'] = img_features.word_fe()
    display_options = ['Word Segmentation', 'Slant Variance', 'Height Uniformity', 'Area Uniformity']
    imgs = [img_features.slant_img, img_features.height_img, img_features.area_img]
    tabs = st.tabs(display_options)
    for i, tab in enumerate(tabs):
        feature = display_options[i]
        tab.subheader(feature)
        if i:
            if i == 4:
                tab.image(img_features.img)
            else:
                fig = px.imshow(
                    imgs[i-1],
                    color_continuous_scale = 'magma',
                    template = 'plotly_dark',
                    height = gray.shape[0]//2.4,
                    width = gray.shape[1]//2.4
                )
                tab.image(fig.to_image('png'))
        else:
            tab.image(label2rgb(img_features.label))

    model = joblib.load("model2.pkl")
    scaler = joblib.load("scaler2.joblib")
    prediction = model.predict(scaler.transform(st.session_state['features'].to_numpy().reshape(1, -1)))
    if prediction[0]:
        st.success('Your Handwriting is quite good. 👌')
    else:
        st.error('Your Handwriting is bad. 🙅‍♂️')
else:
    st.session_state['features'] = False
