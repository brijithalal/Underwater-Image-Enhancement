#from __future__ import absolute_import
#from __future__ import division
import streamlit as st 
from PIL import Image
import numpy as np
import os
import time
import ntpath
import random
import fnmatch
from os.path import join, exists
from keras.models import model_from_json
def deprocess(x, np_uint8=True):
    # [-1,1] -> [0, 255]
    x = (x+1.0)*127.5
    return np.uint8(x) if np_uint8 else x

def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x/127.5)-1.0

def getPaths(data_dir):
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)

def read_and_resize(path, img_res):
    im = Image.open(path).resize(img_res)
    if im.mode=='L': 
        copy = np.zeros((res[1], res[0], 3))
        copy[:, :, 0] = im
        copy[:, :, 1] = im
        copy[:, :, 2] = im
        im = copy
    return np.array(im).astype(np.float32)


samples_dir = "/content/drive/MyDrive/image/FUnIE-GAN-master/data/output/"
checkpoint_dir  = '/content/drive/MyDrive/image/FUnIE-GAN-master/TF-Keras/models/gen_p/'
model_name_by_epoch = "model_15320_"
model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (exists(model_h5) and exists(model_json))
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
# load weights into new model
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")
def predict(img):
        gen = funie_gan_generator.predict(im)
        gen_img = deprocess(gen)[0]
        return (Image.fromarray(gen_img))


st.title("Underwater Image Enhancement ")
st.subheader("Upload an image which you want to enhance")   
st.spinner("Testing spinner")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    image = read_and_resize(uploaded_file, (256, 256))
    im = preprocess(image)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)
    st.image(img, caption='Uploaded Image.')
    st.write("")
    if st.button('Enhance Now'):
        st.write("enhancing...") 
        pred = predict(im)
        st.image(pred, caption='Enhanced Image')        
