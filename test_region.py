from face_network import create_face_network
import cv2
import numpy as np
from keras.optimizers import Adam, SGD
import streamlit as st
from PIL import Image, ImageOps

region = {0: 'akdeniz', 1: 'ege', 2: "ic_anadolu", 3: "karadeniz"}

def predict_region(im):
	means = np.load('means_region.npy')

	model = create_face_network(nb_class=4, hidden_dim=256, shape=(224, 224, 3))
	model.load_weights('weights_region.hdf5')

	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
	im = cv2.resize(im, (224, 224))
	# cv2.imshow("fsfs",im)
	# cv2.waitKey(0)
	im = np.float64(im)
	im /= 255.0
	im = im - means

	return model.predict(np.array([im]))


if __name__ == "__main__":

	st.write("""  
         # birthplace regions prediction  
         """  
         )
	st.write("This is a simple image classification web app to predict region of birthplace from image")
	

	file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

	if file is None:
		st.text("You haven't uploaded an image file")
	else:
		image = Image.open(file)
		st.image(image, use_column_width=True)
		im = np.array(image)
		result = predict_region(im)
		st.write(region[np.argmax(result)])
		
		st.text("Probability (0: akdeniz, 1: ege, 2: ic_anadolu, 3: karadeniz)")
		st.write(result)

	
	custom_footer = """<myfooter style={bottom:0;position:fixed;background-color:white;}>Made by <a href="//github.com/mguludag" target="_blank">mguludag</a></myfooter>"""
	hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """

	st.markdown(hide_footer_style, unsafe_allow_html=True)
	st.markdown(custom_footer, unsafe_allow_html=True)
