from keras.optimizers import SGD
# import h5py
import cv2
from face_network import create_face_network
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

train_split = 0.7

# Folder path
PATH = "D:\\Users\\mguludag\\Desktop\\staj_proj\\bolgeler"
FILE_FORMAT = (".png", ".jpg")

# Get first three digits
def getImageId(name):
	return name

images = []
imagesResized = []
region = []

for subdir, dirs, files in os.walk(PATH):
	for file in files:
		if file.endswith(FILE_FORMAT):
			name = os.path.join(subdir, file)
			im = cv2.imread(name, cv2.IMREAD_COLOR)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
			
			# im.show()
			images.append(np.array(im))

			im = cv2.resize(im, (224, 224))
			imagesResized.append(np.array(im))

			imageId = getImageId(os.path.basename(subdir))
			if(imageId=="akdeniz"):
				region.append(0)
			if(imageId=="ege"):
				region.append(1)
			if(imageId=="ic_anadolu"):
				region.append(2)
			if(imageId=="karadeniz"):
				region.append(3)
# cv2.imshow("sfsf",im)
# cv2.waitKey(0)



# Concatenate
# images = np.float64(np.stack(images))
# print(images.shape)
imagesResized = np.float64(np.stack(imagesResized))
region = np.stack(region)

			
	
# Normalize data
# images /= 255.0
imagesResized /= 255.0

# f = h5py.File('images.h5', 'r') 
X_data = imagesResized
y_data = region

#One-hot
y_data = to_categorical(y_data, 4)

# Split into training and validation sets
num_images = len(y_data)
p = np.random.permutation(num_images)
X_data = X_data[p]
y_data = y_data[p]


X_train = X_data[0:int(round(train_split*num_images))]
y_train = y_data[0:int(round(train_split*num_images))]
X_test = X_data[int(round(train_split*num_images))+1:-1]
y_test = y_data[int(round(train_split*num_images))+1:-1]
# Zero center
means = np.mean(X_train, axis = 0)
X_train -= means
X_test -= means
# Save means (for testing)
np.save('means_region.npy',means)

opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
checkpoint = ModelCheckpoint('weights_region.hdf5', monitor='val_acc', verbose=1, save_best_only=False,
								 save_weights_only=True, mode='max')
model = create_face_network(nb_class=4, hidden_dim=256, shape=(224, 224, 3))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train,
	batch_size=32,
	epochs=10,
	verbose=1,
	callbacks=[checkpoint],
	validation_data=(X_test, y_test),
	shuffle=True,
	class_weight=None,
	sample_weight=None,
	initial_epoch=0)
