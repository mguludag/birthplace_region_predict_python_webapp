import os
import numpy as np
from PIL import Image
import h5py
import cv2

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
# Save to disk
f = h5py.File("images.h5", "w")
# Create dataset to store images
# X_dset = f.create_dataset('data', images.shape, dtype='f')
# X_dset[:] = images
X_dset = f.create_dataset('dataResized', imagesResized.shape, dtype='f')
X_dset[:] = imagesResized

# Create dataset to store labels
y_dset = f.create_dataset('region', region.shape, dtype='i')
y_dset[:] = region

    
f.close()
