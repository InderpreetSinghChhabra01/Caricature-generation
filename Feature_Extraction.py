from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
from keras.models import Model
from PIL import Image                
import os.path
import sys, os
import numpy





#path1='/home/inderpreet/DeepLearning/Caricature/Dataset/FinalDistribute/Caricature_class_wise'
path='/home/inderpreet/DeepLearning/Caricature/Dataset/FinalDistribute/Caricature'

list_dataset = sorted(os.listdir(path))

i=0

for file in list_dataset:
	img_path = path + '/' + file
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	base_model = Model(input=model.input, output=model.get_layer('fc2').output)
	# obtain the outpur of fc2 layer
	fc2_features = base_model.predict(x)
	
	feature_matrix[i]=fc2_features
	
	i=i+1
		
	print "Feature vector dimensions: ",fc2_features.shape
        print i
        
numpy.save('/home/inderpreet/DeepLearning/MatData/caricature269_feature4096_matrix.npy',feature_matrix)
#features = model.predict(x)
#print "Feature vector dimensions: ",features.shape



    
