from keras.preprocessing import image
import numpy as np


image_path='captcha1'

def readImg(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x
    
    
print(readImg(image_path='captcha1.jpg'))