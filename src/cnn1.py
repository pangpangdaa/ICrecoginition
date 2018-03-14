from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, AveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from generator import JrttCaptchaGenerator
from captcha import JrttCaptcha
import numpy as np
from vocab import Vocab
from keras.optimizers import SGD
batch_size = 32
ocr_shape = (30, 120, 3) # height, width, channels
nb_classes = 62
inputs = Input(shape = ocr_shape, name = "inputs")
conv1 = Convolution2D(32, 3, 3, name = "conv1")(inputs)
bn1=BatchNormalization()(conv1)
relu1 = Activation('relu', name="relu1")(bn1)
conv2 = Convolution2D(32, 3, 3, name = "conv2")(relu1)
bn2=BatchNormalization()(conv2)
relu2 = Activation('relu', name="relu2")(bn2)

pool1 = MaxPooling2D(pool_size=(2,2), border_mode='same', name="pool1")(relu2)
conv3 = Convolution2D(64, 3, 3, name = "conv3")(pool1)
bn3=BatchNormalization()(conv3)
relu3 = Activation('relu', name="relu3")(bn3)

conv4 = Convolution2D(64, 3, 3, name = "conv4")(relu3)
bn4=BatchNormalization()(conv4)
relu4=Activation('relu',name='relu4')(bn4)
pool2 = AveragePooling2D(pool_size=(2,2), name="pool2")(relu4)
#drop = Dropout(0.25, name = "dropout1")(fc1)
fl = Flatten()(relu4)


fc21= Dense(128, name="fc21", activation="relu")(fl)
fc22= Dense(128, name="fc22", activation="relu")(fl)
fc23= Dense(128, name="fc23", activation="relu")(fl)
fc24= Dense(128, name="fc24", activation="relu")(fl)
out1=Dense(nb_classes,name='out1',activation='softmax')(fc21)
out2=Dense(nb_classes,name='out2',activation='softmax')(fc22)
out3=Dense(nb_classes,name='out3',activation='softmax')(fc23)
out4=Dense(nb_classes,name='out4',activation='softmax')(fc24)

model = Model(input = inputs, output = [out1,out2,out3,out4])
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
metrics=['accuracy'])
model.summary()
model.fit_generator(JrttCaptchaGenerator(batch_size, "/dataset/"), 32000, 10)

for i in range(100):
    img, text = JrttCaptcha().get_captcha()
    X = np.empty((1, 30, 120, 3))
    X[0] = np.array(img, dtype = np.uint8) / 255
    Y_pred = model.predict(X, 1, 1)
    Pred_text = Vocab().one_hot_to_text(Y_pred[0])
    if Pred_text != text:
        print("True value is ", text)
        print("Prediceted value is ", Pred_text)