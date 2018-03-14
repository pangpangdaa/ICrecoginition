from vocab import Vocab
from captcha import Captcha,JrttCaptcha
import numpy as np



def JrttCaptchaGenerator(batch_size, path):
    # to determine dimensions
    cap = JrttCaptcha()
    img, text = cap.get_captcha()
    shape = np.asarray(img).shape
    vocab = Vocab()
    while (1):
        X = np.empty((batch_size, shape[0], shape[1], shape[2]))
        Y1 = np.empty((batch_size, vocab.size))
        Y2 = np.empty((batch_size, vocab.size))
        Y3 = np.empty((batch_size, vocab.size))
        Y4 = np.empty((batch_size, vocab.size))
        
        for j in range(batch_size):
            img, text = cap.get_captcha()
            #img.save(path + text + ".jpg")
            X[j] = np.array(img) / 255
            Y = vocab.text_to_four_hot(text)
            Y1[j]=Y[0,:]
            Y2[j]=Y[1,:]
            Y3[j]=Y[2,:]
            Y4[j]=Y[3,:]

        yield X, [Y1,Y2,Y3,Y4]