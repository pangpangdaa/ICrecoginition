from PIL import Image, ImageFont, ImageDraw, ImageFilter
import random
from vocab import Vocab
import numpy as np


class Captcha:
    '''
    size: width, height in pixel
    font: font family(string), size (unit pound) and font color (in "#rrggbb" format)
    bgcolor: in "#rrggbb" format
    '''
    def __init__(self, size, font, bgcolor, length = 4):
        #todo: add param check and transform here
        self.width, self.height = size
        self.font_family, self.font_size, self.font_color = font
        self.bgcolor = bgcolor
        self.len = length
        self.vocab = Vocab()
        self.font = ImageFont.truetype(self.font_family, self.font_size)
    def get_text(self):
        return self.vocab.rand_string(self.len)
    # by default, draw center align text
    def draw_text(self, str):
        dr = ImageDraw.Draw(self.im)
        font_width, font_height = self.font.getsize(str)
        # don't know why, but for center align, I should divide it by 2, other than 3
        dr.text(((self.width - font_width) / 3, (self.height - font_height) / 3), str, fill = self.font_color, font = self.font)
    def draw_background(self):
        pass
    def transform(self):
        params = [1 - float(random.randint(1, 2)) / 100,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 10)) / 100,
                  float(random.randint(1, 2)) / 500,
                  0.001,
                  float(random.randint(1, 2)) / 500
                  ]
        self.im = self.im.transform((self.width, self.height), Image.PERSPECTIVE, params)
    def filter(self):
        self.im.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # by default, add no noises
    def add_noise(self):
        pass
    def get_captcha(self):
        self.im = Image.new("RGB", (self.width, self.height), (self.bgcolor))
        self.draw_background()
        str = self.get_text()
        self.draw_text(str)
        self.add_noise()
        self.transform()
        self.filter()
        return self.im, str
    
    
class JrttCaptcha(Captcha):
    def __init__(self, size = (120, 30), font = ("DejaVuSerif.ttf", 20, "#0000ff"), bgcolor = (255, 255, 255), dot_rate = 0.05):
        Captcha.__init__(self, size, font, bgcolor)
        self.dot_rate = dot_rate
    
    def add_noise(self):
        # add lines
        nb_lines = random.randint(1, 2)
        dr = ImageDraw.Draw(self.im)
        for i in range(nb_lines):
            # 避免begin和end太靠近，导致生成的干扰线太短
            begin = (random.randint(0, self.width)/2, random.randint(0, self.height)/2)
            end = (random.randint(self.width / 2, self.width), random.randint(0, self.height))
            dr.line([begin, end], fill = (0, 0, 0))
        # add dots
        for w in range(self.width):
            for h in range(self.height):
                if random.randint(0, 100) / 100 <= self.dot_rate:
                    dr.point((w, h), fill = (0, 0, 0))
    def draw_text(self, str):
        display_text = [" "] * (len(str) * 2 - 1)
        for i in range(len(str)):
            display_text[i * 2] = str[i]
        super().draw_text(str)
        
        
if __name__ == "__main__":
    cap = JrttCaptcha()
    #每调用一次，生成一个<图像,文本>对。其中图象可看成是输入，而文本可以看成是真值
    img, text = cap.get_captcha()
       # img.save('./dataset/'+text+ ".jpg")
    print(text.shape)
