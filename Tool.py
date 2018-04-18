import tensorflow as tf
import numpy as np
from PIL import Image

class Tool:
    def load_image(image_path):
        # load image
        image_SIZE = 224 # image size
        img = Image.open(image_path)
        width, height = img.size
        if(width > image_SIZE and height > image_SIZE):
            tmp_max = max(width, height)
            tmp_min = min(width, height)
            x = int((tmp_max - width)/2)
            y = int((tmp_max - height)/2)
            resized_img = img.crop((x, y, tmp_min+x, tmp_min+y))
        resized_img = img.resize((image_SIZE, image_SIZE))
        return resized_img














        # with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     with tf.device("/gpu:0"):
#         img_data = tf.image.decode_jpeg(image_raw_data)
#         image_arr=tf.image.convert_image_dtype(img_data,tf.float32)
#         image_featuremap = tf.reshape(image_arr,[-1,image_SIZE,image_SIZE,3])
#         layer_data1 = cov_layer1(image_featuremap,  64)
#         print (layer_data1)
#         layer_data2 = conv_layer2(layer_data1, 128)
#         print (layer_data2)
#         layer_data3 = conv_layer3(layer_data2, 256)
#         print (layer_data3)
#         layer_data4 = conv_layer4(layer_data3, 512)
#         print (layer_data4)
#         layer_data5 = conv_layer5(layer_data4, 512)
#         print (layer_data5)
#         layer_data6 =fc_layer(layer_data5, 4096)
#         print (layer_data6)
#         sm_data = tf.nn.softmax(layer_data6)
#         print (sm_data)