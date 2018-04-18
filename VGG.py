import tensorflow as tf
import numpy as np
from PIL import Image
image_SIZE = 224 # image size
image_PATH = "/home/linux/Desktop/test1/11.jpg"
image_raw_data = tf.gfile.FastGFile("/home/linux/Desktop/test11.jpg", 'rb').read()
# _inputRGB = tf.placeholder(tf.float32,    #_inputRGB  initial
#                               [None,
#                               224,224,3])
def load_image(path):
    # load image
    img = Image.open(path)
    width, height = img.size
    if(width > image_SIZE and height > image_SIZE):
        tmp_max = max(width, height)
        tmp_min = min(width, height)
        x = int((tmp_max - width)/2)
        y = int((tmp_max - height)/2)
        resized_img = img.crop((x, y, tmp_min+x, tmp_min+y))
    resized_img = img.resize((image_SIZE, image_SIZE))
    return resized_img
def weight_variable(shape):
    initial = tf.random_normal(shape, seed = 0.5)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape =shape)
    return tf.Variable(initial)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
#first layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x , ksize = (1 , 2 ,2 , 1),
    strides = [1 ,2 ,2 , 1], padding = 'SAME' )
def cov_layer1(x, kernelmap_number):
    W_conv1 = weight_variable([3,3, 3, kernelmap_number])
    b_conv1 = bias_variable([kernelmap_number])
    h_conv1 = tf.nn.relu( conv2d(x, W_conv1) + b_conv1)
    # second layer
    W_conv1_1 = weight_variable([3,3, 64, kernelmap_number])
    b_conv1_1 = bias_variable([kernelmap_number])
    h_conv1_1 = tf.nn.relu( conv2d(h_conv1, W_conv1_1) + b_conv1_1)
    #maxpool
    h_pool1 = max_pool_2x2(h_conv1_1)
    return h_pool1
def conv_layer2(x, kernelmap_number):
    # 64x128
    W_conv2 = weight_variable([3,3, 64, kernelmap_number])
    b_conv2= bias_variable([kernelmap_number])
    h_conv2 = tf.nn.relu( conv2d(x, W_conv2) + b_conv2)
    # second layer
    W_conv2_1 = weight_variable([3,3, 128, kernelmap_number])
    b_conv2_1= bias_variable([kernelmap_number])
    h_conv2_1 = tf.nn.relu( conv2d(h_conv2, W_conv2_1) + b_conv2_1)
    #maxpool
    h_pool2 = max_pool_2x2(h_conv2_1)
    return h_pool2
def conv_layer3(x, kernelmap_number):
    # 64x128
    W_conv3 = weight_variable([3,3, 128, kernelmap_number])
    b_conv3= bias_variable([kernelmap_number])
    h_conv3 = tf.nn.relu( conv2d(x, W_conv3) + b_conv3)
    # second layer
    W_conv3_1 = weight_variable([3,3, 256, kernelmap_number])
    b_conv3_1= bias_variable([kernelmap_number])
    h_conv3_1 = tf.nn.relu( conv2d(h_conv3, W_conv3_1) + b_conv3_1)
    # third layer
    W_conv3_2 = weight_variable([3,3, 256, kernelmap_number])
    b_conv3_2= bias_variable([kernelmap_number])
    h_conv3_2 = tf.nn.relu( conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)
    #maxpool
    h_pool3 = max_pool_2x2(h_conv3_2)
    return h_pool3
def conv_layer4(x, kernelmap_number):
    # 64x128
    W_conv4 = weight_variable([3,3, 256, kernelmap_number])
    b_conv4= bias_variable([kernelmap_number])
    h_conv4 = tf.nn.relu( conv2d(x, W_conv4) + b_conv4)
    # second layer
    W_conv4_1 = weight_variable([3,3, 512, kernelmap_number])
    b_conv4_1= bias_variable([kernelmap_number])
    h_conv4_1 = tf.nn.relu( conv2d(h_conv4, W_conv4_1) + b_conv4_1)
    #third layer
    W_conv4_2 = weight_variable([3,3, 512, kernelmap_number])
    b_conv4_2= bias_variable([kernelmap_number])
    h_conv4_2 = tf.nn.relu( conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)
    #maxpool
    h_pool4 = max_pool_2x2(h_conv4_2)
    return h_pool4
def conv_layer5(x, kernelmap_number):
    # 64x128
    W_conv5 = weight_variable([3,3, 512, kernelmap_number])
    b_conv5= bias_variable([kernelmap_number])
    h_conv5 = tf.nn.relu( conv2d(x, W_conv5) + b_conv5)
    # second layer
    W_conv5_1 = weight_variable([3,3, 512, kernelmap_number])
    b_conv5_1= bias_variable([kernelmap_number])
    h_conv5_1 = tf.nn.relu( conv2d(h_conv5, W_conv5_1) + b_conv5_1)
    #third layer
    W_conv5_2 = weight_variable([3,3, 512, kernelmap_number])
    b_conv5_2= bias_variable([kernelmap_number])
    h_conv5_2 = tf.nn.relu( conv2d(h_conv5_1, W_conv5_2) + b_conv5_2)
    #maxpool
    h_pool5 = max_pool_2x2(h_conv5_2)
    return h_pool5
def fc_layer(x, kernelmap_number):
    W_fc1 = weight_variable([7, 7, 512 , kernelmap_number])
    b_fc1 = bias_variable([kernelmap_number])
    h_fc1 = tf.nn.relu(tf.nn.conv2d(x , W_fc1 , strides=[1, 1, 1, 1], padding='VALID' ) + b_fc1)
    #second fc layer 4096
    W_fc2 = weight_variable([1, 1, 4096 , kernelmap_number])
    b_fc2 = bias_variable([kernelmap_number])
    h_fc2 = tf.nn.relu(tf.nn.conv2d(h_fc1 , W_fc2 , strides=[1, 1, 1, 1], padding='VALID' ) + b_fc2)
    #third fc layer 1000
    W_fc3 = weight_variable([1, 1, 4096 , 2])
    b_fc3 = bias_variable([2])
    h_fc3 = tf.nn.relu(tf.nn.conv2d(h_fc2 , W_fc3 , strides=[1, 1, 1, 1], padding='VALID' ) + b_fc3)
    return h_fc3
def softmax_layer(x):
    W_sf = weight_variable([1, 1, 2 , 2])
    b_sf = bias_variable([2])
    h_sf = tf.nn.softmax(tf.matmul( x , W_sf) + b_sf)
    return h_sf
# 这一行设置 gpu 随使用增长，我一般都会加上
# config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


img_data = tf.image.decode_jpeg(image_raw_data)
image_arr=tf.image.convert_image_dtype(img_data,tf.float32)
image_featuremap = tf.reshape(image_arr,[1,image_SIZE,image_SIZE,3])
layer_data1 = cov_layer1(image_featuremap,  64)

layer_data2 = conv_layer2(layer_data1, 128)
print (layer_data2)
layer_data3 = conv_layer3(layer_data2, 256)
print (layer_data3)
layer_data4 = conv_layer4(layer_data3, 512)
print (layer_data4)
layer_data5 = conv_layer5(layer_data4, 512)
print (layer_data5)
layer_data6 =fc_layer(layer_data5, 4096)
print (layer_data6)

sm_data = softmax_layer(layer_data6)
print (sm_data)
sess.run(tf.global_variables_initializer())
print (sess.run(layer_data6))
print (sess.run(sm_data))
# print(sess.run(sm_data))

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