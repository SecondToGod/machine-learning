import tensorflow as tf
import numpy as np
import os
import cv2
 
IMAGE_W = 600
IMAGE_H = 600
 
Ratio = None
 
INI_NOISE_RATIO = 0.7
STYLE_STRENGTH = 500
ITERATION = 2000
CONTENT_LAYERS = [('conv4_2', 1.)]
STYLE_LAYERS = [('conv1_1', 2.), ('conv2_1', 1.), ('conv3_1', 0.5), ('conv4_1', 0.25), ('conv5_1', 0.125)]
layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
          'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', ]
 
 
def vgg19(input, model_path=None):
    '''
    Build the VGG19 network, which is initialized with the pre-trained VGG19 model.
    :param input: The input image.
    :param model_path:Which path the VGG19 model is stored.
    :return:A python dict, which contains all the layers needed.
    '''
    if model_path is None:
        model_path = 'vgg19.npy'
 
    if os.path.isfile(model_path) is False:
        raise FileNotFoundError('vgg19.npy cannot be found!!!')
 
    wDict = np.load(model_path, encoding="bytes").item()
 
    net = {}
    net['input'] = input
 
    # conv1_1
    weight1_1 = tf.Variable(initial_value=wDict['conv1_1'][0], trainable=False)
    bias1_1 = tf.Variable(wDict['conv1_1'][1], trainable=False)
    net['conv1_1'] = tf.nn.relu(tf.nn.conv2d(net['input'], weight1_1, [1, 1, 1, 1], 'SAME') + bias1_1)
 
    # conv1_2
    weight1_2 = tf.Variable(wDict['conv1_2'][0], trainable=False)
    bias1_2 = tf.Variable(wDict['conv1_2'][1], trainable=False)
    net['conv1_2'] = tf.nn.relu(tf.nn.conv2d(net['conv1_1'], weight1_2, [1, 1, 1, 1], 'SAME') + bias1_2)
 
    # pool1
    net['pool1'] = tf.nn.avg_pool(net['conv1_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
 
    # conv2_1
    weight2_1 = tf.Variable(wDict['conv2_1'][0], trainable=False)
    bias2_1 = tf.Variable(wDict['conv2_2'][1], trainable=False)
    net['conv2_1'] = tf.nn.relu(tf.nn.conv2d(net['pool1'], weight2_1, [1, 1, 1, 1], 'SAME') + bias2_1)
 
    # conv2_2
    weight2_2 = tf.Variable(wDict['conv2_2'][0], trainable=False)
    bias2_2 = tf.Variable(wDict['conv2_2'][1], trainable=False)
    net['conv2_2'] = tf.nn.relu(tf.nn.conv2d(net['conv2_1'], weight2_2, [1, 1, 1, 1], 'SAME') + bias2_2)
 
    # pool2
    net['pool2'] = tf.nn.avg_pool(net['conv2_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
 
    # conv3_1
    weight3_1 = tf.Variable(wDict['conv3_1'][0], trainable=False)
    bias3_1 = tf.Variable(wDict['conv3_1'][1], trainable=False)
    net['conv3_1'] = tf.nn.relu(tf.nn.conv2d(net['pool2'], weight3_1, [1, 1, 1, 1], 'SAME') + bias3_1)
 
    # conv3_2
    weight3_2 = tf.Variable(wDict['conv3_2'][0], trainable=False)
    bias3_2 = tf.Variable(wDict['conv3_2'][1], trainable=False)
    net['conv3_2'] = tf.nn.relu(tf.nn.conv2d(net['conv3_1'], weight3_2, [1, 1, 1, 1], 'SAME') + bias3_2)
 
    # conv3_3
    weight3_3 = tf.Variable(wDict['conv3_3'][0], trainable=False)
    bias3_3 = tf.Variable(wDict['conv3_3'][1], trainable=False)
    net['conv3_3'] = tf.nn.relu(tf.nn.conv2d(net['conv3_2'], weight3_3, [1, 1, 1, 1], 'SAME') + bias3_3)
 
    # conv3_4
    weight3_4 = tf.Variable(wDict['conv3_4'][0], trainable=False)
    bias3_4 = tf.Variable(wDict['conv3_4'][1], trainable=False)
    net['conv3_4'] = tf.nn.relu(tf.nn.conv2d(net['conv3_3'], weight3_4, [1, 1, 1, 1], 'SAME') + bias3_4)
 
    # pool3
    net['pool3'] = tf.nn.avg_pool(net['conv3_4'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
 
    # conv4_1
    weight4_1 = tf.Variable(wDict['conv4_1'][0], trainable=False)
    bias4_1 = tf.Variable(wDict['conv4_1'][1], trainable=False)
    net['conv4_1'] = tf.nn.relu(tf.nn.conv2d(net['pool3'], weight4_1, [1, 1, 1, 1], 'SAME') + bias4_1)
 
    # conv4_2
    weight4_2 = tf.Variable(wDict['conv4_2'][0], trainable=False)
    bias4_2 = tf.Variable(wDict['conv4_2'][1], trainable=False)
    net['conv4_2'] = tf.nn.relu(tf.nn.conv2d(net['conv4_1'], weight4_2, [1, 1, 1, 1], 'SAME') + bias4_2)
 
    # conv4_3
    weight4_3 = tf.Variable(wDict['conv4_3'][0], trainable=False)
    bias4_3 = tf.Variable(wDict['conv4_3'][1], trainable=False)
    net['conv4_3'] = tf.nn.relu(tf.nn.conv2d(net['conv4_2'], weight4_3, [1, 1, 1, 1], 'SAME') + bias4_3)
 
    # conv4_4
    weight4_4 = tf.Variable(wDict['conv4_4'][0], trainable=False)
    bias4_4 = tf.Variable(wDict['conv4_4'][1], trainable=False)
    net['conv4_4'] = tf.nn.relu(tf.nn.conv2d(net['conv4_3'], weight4_4, [1, 1, 1, 1], 'SAME') + bias4_4)
 
    # pool4
    net['pool4'] = tf.nn.avg_pool(net['conv4_4'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
 
    # conv5_1
    weight5_1 = tf.Variable(wDict['conv5_1'][0], trainable=False)
    bias5_1 = tf.Variable(wDict['conv5_1'][1], trainable=False)
    net['conv5_1'] = tf.nn.relu(tf.nn.conv2d(net['pool4'], weight5_1, [1, 1, 1, 1], 'SAME') + bias5_1)
 
    # conv5_2
    weight5_2 = tf.Variable(wDict['conv5_2'][0], trainable=False)
    bias5_2 = tf.Variable(wDict['conv5_2'][1], trainable=False)
    net['conv5_2'] = tf.nn.relu(tf.nn.conv2d(net['conv5_1'], weight5_2, [1, 1, 1, 1], 'SAME') + bias5_2)
 
    # conv5_3
    weight5_3 = tf.Variable(wDict['conv5_3'][0], trainable=False)
    bias5_3 = tf.Variable(wDict['conv5_3'][1], trainable=False)
    net['conv5_3'] = tf.nn.relu(tf.nn.conv2d(net['conv5_2'], weight5_3, [1, 1, 1, 1], 'SAME') + bias5_3)
 
    # conv5_4
    weight5_4 = tf.Variable(wDict['conv5_4'][0], trainable=False)
    bias5_4 = tf.Variable(wDict['conv5_4'][1], trainable=False)
    net['conv5_4'] = tf.nn.relu(tf.nn.conv2d(net['conv5_3'], weight5_4, [1, 1, 1, 1], 'SAME') + bias5_4)
 
    # pool5
    net['pool5'] = tf.nn.avg_pool(net['conv5_4'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
 
    return net
 
def gram_matrix(tensor, length, depth):
    '''
    :param tensor:The tensor you need to convert, which could be a numpy array or a TensorFlow tensor.
    :param length:The length you need to convert to.
    :param depth:The depth you need to convert to.
    :return:A tensor.
    '''
    tensor = tf.reshape(tensor, (length, depth))
    return tf.matmul(tf.transpose(tensor), tensor)
    pass
 
def build_content_loss(combination, content):
    '''
    :param combination:The network which is the combination of the style network and content network,
                       that is the style-transfered network
    :param content:The network of the content image
    :return:The loss between the combination and the content
    '''
    content_sum = 0.0
    for i, l in enumerate(CONTENT_LAYERS):
        shape = combination[l[0]].get_shape()
        M = shape[1].value * shape[2].value
        N = shape[3].value
        content_sum += l[1] * 0.25/(M ** 0.5 + N ** 0.5) * tf.reduce_sum(tf.pow(combination[l[0]] - content[l[0]], 2))
    return content_sum
    pass
 
def build_style_loss(combination, style):
    '''
    :param combination: The network which is the combination of the style network and content network,
                       that is the style-transfered network
    :param style: The network of the style image
    :return: The loss between the combination and the style
    '''
    style_sum = 0.0
    for i, l in enumerate(STYLE_LAYERS):
        shape = combination[l[0]].get_shape()
        M = shape[1].value * shape[2].value
        N = shape[3].value
        para1 = combination[l[0]]
        para2 = style[l[0]]
 
        sub = gram_matrix(para1, M, N) - gram_matrix(para2, M, N)
        sum = tf.reduce_sum(tf.pow(sub, 2))
        pre = l[1] * 1.0 / (4 * N ** 2 * M ** 2)
 
        style_sum += tf.multiply(pre, sum)
 
    return style_sum
    pass
 
def main():
    # Define a placeholder
    myinput = tf.placeholder(dtype=tf.float32, shape=[1, IMAGE_H, IMAGE_W, 3])
 
    # Read the style image
    raw_styleimg = cv2.imread("style.jpg")
    raw_styleimg = cv2.resize(raw_styleimg, (IMAGE_H, IMAGE_W))
    #cv2.resize(raw_styleimg,raw_styleimg,(IMAGE_H,IMAGE_W),0.5,0.5)
 
    styleimg = np.expand_dims(raw_styleimg, 0)
 
    # The normalization method of th style image.
    # Actually, I have tried many methods, and this one is the most useful and powerful.
    styleimg[0][0] -= 123
    styleimg[0][1] -= 117
    styleimg[0][2] -= 104
    style = tf.Variable(styleimg, dtype=tf.float32, trainable=False)
 
    raw_contentimg = cv2.imread("content.jpg")
 
    # Store the ratio of the content image.
    Ratio = raw_contentimg.shape
    raw_contentimg = cv2.resize(raw_contentimg, (IMAGE_H, IMAGE_W))
 
    contentimg = np.expand_dims(raw_contentimg, 0)
    contentimg[0][0] -= 123
    contentimg[0][1] -= 117
    contentimg[0][2] -= 104
    content = tf.Variable(contentimg, dtype=tf.float32, trainable=False)
 
    # The combination image, which is consisted of noise image and content image.
    combination = INI_NOISE_RATIO*np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32') + (1.-INI_NOISE_RATIO) * contentimg
 
    combinat = tf.Variable(combination, dtype=tf.float32, trainable=True)
 
    # Build all the networks
    stylenet = vgg19(myinput * style)
 
    contentnet = vgg19(myinput * content)
 
    combinationnet = vgg19(myinput * combinat)
 
    # Define the loss function
    loss = 500 * build_style_loss(combinationnet, stylenet) + build_content_loss(combinationnet, contentnet)
 
    # Here, AdamOptimizer is used, and the learning rate is 2.0
    train = tf.train.AdamOptimizer(2).minimize(loss)
 
    # Input Image, consisted of 1s.
    img = np.ones(dtype=np.float32, shape=[1, IMAGE_H, IMAGE_W, 3])
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
 
        for i in range(ITERATION):
            print(sess.run(loss, feed_dict={myinput: img}))
            sess.run(train, feed_dict={myinput: img})
 
            # Actually, the COMBINATION is the final output.
            pic = sess.run(combinat, feed_dict={myinput: img})[0]
            pic[0] += 123
            pic[1] += 117
            pic[2] += 104
            cv2.imwrite('results/%d.jpg' % i, cv2.resize(pic, (Ratio[1], Ratio[0])))
 
 
if __name__ == '__main__':
    main()