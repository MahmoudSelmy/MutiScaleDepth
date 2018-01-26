import tensorflow as tf
import numpy as np
import HelperAPI as helper
import math

output_size = 55 * 74

def build_scale1(batch_data):
    # print(batch_data.shape())
    conv_1_s1 = helper.conv2d(input=batch_data, filter_size=11, number_of_channels=3, number_of_filters=96, strides=(4,4),
                              padding='VALID', max_pool=True)
    print(conv_1_s1.get_shape())
    conv_2_s1 = helper.conv2d(input=conv_1_s1, filter_size=5, number_of_channels=96, number_of_filters=256, padding='VALID',
                              max_pool=True)
    print(conv_2_s1.get_shape())
    conv_3_s1 = helper.conv2d(input=conv_2_s1, filter_size=3, number_of_channels=256, number_of_filters=384, padding='SAME',
                              max_pool=False)
    print(conv_3_s1.get_shape())
    conv_4_s1 = helper.conv2d(input=conv_3_s1, filter_size=3, number_of_channels=384, number_of_filters=384, padding='SAME',
                              max_pool=False)
    print(conv_4_s1.get_shape())
    conv_5_s1 = helper.conv2d(input=conv_4_s1, filter_size=3, number_of_channels=384, number_of_filters=256, padding='VALID',
                              max_pool=False)
    print(conv_4_s1.get_shape())
    conv_6_s1 = helper.conv2d(input=conv_5_s1, filter_size=1, number_of_channels=256, number_of_filters=32,
                              padding='VALID',
                              max_pool=False)
    print(conv_6_s1.get_shape())
    flat_con5, num_elements = helper.flatten(conv_6_s1)

    fc_1_s1 = helper.fully_connected(input=flat_con5, input_shape=num_elements, output_shape= output_size,dropout=0.5)
    '''
    fc_1_s1_shape = fc_1_s1.get_shape()
    fc_1_s1_size = fc_1_s1_shape[1:4].num_elements()
    fc_2_s1 = helper.fully_connected(input=fc_1_s1, input_shape=fc_1_s1_size, output_shape=output_size, dropout=0.5)
    '''
    reshaped_op = tf.reshape(fc_1_s1, [-1, 55, 74, 1])

    # print(reshaped_op.get_shape())
    return reshaped_op


def build_scale2(batch_data,scale1_op):
    conv_1_s2 = helper.conv2d(input=batch_data, filter_size=9, number_of_channels=3, number_of_filters=63,
                              strides=(2, 2),
                              padding='VALID', max_pool=True)

    conv_1_s2_dropped = tf.nn.dropout(conv_1_s2, 0.8)

    conv_2_s2_ip = tf.concat([conv_1_s2_dropped, scale1_op],3)

    conv_2_s2 = helper.conv2d(input=conv_2_s2_ip, filter_size=5, number_of_channels=64, number_of_filters=64,
                              padding='SAME',
                              max_pool=False)

    conv_2_s2_dropped = tf.nn.dropout(conv_2_s2, 0.8)

    conv_3_s2 = helper.conv2d(input=conv_2_s2_dropped, filter_size=5, number_of_channels=64, number_of_filters=1,
                              padding='SAME',
                              max_pool=False)
    print(conv_3_s2.get_shape())
    return conv_3_s2

def build_loss(scale2_op, depths, pixels_mask):
    # print(pixels_mask.get_shape())
    predictions_all = tf.reshape(scale2_op, [-1, output_size])
    depths_all = tf.reshape(depths, [-1, output_size])
    pixels_mask = tf.reshape(pixels_mask, [-1, output_size])

    # print(predictions_all.get_shape())
    # print(pixels_mask.get_shape())

    n =tf.reduce_sum(pixels_mask,1) # all_subset_data images does not have any invalid pixels
    # print("n")
    # print(n.get_shape())
    predictions_valid = tf.multiply(predictions_all, pixels_mask)
    target_valid = tf.multiply(depths_all, pixels_mask)

    # print(predictions_valid.get_shape())
    # print(target_valid.get_shape())

    d = tf.subtract(predictions_valid, target_valid)
    square_d = tf.square(d)

    sum_square_d = tf.reduce_sum(square_d, 1)
    # print(sum_square_d.get_shape())
    sum_d = tf.reduce_sum(d, 1)
    # print(sum_square_d.get_shape())
    sqare_sum_d = tf.square(sum_d)

    cost = tf.reduce_mean( (sum_square_d / n ) - 0.5* (sqare_sum_d / tf.pow(n, 2) ))

    return cost