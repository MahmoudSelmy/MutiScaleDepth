from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from data_preprocessing import BatchGenerator
import model
import logging as log

BATCH_SIZE = 2
TRAIN_FILE = "train.csv"
EPOCHS = 10

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74


INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.9
MOVING_AVERAGE_DECAY = 0.999999
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 30

def train_model():

    with tf.Graph().as_default():
        # get batch
        global_step = tf.Variable(0, name='global_step', trainable=False)
        batch_generator = BatchGenerator(batch_size=BATCH_SIZE)
        images, depths, pixels_mask = batch_generator.csv_inputs(TRAIN_FILE)
        '''
        # placeholders
            images = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="images")
            depths = tf.placeholder(tf.float32, shape=[None, TARGET_HEIGHT, TARGET_WIDTH, 1], name="depths")
            pixels_mask = tf.placeholder(tf.float32, shape=[None, TARGET_HEIGHT, TARGET_WIDTH, 1], name="pixels_mask")
        '''
        # build model
        scale1 = model.build_scale1(images)
        scale2 = model.build_scale2(batch_data=images, scale1_op= scale1)
        loss = model.build_loss(scale2_op= scale2,depths=depths,pixels_mask=pixels_mask)
        #learning rate
        # lr = 0.0001

        num_batches_per_epoch = float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            decay_steps,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True)

        #optimizer
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        # TODO: define model saver
        # Training session
        # sess_config = tf.ConfigProto(log_device_placement=True)
        # sess_config.gpu_options.allocator_type = 'BFC'
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.80
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for epoch in range(EPOCHS):
                for i in range(1000):
                    _, loss_value, predections_s1,predections_s2, batch_images = sess.run([optimizer,loss,scale1,scale2, images])
                if i % 2 == 0:
                    # log.info('step' + loss_value)
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, i, loss_value))
                if i % 500 == 0:
                    '''
                        TODO: save scale1 and scale2 predictions
                    '''
            # stop our queue threads and properly close the session
            coord.request_stop()
            coord.join(threads)
            sess.close()

def main(argv=None):
    train_model()

if __name__ == '__main__':
    tf.app.run()