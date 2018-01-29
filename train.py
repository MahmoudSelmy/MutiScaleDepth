from datetime import datetime
from tensorflow.python.platform import gfile
import tensorflow as tf
from data_preprocessing import BatchGenerator
import model
from Utills import output_predict

BATCH_SIZE = 4
TRAIN_FILE = "sub_train.csv"
TEST_FILE = "test.csv"
EPOCHS = 35

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74


INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_FACTOR = 0.9
MOVING_AVERAGE_DECAY = 0.999999
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 30

SCALE1_DIR = 'Scale1'
SCALE2_DIR = 'Scale2'
logs_path = "/tmp/multi_scale/2"

def train_model(continue_flag,restore_scale2,train_scale2,freeze_scale1):

    # directors to save the chkpts of each scale
    if not gfile.Exists(SCALE1_DIR):
        gfile.MakeDirs(SCALE1_DIR)
    if not gfile.Exists(SCALE2_DIR):
        gfile.MakeDirs(SCALE2_DIR)

    with tf.Graph().as_default():
        # get batch
        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.device('/cpu:0'):
            batch_generator = BatchGenerator(batch_size=BATCH_SIZE)
            train_images, train_depths, train_pixels_mask = batch_generator.csv_inputs(TRAIN_FILE)
        '''
        # placeholders
            training_images = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="training_images")
            depths = tf.placeholder(tf.float32, shape=[None, TARGET_HEIGHT, TARGET_WIDTH, 1], name="depths")
            pixels_mask = tf.placeholder(tf.float32, shape=[None, TARGET_HEIGHT, TARGET_WIDTH, 1], name="pixels_mask")
        '''
        # build model
        scale1 = model.build_scale1(train_images,freeze_weights=freeze_scale1)
        if train_scale2:
            scale2 = model.build_scale2(batch_data=train_images, scale1_op= scale1)
            loss = model.build_loss(scale2_op= scale2,depths=train_depths,pixels_mask=train_pixels_mask)
        else:
            loss = model.build_loss(scale2_op=scale1, depths=train_depths, pixels_mask=train_pixels_mask)

        #learning rate
        num_batches_per_epoch = float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            decay_steps,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True)

        #optimizer
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        # TODO: define model saver

        # Training session
        # sess_config = tf.ConfigProto(log_device_placement=True)
        # sess_config.gpu_options.allocator_type = 'BFC'
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.80

        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        '''
        # loss summary
        tf.summary.scalar("loss", loss)
        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # create log writer object
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

            # Saver
            # dictionary to each scale to define to seprate collections
            scale1_params = {}
            scale2_params = {}
            # add variables to it's corresponding dictionary
            for variable in tf.all_variables():
                variable_name = variable.name
                if variable_name.find('s1') >= 0:
                    scale1_params[variable_name] = variable
                if train_scale2:
                    if variable_name.find('s2') >= 0:
                        scale2_params[variable_name] = variable
            # define savers
            saver_scale1 = tf.train.Saver(scale1_params)
            if train_scale2:
                saver_scale2 = tf.train.Saver(scale2_params)

            # restore params if we need to continue on the previous training
            if continue_flag:
                scale1_ckpt = tf.train.get_checkpoint_state(SCALE1_DIR)
                if scale1_ckpt and scale1_ckpt.model_checkpoint_path:
                    print("Scale1 params Loading.")
                    saver_scale1.restore(sess, scale1_ckpt.model_checkpoint_path)
                    print("Scale1 params Restored.")
                else:
                    print("No Params available")
                if restore_scale2:
                    scale2_ckpt = tf.train.get_checkpoint_state(SCALE2_DIR)
                    if scale2_ckpt and scale2_ckpt.model_checkpoint_path:
                        print("Scale2 params Loading.")
                        scale2_ckpt.restore(sess, scale2_ckpt.model_checkpoint_path)
                        print("Scale2 params Restored.")
                    else:
                        print("No Scale2 Params available")

            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for epoch in range(EPOCHS):
                for i in range(1000):
                    if train_scale2:
                        _, loss_value, predections_s1,predections_s2, batch_images,summary = sess.run([optimizer,loss,scale1,scale2, train_images,summary_op])
                    else:
                        _, loss_value, predections_s1, batch_images, summary = sess.run([optimizer, loss, scale1, train_images, summary_op])

                    writer.add_summary(summary, epoch * 1000 + i)

                    if i % 100 == 0:
                        # log.info('step' + loss_value)
                        print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, i, loss_value))

                    # print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, i, loss_value))
                    if i % 500 == 0:
                        # save predictions
                        if not freeze_scale1:
                            output_predict(predections_s1, batch_images, "data/predictions/predict_scale1_%05d_%05d" % (epoch, i))
                        if train_scale2:
                            output_predict(predections_s2, batch_images, "data/predictions/predict_scale2_%05d_%05d" % (epoch, i))
                if not freeze_scale1:
                    scale1_checkpoint_path = SCALE1_DIR + '/model'
                    saver_scale1.save(sess, scale1_checkpoint_path)
                if train_scale2:
                    scale2_checkpoint_path = SCALE2_DIR + '/model'
                    saver_scale2.save(sess, scale2_checkpoint_path)

            # stop our queue threads and properly close the session
            coord.request_stop()
            coord.join(threads)
            sess.close()

def main(argv=None):
    train_model(True,False,True,True)

if __name__ == '__main__':
    tf.app.run()