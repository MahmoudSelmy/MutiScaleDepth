import tensorflow as tf

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

TRAIN_FILE = "train.csv"
TEST_FILE = "train.csv"

class BatchGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    '''
    this function takes the train.csv file and use tensorflow to prepare batches to you 
    we also used it to resize to required size

    '''

    def csv_inputs(self, csv_file_path):
        # print(csv_file_path)
        # list all (image,depth) pairs names and shuffle them
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        # reader to read text files
        reader = tf.TextLineReader()
        # get examples names
        _ , data_examples = reader.read(filename_queue)
        # record csv data into tensors
        image_examples, depth_targets = tf.decode_csv(data_examples, [["path"], ["annotation"]])
        # images
        jpg = tf.read_file(image_examples)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)
        # depth
        depth_png = tf.read_file(depth_targets)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        # nan depth values > zero
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=50 + 3 * self.batch_size,
        )
        return images, depths, invalid_depths
