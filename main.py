import tensorflow as tf
import tensorflow_datasets as tfds
import os
import yaml
import boto3
import tarfile
import time
import logging
from cloudwatch import cloudwatch
from util import read_mnist


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger, s3_session, s3_bucket):
        self.logger = logger
        self.s3_session = s3_session
        self.s3_bucket = s3_bucket

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.info(f'Started epoch {epoch} of training...')

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(f'Accuracy at end of epoch {epoch}: {logs["val_sparse_categorical_accuracy"]}')

        intermediate_model_filename = f'intermediate-model-{epoch}'
        self.model.save('data/' + intermediate_model_filename)
        with tarfile.open(f'data/{intermediate_model_filename}.tar.gz', 'w:gz') as tar:
            tar.add('data/' + intermediate_model_filename,
                    arcname=os.path.basename('data/' + intermediate_model_filename))
        s3 = self.s3_session.client('s3')
        s3.upload_file(f'data/{intermediate_model_filename}.tar.gz', self.s3_bucket,
                       f'{intermediate_model_filename}.tar.gz')

    def on_train_end(self, logs=None):
        self.logger.info(f'Training complete! Final model accuracy: {logs["val_sparse_categorical_accuracy"]}')


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def training_pipeline(ds_train, ds_info):
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    return ds_train


def evaluation_pipeline(ds_test):
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_test


def load_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    return ds_train, ds_test, ds_info


def load_data_from_s3(
        boto_session, bucket_name, data_filename, label_filename):
    s3 = boto_session.client('s3')

    s3.download_file(bucket_name, data_filename, 'data/' + data_filename)
    s3.download_file(bucket_name, label_filename, 'data/' + label_filename)
    train_examples, train_labels = read_mnist('data/' + data_filename, 'data/' + label_filename)
    ds = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

    return ds


def train_model(ds_train, ds_test, logger, s3_session, s3_bucket):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.fit(
        ds_train,
        epochs=25,
        validation_data=ds_test,
        verbose=2,
        callbacks=[CustomCallback(logger=logger, s3_session=s3_session, s3_bucket=s3_bucket)]
    )
    return model


def bundle_directory(dir_name):
    with tarfile.open(f'{dir_name}.tar.gz', 'w:gz') as tar:
        tar.add(dir_name)


def upload_to_s3(boto_session, filename, bucket_name):
    s3 = boto_session.client('s3')
    s3.upload_file('data/' + filename, bucket_name, filename)


if __name__ == "__main__":
    # Get configuration
    CONFIG_FILE = 'config.yaml'
    with open(CONFIG_FILE, 'r') as f:
        config_data = yaml.safe_load(f)
    REGION = config_data['AWS']['REGION']
    S3_BUCKET = config_data['AWS']['S3']['S3_BUCKET']
    LOG_GROUP = config_data['AWS']['CLOUDWATCH']['LOG_GROUP']
    AWS_ACCESS_KEY_ID = config_data['AWS']['AUTH']['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = config_data['AWS']['AUTH']['AWS_SECRET_ACCESS_KEY']
    MODEL_NAME = config_data['MODEL']['NAME']
    LOG_STREAM = str(int(time.time()))
    TRAIN_DATA_FILENAME = config_data['DATA']['TRAIN']
    TEST_DATA_FILENAME = config_data['DATA']['TEST']
    TRAIN_LABEL_FILENAME = config_data['DATA']['TRAIN_LABELS']
    TEST_LABEL_FILENAME = config_data['DATA']['TEST_LABELS']
    CLOUDWATCH_LOGS = config_data['OPTIONS']['LOG_TO_CLOUDWATCH']
    S3_UPLOAD = config_data['OPTIONS']['SAVE_TO_S3']

    # Create S3 Boto3 Session
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if CLOUDWATCH_LOGS:
        # Set up CloudWatch
        handler = cloudwatch.CloudwatchHandler(
            log_group=LOG_GROUP,
            log_stream=LOG_STREAM,
            region=REGION,
            access_id=AWS_ACCESS_KEY_ID,
            access_key=AWS_SECRET_ACCESS_KEY
        )
        logger.addHandler(handler)

    # Load the dataset
    logger.info('Loading dataset...')
    # ds_train, ds_test, ds_info = load_data()
    ds_train = load_data_from_s3(session, S3_BUCKET, TRAIN_DATA_FILENAME, TRAIN_LABEL_FILENAME)
    ds_test = load_data_from_s3(session, S3_BUCKET, TEST_DATA_FILENAME, TEST_LABEL_FILENAME)

    # Build the training and testing pipelines
    logger.info('Building training and testing pipelines...')
    # ds_train = training_pipeline(ds_train, ds_info)
    # ds_test = evaluation_pipeline(ds_test)
    ds_train = ds_train.shuffle(100).batch(64)
    ds_test = ds_test.batch(64)

    # Train the model
    logger.info('Training model...')
    model = train_model(ds_train, ds_test, logger, session, S3_BUCKET)
    model.save('data/' + MODEL_NAME)

    # Bundle model
    logger.info('Bundling finished model...')
    bundle_directory('data/' + MODEL_NAME)

    if S3_UPLOAD:
        # Persist final model in S3
        logger.info('Uploading to data store...')
        upload_to_s3(session, f'{MODEL_NAME}.tar.gz', S3_BUCKET)
