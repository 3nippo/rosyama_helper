import sys
import shutil
import tensorflow as tf
import os
import numpy as np
import cv2


"""
frames_dir
model_dir
masks_dir
"""


HEIGHT = 256
WIDTH = 256
FRAMES_PER_PREDICT = 5000


def list_files(dir):
    return tf.data.Dataset.list_files(
        os.path.join(dir, '*'),
        shuffle=False
    )


def load_model(model_dir):
    return tf.keras.models.load_model(model_dir)


# preprocessing
#
#
def load_image(image_path):
    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(image, expand_animations=False) 
    
    image = tf.cast(image, tf.float32)
    
    return image


def normalize(image):
    image = (image / 127.5) - 1
    
    return image


def resize(image, height, width):
    image = tf.image.resize(
        image, 
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return image


def basic_preprocess(image):
    image = resize(image, HEIGHT, WIDTH)
    image = normalize(image)
    return image
#
#
# preprocessing

def frames_to_dataset(frames_dir):
    images = list_files(frames_dir).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    preprocessed_images = images.map(basic_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return preprocessed_images.batch(1)


def get_frames_count(frames_dir):
    return len(os.listdir(frames_dir))


def save_masks(masks, masks_dir, start_idx):
    masks = np.argmax(masks, axis=3)
    
    mask_path_fmt = os.path.join(
        masks_dir,
        "{}.png"
    )

    for i in range(masks.shape[0]):
        cv2.imwrite(
            mask_path_fmt.format(start_idx + i),
            masks[i]
        )


def obtain_masks(frames_dir, model, masks_dir):
    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)

    os.mkdir(masks_dir)

    frames_count = get_frames_count(frames_dir)

    frames_dataset = frames_to_dataset(frames_dir)
    
    full_chunks_count = frames_count // FRAMES_PER_PREDICT

    last_chunk_frames_count = frames_count % FRAMES_PER_PREDICT
    
    for chunk_idx in range(full_chunks_count):
        start_idx = chunk_idx * FRAMES_PER_PREDICT

        predicted = model.predict(
            frames_dataset.skip(start_idx),
            steps=FRAMES_PER_PREDICT
        )

        save_masks(predicted, masks_dir, start_idx)

    if last_chunk_frames_count != 0:
        start_idx = full_chunks_count * FRAMES_PER_PREDICT

        predicted = model.predict(
            frames_dataset.skip(start_idx)
        )

        save_masks(predicted, masks_dir, start_idx)


if __name__ == '__main__':
    frames_dir, model_dir, masks_dir = sys.argv[1:]

    model = load_model(model_dir)

    obtain_masks(frames_dir, model, masks_dir)
