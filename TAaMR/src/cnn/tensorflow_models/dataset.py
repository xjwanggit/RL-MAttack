import tensorflow as tf
import os

class CustomDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [os.path.join(root_dir, fname) for fname in
                          sorted(os.listdir(root_dir), key=lambda x: int(x.split(".")[0]))]
        self.samples_number = len(self.filenames)

    def load_and_preprocess_image(self, file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixels to [0,1]

        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization if no transform is provided
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            image = (image - mean) / std  # Normalize image

        return image

    def to_tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        dataset = dataset.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) # tf.data.experimental.AUTOTUNE自动选择cpu核心
        return dataset
