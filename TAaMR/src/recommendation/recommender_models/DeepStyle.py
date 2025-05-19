import os
# from cnn.tensorflow_models.dataset import *
# from cnn.tensorflow_models.model import *
import pandas as pd
import torch
import torch.nn as nn
from utils.read import *
from utils.write import *
from torchvision import transforms
import torchvision.models as models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, ReLU, Lambda, GlobalAveragePooling2D
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import LayerNormalization
tf.compat.v1.disable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
class DeepStyle:

    def __init__(self, args, num_users, num_items, num_image_feature):
        self.emb_K = args.emb1_K  # emb1_K表示的是嵌入的特征量，这里是64
        self.lr = eval(args.lr)
        self.slr = args.lr
        self.bsz = args.batch_size
        self.val_bsz = args.validation_batch_size

        self.regs = args.regs  # lambdas for regularization
        regs = eval(self.regs)
        self.l1 = regs[0]
        self.l2 = regs[1]
        self.l3 = regs[2]
        self.lmd = args.lmd  # lambda for balance the common loss and adversarial loss
        self.adv = args.adv
        self.adv_type = args.adv_type
        self.epsilon = args.epsilon
        self.num_users = num_users
        self.num_items = num_items
        self.num_image_feature = num_image_feature
        self.watch = []
        self.index_to_image_path = {i: f"'../../../data/{args.dataset}/{args.experiment_name}/images/{i}.jpg" for i in
                                    range(num_items)}
        self.item_category = tf.convert_to_tensor(
            pd.read_csv(f'../data/{args.dataset}/{args.experiment_name}/classes.csv')['ClassNum'].tolist(), dtype=tf.int32
        )
        self.gpu_setting()
        self.build_feature_model()
        # self.feature_extract(args)
        self.build_graph()

    def gpu_setting(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Set GPU memory growth to avoid pre-allocating all memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error initializing GPU: {e}")
        else:
            print("No GPU found. Running on CPU.")

    def build_feature_model(self):
        # Load the AlexNet model with pre-trained weights from PyTorch
        alexnet_model = models.alexnet(pretrained=True)

        # Extract feature layers from AlexNet
        feature_layers = list(alexnet_model.features.children())
        avgpool_layer = [alexnet_model.avgpool]
        classifier_layers = list(alexnet_model.classifier.children())[:-1]  # Remove the final classification layer

        # Combine all layers to match AlexNet's feature extraction
        model_layers = feature_layers + avgpool_layer + classifier_layers

        def pytorch_conv_to_tf(layer, input_tensor):
            weights, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy() if layer.bias is not None else None
            out_channels, in_channels, kh, kw = weights.shape

            # Create a corresponding TensorFlow layer
            tf_layer = Conv2D(
                filters=out_channels,
                kernel_size=(kh, kw),
                strides=layer.stride,
                padding='same' if layer.padding[0] > 0 else 'valid',
                weights=[weights.transpose(2, 3, 1, 0), bias] if bias is not None else [weights.transpose(2, 3, 1, 0)],
                activation=None,
                trainable=True
            )
            return tf_layer(input_tensor)

        # Define the input shape
        inputs = Input(shape=(224, 224, 3))
        x = inputs

        # Function to convert PyTorch Conv2D to TensorFlow Conv2D
        for layer in model_layers:
            if isinstance(layer, nn.Conv2d):
                x = pytorch_conv_to_tf(layer, x)
            elif isinstance(layer, nn.ReLU):
                x = ReLU()(x)
            elif isinstance(layer, nn.MaxPool2d):
                x = MaxPooling2D(pool_size=layer.kernel_size, strides=layer.stride, padding='valid')(x)
            elif isinstance(layer, nn.Flatten):
                flatten_needed = True
            elif isinstance(layer, nn.Dropout):
                x = Dropout(rate=layer.p)(x)
            elif isinstance(layer, nn.Linear):
                # Check the input shape to match the expected shape for the Dense layer
                input_features = layer.in_features
                if input_features != x.shape[-1]:
                    x = Flatten()(x)  # Ensure input is flattened correctly

                # Get weights and biases from PyTorch
                weights, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
                tf_layer = Dense(
                    units=layer.out_features,
                    weights=[weights.T, bias],
                    activation=None,
                    trainable=True
                )
                x = tf_layer(x)

        # Create the final Keras feature model
        self.feature_model = Model(inputs=inputs, outputs=x)


    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="user_input")
            self.pos_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pos_input")
            self.neg_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="neg_input")


            # self.init_image = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_image_feature],  # 初始图像的特征，行为物品的数量，列为图像特征 (82630, 2048)
            #                                            name="pos_image")

            # self.init_image = tf.compat.v1.placeholder(tf.float32, shape=[self.num_items, self.num_image_feature],  # 初始图像的特征，行为物品的数量，列为图像特征 (82630, 2048)
            #                                            name="pos_image")
            self.init_emb_P = tf.compat.v1.placeholder(tf.float32, shape=[self.num_users, self.emb_K],  # 用户嵌入的特征，size为（用户的数量，特征向量）（26155，64）
                                                       name="init_emb_P")
            self.init_emb_Q = tf.compat.v1.placeholder(tf.float32, shape=[self.num_items, self.emb_K],  # 物品嵌入的特征，size为（物品的数量，特征向量）（82630，64）
                                                       name="init_emb_Q")
            self.init_emb_Q_feature = tf.compat.v1.placeholder(tf.float32, shape=[self.num_items, self.num_image_feature],  # 物品嵌入的特征，size为（物品的数量，特征向量）（82630，64）
                                                       name="init_emb_Q")
            self.init_emb_Q_bias = tf.compat.v1.placeholder(tf.float32, shape=[1000, self.emb_K],  # 物品嵌入的特征误差，size为（物品的数量，特征向量）（82630，64）
                                                       name="init_emb_Q_bias")
            self.image_paths = tf.constant([self.index_to_image_path[i] for i in range(self.num_items)], dtype=tf.string)

    def _load_images_by_indices(self):
        """
           Fetches paths for a list of indices and loads the images.
           """
        # Gather paths using pos_input and neg_input
        pos_image_paths = tf.gather(self.image_paths, self.pos_input)
        neg_image_paths = tf.gather(self.image_paths, self.neg_input)

        # Load and preprocess images from the gathered paths using map_fn
        self.pos_images_pixel = tf.map_fn(self._load_and_preprocess_image, pos_image_paths, dtype=tf.float32)
        self.neg_images_pixel = tf.map_fn(self._load_and_preprocess_image, neg_image_paths, dtype=tf.float32)


    def _load_and_preprocess_image(self, file_name):
        """
        Load and preprocess an image from a file name.
        This function reads the image, resizes it, and normalizes the pixel values.
        """

        def decode_image(image_string, file_name):
            try:
                # Decode the image string to an image tensor
                image = tf.image.decode_jpeg(image_string, channels=3)  # Specifically use decode_jpeg for JPEG images
                # Explicitly set the shape of the decoded image
                image.set_shape([None, None, 3])
                return tf.cast(image, tf.float32)
            except Exception as e:
                # Log the problematic image path and the error
                print(f"Error decoding image: {file_name.numpy().decode('utf-8')}, Error: {str(e)}")
                # Return a blank image placeholder (or handle accordingly)
                return tf.zeros([224, 224, 3], dtype=tf.float32)

        # Read the image file content
        image_string = tf.io.read_file(file_name)

        # Use tf.py_function to execute the Python-level decode_image function
        image = tf.py_function(func=decode_image, inp=[image_string, file_name], Tout=tf.float32)

        # Ensure the tensor shape is known after py_function, as it can lose shape information
        image.set_shape([224, 224, 3])

        # Resize and normalize the image
        image = tf.image.resize(image, [224, 224])  # Resize to match the input size of the CNN
        image = image / 255.0  # Normalize pixel values to [0, 1]

        return image


    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.emb_P = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_users, self.emb_K], mean=0.0, stddev=0.01),
                name='emb_P', dtype=tf.float32)  # (users, embedding_size)
            self.emb_Q = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_items, self.emb_K], mean=0.0, stddev=0.01),
                name='emb_Q', dtype=tf.float32)  # (items, embedding_size)
            self.emb_Q_feature = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_items, self.num_image_feature], mean=0.0, stddev=0.01),
                name='emb_Q_feature', dtype=tf.float32, trainable=False)  # (items, embedding_size)
            self.emb_Q_bias = tf.Variable(
                tf.random.truncated_normal(shape=[1000, self.emb_K], mean=0.0, stddev=0.01),
                name='emb_Q_bias', dtype=tf.float32)  # (categories, embedding_size)


        with tf.name_scope("init_op"):
            self.assign_P = tf.compat.v1.assign(self.emb_P, self.init_emb_P)
            self.assign_Q = tf.compat.v1.assign(self.emb_Q, self.init_emb_Q)
            self.assign_Q_feature = tf.compat.v1.assign(self.emb_Q_feature, self.init_emb_Q_feature)
            self.assign_Q_bias = tf.compat.v1.assign(self.emb_Q_bias, self.init_emb_Q_bias)

        with tf.name_scope("image_transfer"):
            self.phi = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_image_feature, self.emb_K], mean=0.0, stddev=0.01),
                name='phi', dtype=tf.float32)  # 这里相当于把图像特征的4096转译成64


    def _create_inference(self, user_input, item_input, item_pixel, adv=False):
        with tf.name_scope("inference"):
            self.emb_p = tf.nn.embedding_lookup(self.emb_P, user_input)  # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
            self.feature = self.feature_model(item_pixel)
            self.emb_q_embed = tf.nn.embedding_lookup(self.emb_Q, item_input)
            self.emb_q = tf.matmul(self.feature, self.phi) + self.emb_q_embed

            item_category_indices = tf.gather(self.item_category, item_input)
            # Lookup the category bias for the target items
            category_bias = tf.nn.embedding_lookup(self.emb_Q_bias, item_category_indices)
            self.emb_q -= category_bias

            # if adv:  # 是否是对抗训练
            #     gd = tf.nn.embedding_lookup(self.delta, item_input)
            #     self.d = self.epsilon * tf.nn.l2_normalize(gd, 1)
            #     self.watch.append(self.d)
            #     self.emb2_q = self.emb2_q + tf.matmul(self.d, self.phi)

            return tf.reduce_sum(self.emb_p * self.emb_q, 1), self.emb_p, self.emb_q  # 返回的是两个特征的乘机，用户特征，商品特征


    # def _update_feature_matrix(self, sess):
    #     # Update the feature matrix whenever the feature model parameters change
    #     self.extract_features_in_batches(sess, self.val_bsz)

    def _specific_perdiction(self):
        with tf.name_scope("specific_prediction"):
            self.emb_p = tf.nn.embedding_lookup(self.emb_P, self.user_input)

            self.temp_emb_Q = tf.matmul(self.emb_Q_feature, self.phi) + self.emb_Q  # 相当于商品特征加上商品的图像特征


            # Lookup category biases for all items
            category_bias = tf.nn.embedding_lookup(self.emb_Q_bias, self.item_category)

            # Subtract category biases from item embeddings
            self.temp_emb_Q = self.temp_emb_Q - category_bias

            self.predictions_specifically = tf.matmul(self.emb_p, self.temp_emb_Q, transpose_b=True)

    # def _prediction(self):
    #     with tf.name_scope("prediction"):
    #         self.emb_Q = self.feature_model(self.image_data, training=False)
    #         # if self.adv:
    #         #     self.d = self.epsilon * tf.nn.l2_normalize(self.delta, 1)
    #         #     self.watch.append(self.d)
    #         #     self.emb2_Q = self.emb2_Q + tf.matmul(self.d, self.phi)
    #
    #         self.predictions = tf.matmul(self.emb_P * -1, self.emb_Q, transpose_b=True)
    # def _update_feature_matrix_op(self):
    #     """
    #     Extract features for all items in batches using the feature model and store them in a TensorFlow tensor.
    #
    #     :param batch_size: The number of images to process in each batch.
    #     :return: A TensorFlow tensor containing extracted features for all items.
    #     """
    #     num_items = self.num_items
    #     assign_ops = []
    #     # Progress bar for visualization
    #     pbar = tqdm(total=num_items, desc="Extracting Features", unit="item")
    #
    #     # Process images in batches
    #     for start in range(0, num_items, self.val_bsz):
    #         end = min(start + self.val_bsz, num_items)
    #         batch_indices = tf.range(start, end)
    #
    #         # Get the corresponding image paths for the batch
    #         image_paths = tf.gather(self.image_paths, batch_indices)
    #
    #         # Load and preprocess images in the batch
    #         images_batch = tf.map_fn(self._load_and_preprocess_image, image_paths, dtype=tf.float32)
    #
    #         # Run the feature extraction model to get features
    #         features_batch = self.feature_model(images_batch, training=False)
    #
    #         assign_op = tf.compat.v1.assign(self.emb_Q_feature[start:end], features_batch)
    #         assign_ops.append(assign_op)
    #         # sess.run(assign_op)
    #
    #         # Update the progress bar
    #         pbar.update(end - start)
    #     # Close the progress bar
    #     pbar.close()
    #     return tf.group(*assign_ops)

    def _update_feature_matrix_with_while(self):
        start = tf.constant(0)

        def condition(start):
            return tf.less(start, self.num_items)

        def body(start):
            end = tf.minimum(start + self.val_bsz, self.num_items)
            batch_indices = tf.range(start, end)

            image_paths = tf.gather(self.image_paths, batch_indices)
            images_batch = tf.map_fn(self._load_and_preprocess_image, image_paths, dtype=tf.float32)
            features_batch = self.feature_model(images_batch, training=False)

            assign_op = tf.compat.v1.assign(self.emb_Q_feature[start:end], features_batch)

            with tf.control_dependencies([assign_op]):
                return start + self.val_bsz

        # Use tf.while_loop to iterate over batches
        update_op = tf.while_loop(condition, body, [start])
        return update_op

    def _create_loss(self):
        # self.model.train()
        self.pos_pred, emb_p, emb_pos_q = self._create_inference(self.user_input, self.pos_input, self.pos_images_pixel)
        self.neg_pred, _, emb_neg_q = self._create_inference(self.user_input, self.neg_input, self.neg_images_pixel)

        self.result = self.pos_pred - self.neg_pred
        self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

        self.adv_loss = 0

        # if self.adv:
        #     if self.adv_type == 'rand':
        #         self.delta = tf.random.truncated_normal(shape=self.image_feature.shape, mean=0.0, stddev=0.01)
        #     else:
        #         self.delta = tf.gradients(self.loss, [self.image_feature])[0]
        #     self.delta = tf.stop_gradient(self.delta)
        #
        #     self.pos_pred_adv, _, _, _ = self._create_inference(self.user_input, self.pos_input, adv=True)
        #     self.neg_pred_adv, _, _, _ = self._create_inference(self.user_input, self.neg_input, adv=True)
        #
        #     result_adv = self.pos_pred_adv - self.neg_pred_adv
        #     self.adv_loss = tf.reduce_sum(tf.nn.softplus(-result_adv))

        feature_model_weights = tf.concat([tf.reshape(w, [-1]) for w in self.feature_model.trainable_weights], axis=0)
        feature_model_reg = tf.nn.l2_loss(feature_model_weights)
        self.opt_loss = self.loss + self.lmd * self.adv_loss \
                        + self.l1 * (tf.nn.l2_loss(self.emb_p)) \
                        + self.l2 * (tf.nn.l2_loss(emb_pos_q) + tf.nn.l2_loss(emb_neg_q) + tf.nn.l2_loss(self.phi)) \
                        + self.l3 * feature_model_reg \

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):

            vlist = [self.emb_P]  # 用户嵌入
            vlist2 = [self.emb_Q] # 商品的嵌入
            vlist3 = [self.emb_Q_bias] # 类别嵌入
            vlist4 = [self.phi]  # 图像嵌入
            vlist5 = self.feature_model.trainable_weights  # 模型嵌入

            if isinstance(self.lr, list):
                lr = self.lr
            else:
                lr = [self.lr, self.lr, self.lr, self.lr, self.lr]

            opt = tf.compat.v1.train.AdamOptimizer(lr[0])
            opt2 = tf.compat.v1.train.AdamOptimizer(lr[1])
            opt3 = tf.compat.v1.train.AdamOptimizer(lr[2])
            opt4 = tf.compat.v1.train.AdamOptimizer(lr[3])
            opt5 = tf.compat.v1.train.AdamOptimizer(lr[4])

            grads_vlist = tf.gradients(self.opt_loss, vlist)  # Gradients for user embedding variables
            grads_vlist2 = tf.gradients(self.opt_loss, vlist2)  # Gradients for item embedding variables
            grads_vlist3 = tf.gradients(self.opt_loss, vlist3)  # Gradients for item category embedding variables
            grads_vlist4 = tf.gradients(self.opt_loss, vlist4)  # Gradients for item images embedding variables
            grads_vlist5 = tf.gradients(self.opt_loss, vlist5)  # Gradients for feature model

            train_op = opt.apply_gradients(zip(grads_vlist, vlist))
            train_op2 = opt2.apply_gradients(zip(grads_vlist2, vlist2))
            train_op3 = opt3.apply_gradients(zip(grads_vlist3, vlist3))
            train_op4 = opt4.apply_gradients(zip(grads_vlist4, vlist4))
            train_op5 = opt5.apply_gradients(zip(grads_vlist5, vlist5))

            self.optimizer = tf.group(train_op, train_op2, train_op3, train_op4, train_op5)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._load_images_by_indices()
        # self._map_paths_to_images()
        self._specific_perdiction()
        self._create_loss()
        self._create_optimizer()
        self.update_feature_matrix_op = self._update_feature_matrix_with_while()
        # self._prediction()


    def get_saver_name(self):
        return "stored_dvbpr_k_%d_lr_%s_regs_%s_eps_%f_lmd_%f" % \
               (self.emb_K, self.slr, self.regs, self.epsilon, self.lmd)
