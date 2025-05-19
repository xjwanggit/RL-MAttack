import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
class VBPR:

    def __init__(self, args, num_users, num_items, num_image_feature):
        self.emb_K = args.emb1_K  # emb1_K表示的是嵌入的特征量，这里是64
        self.lr = eval(args.lr)  # 这个是学习率
        self.slr = args.lr

        self.regs = args.regs  # lambdas for regularization
        regs = eval(self.regs)
        self.l1 = regs[0]
        self.l2 = regs[1]
        self.l3 = regs[2]
        self.lmd = args.lmd  # 用于平衡鲁棒性与性能
        self.adv = args.adv
        self.adv_type = args.adv_type
        self.epsilon = args.epsilon  # 扰动大小
        self.num_users = num_users
        self.num_items = num_items
        self.num_image_feature = num_image_feature  # 这里表示的是图像的特征量，这里是2048
        self.watch = []
        self.build_graph()

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="user_input")
            self.pos_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pos_input")
            self.neg_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="neg_input")

            self.init_image = tf.compat.v1.placeholder(tf.float32, shape=[self.num_items, self.num_image_feature],  # 初始图像的特征，行为物品的数量，列为图像特征 (82630, 2048)
                                                       name="pos_image")
            self.init_emb_P = tf.compat.v1.placeholder(tf.float32, shape=[self.num_users, self.emb_K],  # 用户嵌入的特征，size为（用户的数量，特征向量）（26155，64）
                                                       name="init_emb_P")
            self.init_emb_Q = tf.compat.v1.placeholder(tf.float32, shape=[self.num_items, self.emb_K],  # 物品嵌入的特征，size为（物品的数量，特征向量）（82630，64）
                                                       name="init_emb_Q")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.emb_P = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_users, self.emb_K], mean=0.0, stddev=0.01),
                name='emb_P', dtype=tf.float32)  # (users, embedding_size)
            self.emb_Q = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_items, self.emb_K], mean=0.0, stddev=0.01),
                name='emb_Q', dtype=tf.float32)  # (items, embedding_size)

        with tf.name_scope("feature"):
            self.image_feature = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_items, self.num_image_feature], mean=0.0, stddev=0.01),
                name='image_feature', dtype=tf.float32, trainable=False)  # (items, embedding_size)

        with tf.name_scope("init_op"):
            self.assign_image = tf.compat.v1.assign(self.image_feature, self.init_image)  # tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number
            self.assign_P = tf.compat.v1.assign(self.emb_P, self.init_emb_P)
            self.assign_Q = tf.compat.v1.assign(self.emb_Q, self.init_emb_Q)

        with tf.name_scope("image_transfer"):
            self.phi = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_image_feature, self.emb_K], mean=0.0, stddev=0.01),
                name='phi', dtype=tf.float32)  # 这里相当于把图像特征的2048转译成64

    def _create_inference(self, user_input, item_input, adv=False):
        with tf.name_scope("inference"):
            self.emb_p = tf.nn.embedding_lookup(self.emb_P, user_input)  # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
            self.emb1_q = tf.nn.embedding_lookup(self.emb_Q, item_input)
            image_input = tf.nn.embedding_lookup(self.image_feature, item_input)
            self.emb2_q = tf.matmul(image_input, self.phi)  # 对图像特征进行一个转换
            if adv: # 是否是对抗训练
                gd = tf.nn.embedding_lookup(self.delta, item_input)
                self.d = self.epsilon * tf.nn.l2_normalize(gd, 1)
                self.watch.append(self.d)
                self.emb2_q = self.emb2_q + tf.matmul(self.d, self.phi)

            self.emb_q = self.emb1_q + self.emb2_q  # 这个就是物品的商品特征加上物品的图像特征

            return tf.reduce_sum(self.emb_p * self.emb_q, 1), self.emb_p, self.emb_q, image_input # 返回的是两个特征的乘机，用户特征，商品特征，图像的原始特征


    def _specific_perdiction(self):
        with tf.name_scope("specific_prediction"):
            self.predictions_specifically = tf.matmul(self.emb_p, self.emb_q, transpose_b=True)


    def _prediction(self):
        with tf.name_scope("prediction"):
            self.emb2_Q = tf.matmul(self.image_feature, self.phi)
            if self.adv:
                self.d = self.epsilon * tf.nn.l2_normalize(self.delta, 1)
                self.watch.append(self.d)
                self.emb2_Q = self.emb2_Q + tf.matmul(self.d, self.phi)

            self.temp_emb_Q = self.emb_Q + self.emb2_Q  # 相当于商品特征加上商品的图像特征

            self.predictions = tf.matmul(self.emb_P * -1, self.temp_emb_Q, transpose_b=True)

    def _create_loss(self):
        self.pos_pred, emb_p, emb_pos_q, self.emb_pos_feature = self._create_inference(self.user_input, self.pos_input)
        self.neg_pred, _, emb_neg_q, _ = self._create_inference(self.user_input, self.neg_input)

        self.result = self.pos_pred - self.neg_pred
        self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))
        # self.loss = tf.reduce_sum(-tf.math.log(tf.nn.sigmoid(self.result)))
        self.adv_loss = 0

        if self.adv:
            if self.adv_type == 'rand':
                self.delta = tf.random.truncated_normal(shape=self.image_feature.shape, mean=0.0, stddev=0.01)
            else:
                self.delta = tf.gradients(self.loss, [self.image_feature])[0]
            self.delta = tf.stop_gradient(self.delta)

            self.pos_pred_adv, _, _, _ = self._create_inference(self.user_input, self.pos_input, adv=True)
            self.neg_pred_adv, _, _, _ = self._create_inference(self.user_input, self.neg_input, adv=True)

            result_adv = self.pos_pred_adv - self.neg_pred_adv
            self.adv_loss = tf.reduce_sum(tf.nn.softplus(-result_adv))

        self.opt_loss = self.loss + self.lmd * self.adv_loss \
                        + self.l1 * (tf.reduce_sum(self.emb_p)) \
                        + self.l2 * (tf.reduce_sum(emb_pos_q) + tf.reduce_sum(emb_neg_q)) \
                        + self.l3 * (tf.reduce_sum(self.phi)) # 后面的相当于是正则化

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):

            vlist = [self.emb_P]  # 用户嵌入
            vlist2 = [self.emb_Q]  # 商品嵌入
            vlist3 = [self.phi]  # 图像嵌入

            if isinstance(self.lr, list):
                lr = self.lr
            else:
                lr = [self.lr, self.lr, self.lr]

            # opt = tf.compat.v1.train.AdagradOptimizer(lr[0])
            # opt2 = tf.compat.v1.train.AdagradOptimizer(lr[1])
            # opt3 = tf.compat.v1.train.AdagradOptimizer(lr[2])
            opt = tf.compat.v1.train.AdamOptimizer(lr[0])
            opt2 = tf.compat.v1.train.AdamOptimizer(lr[1])
            opt3 = tf.compat.v1.train.AdamOptimizer(lr[2])
            grads_all = tf.gradients(self.opt_loss, vlist + vlist2 + vlist3)  # 对于列表，“+”号表示两个列表的合并，实现第一项对第二项的求导
            grads = grads_all[0:1]
            grads2 = grads_all[1:2]
            grads3 = grads_all[2:3]
            train_op = opt.apply_gradients(zip(grads, vlist))
            train_op2 = opt2.apply_gradients(zip(grads2, vlist2))
            train_op3 = opt3.apply_gradients(zip(grads3, vlist3))

            self.optimizer = tf.group(train_op, train_op2, train_op3)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._prediction()
        self._specific_perdiction()


    def get_saver_name(self):
        return "stored_vbpr_k_%d_lr_%s_regs_%s_eps_%f_lmd_%f" % \
               (self.emb_K, self.slr, self.regs, self.epsilon, self.lmd)
