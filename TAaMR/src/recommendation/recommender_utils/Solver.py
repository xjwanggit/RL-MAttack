import math
import time
import pandas as pd
import utils.read as read
import utils.write as write
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append('/home/lxh/wangchengyi/TAaMR-master/src')
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from cnn.tensorflow_models.model import *
from utils.read import *

from recommendation.recommender_models.VBPR import VBPR
from recommendation.recommender_models.DVBPR import DVBPR
from recommendation.recommender_models.DeepStyle import DeepStyle
from recommendation.recommender_models.AMR import AMR
from recommendation.recommender_dataset.Dataset import Dataset
from recommendation.recommender_dataset.DeepVersion_Dataset import DeepVersion_Dataset



class Solver:
    def __init__(self, args):
        self.model_name = args.model
        self.bsz = args.batch_size
        if self.model_name == 'DVBPR' or self.model_name == 'DeepStyle':
            self.dataset = DeepVersion_Dataset(args)
        else:
            self.dataset = Dataset(args)
        self.dataset_name = args.dataset
        self.experiment_name = args.experiment_name
        self.adv = args.adv
        self.adv_type = args.adv_type
        if args.model == 'AMR':
            self.model = AMR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
        elif args.model == 'VBPR':
            self.model = VBPR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)  # usz表示的是用户的数量，isz表示的是商品的数量，fsz表示的是特征量
        elif args.model == 'DVBPR':
            self.model = DVBPR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
        elif args.model == 'DeepStyle':
            self.model = DeepStyle(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
        self.epoch = args.epoch
        self.verbose = args.verbose

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=0)
        if self.model_name == 'VBPR' or self.model_name == 'AMR':
            self.sess.run(self.model.assign_image, feed_dict={self.model.init_image: self.dataset.emb_image})

        self.tp_k_predictions = args.tp_k_predictions  # top k predictions to store before the evaluation
        self.weight_dir = '../' + args.weight_dir + '/'  # 保存的位置为../rec_model_weights/
        self.result_dir = '../' + args.result_dir + '/'  # 保存的位置为../rec_results/

        if self.experiment_name == 'original_images':
            self.attack_type = 'original_images'
            self.attacked_categories = ''
            self.eps_cnn = ''
            self.iteration_attack_type = ''
            self.norm = ''
        else:
            self.attack_type = self.experiment_name.split('_')[0]
            self.attacked_categories = '_' + self.experiment_name.split('_')[1] + '_' + self.experiment_name.split('_')[
                2]
            self.eps_cnn = '_' + self.experiment_name.split('_')[3]
            self.iteration_attack_type = '_' + self.experiment_name.split('_')[4]
            self.norm = '_' + self.experiment_name.split('_')[5]

        self.experiment_name = '{0}/{1}'.format(self.dataset_name, self.experiment_name)

        if self.adv:
            self.load('best')


    def one_epoch(self):
        generator = self.dataset.batch_generator(mode='Train')
        api = [self.model.user_input, self.model.pos_input, self.model.neg_input]
        loss_lis = []
        correct_lis = []
        training_phase_pbar = tqdm(
            range((len(self.dataset.training_dataset) - 1) // self.bsz + 1),
            desc="Batches",
            unit="batch",
        )
        training_phase_pbar.reset()
        while True:
            try:
                feed_dict = dict(zip(api, next(generator)))
                loss, judge_result = self.sess.run([self.model.loss, self.model.result], feed_dict=feed_dict)
                self.sess.run([self.model.optimizer], feed_dict=feed_dict)
                loss_lis.append(round(loss/len(feed_dict[self.model.user_input]), 4))
                correct_lis.append((judge_result > 0).sum())
                training_phase_pbar.update(1)
            except StopIteration:
                break
        training_phase_pbar.close()
        data['loss_value'].append(round(np.array(loss_lis).mean(), 4))
        data['acc'].append(round(np.array(correct_lis).sum() / len(self.dataset.training_dataset), 4))

    def train(self):
        global data
        data = {'loss_value': [], 'acc': [], 'auc_validation': [], 'auc_test': []}
        start_epoch = 0
        best_auc = 0.0
        best_epoch = 0
        if self.adv:
            self.epoch = self.epoch // 2

        epoch_pbar = tqdm(
            range(start_epoch + 1, self.epoch + 1),
            desc="Epochs",
            unit="epoch",
            ncols=180,
            postfix=dict(time=None, best_auc=None, best_epoch=None, loss=None, acc=None, auc=None),
        )
        if self.model_name == 'DVBPR' or self.model_name == 'DeepStyle':
            count_number = 0
            for i in epoch_pbar:
                start = time.time()
                self.one_epoch()
                current_acc = data['acc'][-1]
                max_acc = max(data['acc'][:-1], default=1e-3)
                if (current_acc - max_acc)/max_acc < 0.01 or len(data['acc']) == 1:
                    auc = self.evaluation('validation')
                    if auc > best_auc:
                        best_epoch = i
                        best_auc = auc
                        epoch_pbar.set_postfix(
                            acc=f"{data['acc'][i - 1]:.2%}", loss=f"{data['loss_value'][i - 1]:.4f}", auc=f"{auc:.4f}",
                            best_epoch=best_epoch, best_auc=f"{data['auc_validation'][best_epoch - 1]:.4f}",
                            time=f"{time.time() - start:.2f}secs"
                        )
                        count_number = 0
                        if self.model_name == 'DVBPR':
                            self.save_user_embeddings_DVBPR('best')
                        elif self.model_name == 'DeepStyle':
                            self.save_user_embeddings_DeepStyle('best')
                    else:
                        count_number += 1
                        epoch_pbar.set_postfix(
                            acc=f"{data['acc'][i - 1]:.2%}", loss=f"{data['loss_value'][i - 1]:.4f}", auc=f"{auc:.4f}",
                            time=f"{time.time() - start:.2f}secs"
                        )
                else:
                    epoch_pbar.set_postfix(
                        acc=f"{data['acc'][i - 1]:.2%}", loss=f"{data['loss_value'][i - 1]:.4f}", best_auc=f"{data['auc_validation'][best_epoch - 1]:.4f}",
                        time=f"{time.time() - start:.2f}secs"
                    )
                    data['auc_validation'].append(None)

                if count_number > 5 and self.model_name != 'DeepStyle':
                    print("Overfitted maybe...")
                    break
                elif self.model_name == 'DeepStyle' and count_number > 2:
                    print("Overfitted maybe...")
                    break
                data['auc_test'].append(None)
                self.save_training_process(data)
        else:
            for i in epoch_pbar:
                start = time.time()
                self.one_epoch()
                auc = self.evaluation('validation')
                if auc > best_auc and self.model_name != 'AMR':
                    best_epoch = i
                    best_auc = auc
                    epoch_pbar.set_postfix(
                        acc=f"{data['acc'][i - 1]:.2%}", loss=f"{data['loss_value'][i - 1]:.4f}", auc=f"{auc:.4f}", best_epoch=best_epoch, best_auc=f"{data['auc_validation'][best_epoch - 1]:.4f}", time=f"{time.time()-start:.2f}secs"
                    )
                    self.save('best')
                elif self.model_name == 'AMR' and auc > best_auc and i > 300:
                    best_epoch = i
                    best_auc = auc
                    epoch_pbar.set_postfix(
                        acc=f"{data['acc'][i - 1]:.2%}", loss=f"{data['loss_value'][i - 1]:.4f}", auc=f"{auc:.4f}", best_epoch=best_epoch, best_auc=f"{data['auc_validation'][best_epoch - 1]:.4f}", time=f"{time.time()-start:.2f}secs"
                    )
                    self.save('best')
                elif self.model_name == 'AMR' and i < 300:
                    best_epoch = i
                    epoch_pbar.set_postfix(
                        acc=f"{data['acc'][i - 1]:.2%}", loss=f"{data['loss_value'][i - 1]:.4f}", auc=f"{auc:.4f}", time=f"{time.time()-start:.2f}secs"
                    )
                else:
                    epoch_pbar.set_postfix(
                        acc=f"{data['acc'][i - 1]:.2%}", loss=f"{data['loss_value'][i - 1]:.4f}", auc=f"{auc:.4f}", time=f"{time.time()-start:.2f}secs"
                    )

                if i >= (best_epoch + 20):
                    print("Overfitted maybe...")
                    break
                data['auc_test'].append(None)
                self.save_training_process(data)
        epoch_pbar.refresh()
        epoch_pbar.close()
        auc_eval = self.evaluation('test')
        print(f"The Best [Validation] AUC = {data['auc_validation'][best_epoch - 1]:.4f} (best epoch = {best_epoch})")
        print(f"[Evaluation] AUC = {auc_eval:.4f} (All Items)")


        max_length = len(data['loss_value'])
        data['auc_test'] = [None] * (max_length - len(data['auc_test'])) + data['auc_test']

        df = pd.DataFrame(data)
        file_path = os.path.join('../rec_model_logs/', self.experiment_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, f'Best_epoch{best_epoch}_Total_Epoch{self.epoch}_{self.model_name}.csv'))

        # self.store_predictions(i)

    def save_training_process(self, training_data):
        df = pd.DataFrame(training_data)
        file_path = os.path.join('../rec_model_logs/', self.experiment_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if self.adv:
            df.to_csv(os.path.join(file_path, f'Epoch_Total_Epoch{self.epoch}_{self.model_name}_ARM.csv'))
        else:
            df.to_csv(os.path.join(file_path, f'Epoch_Total_Epoch{self.epoch}_{self.model_name}.csv'))



    def evaluation(self, mode):
        if mode == 'validation':
            generator = self.dataset.batch_generator(mode='Validation')
        elif mode == 'test':
            generator = self.dataset.batch_generator(mode='Test')
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'Validation', or 'Test'.")
        api = [self.model.user_input, self.model.pos_input, self.model.neg_input]
        AUC_eval = np.zeros(self.dataset.usz)
        user_neg_items = np.array(range(self.dataset.isz))
        if self.model_name == 'DVBPR' or self.model_name == 'DeepStyle':
            self.sess.run(self.model.update_feature_matrix_op)
            eval_all_pbar = tqdm(
                desc="Evaluation (All Items)",
                total=len(self.dataset.validation_dataset) // self.dataset.val_bsz + 1,
                unit="batch",
                postfix=dict(auc=None),
            )
        else:
            eval_all_pbar = tqdm(
                desc="Evaluation (All Items)",
                total=len(self.dataset.validation_dataset) // self.dataset.bsz + 1,
                unit="batch",
                postfix=dict(auc=None),
            )

        while True:
            """
            this is the version that deals with one user each time
            """
            # try:
            #     user_pos = dict(zip(api[:2], next(generator)))
            #     pos_rate = self.sess.run([self.model.pos_pred], feed_dict=user_pos)[0]
            #     user_neg = {}
            #     for index, user in enumerate(user_pos[self.model.user_input]):
            #         correct_number = 0
            #         left_out_items = list(self.dataset.inter[user])
            #         max_possible = self.dataset.isz - len(left_out_items)
            #         user_neg[self.model.user_input] = np.array([user])
            #         # user_neg[self.model.user_input] = np.array(user).repeat(len(user_neg_items))
            #         user_neg[self.model.neg_input] = user_neg_items
            #         neg_rate = self.sess.run([self.model.predictions_specifically], feed_dict=user_neg)[0].squeeze()
            #         # neg_rate = self.sess.run([self.model.neg_pred], feed_dict=user_neg)[0]
            #         correct_number += (pos_rate[index] > neg_rate).sum() - (pos_rate[index] > neg_rate[left_out_items]).sum()
            #         AUC_eval[user] = correct_number / max_possible if max_possible > 0 else 0
            #
            #     eval_all_pbar.update(1)
            # except StopIteration:
            #     break

            """
            this is the version that deals with all users each time
            """
            try:
                user_pos = dict(zip(api[:2], next(generator)))
                pos_rate = self.sess.run([self.model.pos_pred], feed_dict=user_pos)[0]
                # user_neg = {}
                #
                # user_ids = user_pos[self.model.user_input]  # Example: numpy array of user IDs
                #
                # # Initialize an empty list to store left-out items for each user
                # left_out_items_list = []
                # max_possible_items = []
                #
                # # Iterate through each user ID in the array
                # for user in user_ids:
                #     # Fetch the left-out items for the current user from the dictionary
                #     left_out_items = list(
                #         self.dataset.inter.get(user, []))  # Use .get() to handle missing keys gracefully
                #     left_out_items_list.append(left_out_items)
                #     max_possible_items.append(self.dataset.isz - len(left_out_items))
                #
                # user_neg[self.model.user_input] = user_ids
                # user_neg[self.model.neg_input] = user_neg_items
                # neg_rate = self.sess.run([self.model.predictions_specifically], feed_dict=user_neg)[0]
                # correct_number_list = []
                # for index, pos_item in enumerate(left_out_items_list):
                #     correct_number = (pos_rate[index] > neg_rate[index]).sum() - (pos_rate[index] > neg_rate[index, np.array(pos_item)]).sum()
                #     correct_number_list.append(correct_number)
                # AUC_eval[user_ids] = np.array(correct_number_list) / np.array(max_possible_items)

                # this is the second polished version
                # Fetch user IDs and corresponding left-out items in a vectorized manner
                user_ids = user_pos[self.model.user_input]
                left_out_items_list = [list(self.dataset.inter.get(user, [])) for user in user_ids]
                max_possible_items = [self.dataset.isz - len(items) for items in left_out_items_list]

                # Prepare negative inputs for TensorFlow session
                user_neg = {
                    self.model.user_input: user_ids,
                    self.model.neg_input: user_neg_items
                }

                # Execute TensorFlow session in one call for efficiency
                neg_rate = self.sess.run([self.model.predictions_specifically], feed_dict=user_neg)[0]

                # Calculate correct numbers and AUC values in a vectorized manner
                correct_number_list = [
                    (pos_rate[index] > neg_rate[index]).sum() - (
                                pos_rate[index] > neg_rate[index, np.array(pos_items)]).sum()
                    for index, pos_items in enumerate(left_out_items_list)
                ]
                AUC_eval[user_ids] = np.array(correct_number_list) / np.maximum(1, np.array(max_possible_items))

                eval_all_pbar.update(1)
            except StopIteration:
                break
        auc = round(AUC_eval.mean(), 4)
        if mode == 'validation':
            data['auc_validation'].append(auc)
        elif mode == 'test':
            data['auc_test'].append(auc)
        eval_all_pbar.set_postfix(auc=auc)
        eval_all_pbar.refresh()
        eval_all_pbar.close()
        return auc
        # feed_dict[self.model.neg_input] = np.array(range(self.dataset.isz))
        # pos_rate, neg_rate = self.sess.run([self.model.pos_pred, self.model.neg_pred], feed_dict=feed_dict)





    def evaluate_rec_metrics(self, para):
        r, K = para
        hr = 1 if r < K else 0
        if hr:
            ndcg = math.log(2) / math.log(r + 2)
        else:
            ndcg = 0
        return hr, ndcg

    def original_test(self, message):
        st = time.time()
        generator = self.dataset.test_generator()
        api = [self.model.user_input, self.model.pos_input]
        d = []
        i = 0
        start = time.time()
        while True:
            try:
                feed_dict = dict(zip(api, next(generator)))
                preds = self.sess.run(self.model.pos_pred, feed_dict=feed_dict)

                rank = np.sum(preds[1:] >= preds[0])
                d.append(rank)

                i += 1
                if i % 1000 == 0:
                    print("Tested {0}/{1} in {2}".format(i, self.dataset.usz, time.time() - start))
                    start = time.time()
                    break
            except Exception as e:
                # print type(e), e.message
                break
        score5 = np.mean([ele for ele in map(self.evaluate_rec_metrics, zip(d, [5] * len(d)))], 0)
        score10 = np.mean([ele for ele in map(self.evaluate_rec_metrics, zip(d, [10] * len(d)))], 0)
        score20 = np.mean([ele for ele in map(self.evaluate_rec_metrics, zip(d, [20] * len(d)))], 0)

        print(message, score5, score10, score20)
        print('evaluation cost', time.time() - st)

    def store_predictions(self, epoch):
        # We multiply the users embeddings by -1 to have the np sorting operation in the correct order

        print('Start Store Predictions at epoch {0}'.format(epoch))
        start = time.time()
        # predictions = self.sess.run(self.model.predictions)
        emb_P = self.sess.run(self.model.emb_P) * -1  # [用户数量， 用户特征]
        if self.model_name != 'DVBPR' and self.model_name != 'DeepStyle':
            temp_emb_Q = self.sess.run(self.model.temp_emb_Q)  # 图像的商品特征
        else:
            self.sess.run(self.model.update_feature_matrix_op)
            temp_emb_Q = self.sess.run(self.model.emb_Q)
        predictions = np.matmul(emb_P, temp_emb_Q.transpose())
        predictions = predictions.argsort(axis=1)  # np.argsort(), Returns the indices that would sort this array. 从小到大排列
        predictions = [predictions[i][:self.tp_k_predictions] for i in range(predictions.shape[0])]
        file_path = self.result_dir + self.experiment_name
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        prediction_name = os.path.join(file_path, 'top{0}_predictions_epoch{1}_{2}'.format(self.tp_k_predictions, epoch, self.model_name))

        # prediction_name = self.result_dir + self.experiment_name + 'top{0}_predictions_epoch{1}'.format(
        #     self.tp_k_predictions, epoch)
        if self.adv:
            prediction_name = prediction_name + '_AMR'

        write.save_obj(predictions, prediction_name)

        print('End Store Predictions {0}'.format(time.time() - start))

    def load(self, epoch):
        try:
            if self.adv and self.model_name == 'AMR':
                weight_path = os.path.join(self.weight_dir + self.experiment_name, '{0}_epoch_{1}.npy'.format('VBPR', epoch))
            elif self.model_name == 'AMR':
                weight_path = os.path.join(self.weight_dir + self.experiment_name, '{0}_epoch_{1}_AMR.npy'.format(self.model_name, epoch))
            else:
                weight_path = os.path.join(self.weight_dir + self.experiment_name, '{0}_epoch_{1}.npy'.format(self.model_name, epoch))
            params = np.load(weight_path, allow_pickle=True)
            self.sess.run([self.model.assign_P, self.model.assign_Q, self.model.phi.assign(params[2])],
                          {self.model.init_emb_P: params[0], self.model.init_emb_Q: params[1]})  # 这里的phi是用于转换的, P是用户，Q是商品特征
            print('Load parameters from {0}'.format(weight_path))
        except Exception as ex:
            print('Start new model from scratch')

    def save(self, step):
        params = self.sess.run(tf.compat.v1.trainable_variables())
        file_path = self.weight_dir + self.experiment_name
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        store_model_path = os.path.join(file_path, '{0}_epoch_{1}'.format(self.model_name, step))
        # store_model_path = self.weight_dir + self.experiment_name + 'epoch{0}'.format(step)
        if self.adv:
            if self.adv_type == 'rand':
                store_model_path = store_model_path + '_RAND'
            else:
                store_model_path = store_model_path + '_AMR'

        np.save(store_model_path + '.npy', params)

    def save_user_embeddings_DVBPR(self, step):
        file_path = self.weight_dir + self.experiment_name
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Extract and save the user embedding matrix
        user_embedding_matrix = self.sess.run(self.model.emb_P)
        embedding_path = os.path.join(file_path, f'{self.model_name}_user_embedding_matrix_epoch_{step}.npy')
        np.save(embedding_path, user_embedding_matrix)
        print(f"User embedding matrix saved to {embedding_path}")

        # Optionally save the feature model if needed
        feature_model_path = os.path.join(file_path, f'{self.model_name}_feature_model_epoch_{step}.h5')
        self.model.feature_model.save(feature_model_path)
        print(f"Feature model saved to {feature_model_path}")

    def save_user_embeddings_DeepStyle(self, step):
        file_path = self.weight_dir + self.experiment_name
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Extract the embeddings from the TensorFlow session
        user_embedding_matrix, item_embedding_matrix, category_embedding_matrix, phi_matrix = self.sess.run(
            [self.model.emb_P, self.model.emb_Q, self.model.emb_Q_bias, self.model.phi]
        )

        # Save all embeddings into a single .npz file
        embedding_path = os.path.join(file_path, f'{self.model_name}_embeddings_epoch_{step}.npz')
        np.savez(
            embedding_path,
            user_embeddings=user_embedding_matrix,
            item_embeddings=item_embedding_matrix,
            category_embeddings=category_embedding_matrix,
            phi_matrix=phi_matrix
        )
        print(f"User, item, and category embeddings saved to {embedding_path}")

        # Optionally save the feature model if needed
        feature_model_path = os.path.join(file_path, f'{self.model_name}_feature_model_epoch_{step}.h5')
        self.model.feature_model.save(feature_model_path)
        print(f"Feature model saved to {feature_model_path}")
