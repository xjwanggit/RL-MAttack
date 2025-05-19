import numpy as np
import random
import pandas as pd
from typing import List, Optional, Tuple, Type, Union, cast


class DeepVersion_Dataset:
    def __init__(self, args):
        path = '../data/' + args.dataset + '/'
        self.f_pos = path + 'pos.txt'
        self.bsz = args.batch_size
        self.val_bsz = args.validation_batch_size
        self.fsz = 4096
        self.pos = np.loadtxt(self.f_pos, dtype=np.int)
        self.usz, self.isz = np.max(self.pos, 0) + 1  # 这个usz其实表示的是用户，isz应该表示的是item，在这个pos中第一列表示用户列，第二列表示的是物品列，分别取了这两列的最大值
        self.pos_elements = pd.read_csv(self.f_pos, sep='\t', header=None)
        self.pos_elements.columns = ['u', 'i']
        self.pos_elements.u = self.pos_elements.u.astype(int)
        self.pos_elements.i = self.pos_elements.i.astype(int)
        # self.coldstart = set(self.neg[:,0].tolist()) - set(self.pos[:,1].tolist())
        self.coldstart = set(range(0, self.isz)) - set(self.pos[:, 1].tolist())  # 不在交互列表中就表示的是冷启动,即这个商品不存在于交互记录中，两个set相减如a-b就是从a中删除b中的值
        self.pos = list(self.pos)

        self.inter = {}
        for u, i in self.pos:  # 这里就是把pos（pos代表的是用户和商品之间的交互记录）中的数据，存入到字典类型即 用户：交互商品1，交互商品2
            if u not in self.inter:
                self.inter[u] = set([])
            self.inter[u].add(i)

        self.item_image_lists = [f"'../../../data/{args.dataset}/{args.experiment_name}/images/{item}.jpg" for item in range(self.isz)]


        if args.filter_cold_users:
            self.userID_to_index = {}
            filtered_inter = {user: items for user, items in self.inter.items() if len(items) >= 5}
            print(f"{len(filtered_inter.keys()) - self.usz}名用户被过滤掉！")
            self.userID_to_index = {user: idx for idx, user in enumerate(filtered_inter)}
            modified_filtered_inter = {
                self.userID_to_index[user]: items for user, items in filtered_inter.items()
            }
            self.inter = modified_filtered_inter
        # for user, item in self.inter.items():
        #     if len(item) < 5:
        #         del self.inter[user]
        #     self.userID_to_index[user] = count_number
        #     count_number += 1
        self.training_dataset, self.validation_dataset, self.evaluation_dataset = self.split_dataset(self.inter)

    @classmethod
    def split_dataset(
            cls: Type['DeepVersion_Dataset'],
            interactions: dict
    ) -> Tuple[dict, dict, dict]:
        """
        Splits the dataset into training, validation, and evaluation sets.

        :param interactions: A dictionary of user interactions.
        :return: A tuple containing training, validation, and evaluation datasets.
        """
        # Implement the actual logic for splitting the dataset
        # This is a placeholder implementation; you need to adjust it to your needs
        training_dataset = {}
        validation_dataset = {}
        evaluation_dataset = {}
        for user, items in interactions.items():
            validation_item, evaluation_item = random.sample(items, 2)
            remaining_items = set(items) - {validation_item, evaluation_item}

            training_dataset[user] = list(remaining_items)
            validation_dataset[user] = validation_item
            evaluation_dataset[user] = evaluation_item
        return training_dataset, validation_dataset, evaluation_dataset




    def shuffle(self):
        random.shuffle(self.pos)

    def shuffle_dict(self):
        items = list(self.mode_dataset.items())
        random.shuffle(items)
        shuffled_dataset = dict(items)
        self.mode_dataset = shuffled_dataset


    def sample(self, p):
        u, i = self.pos[p]
        i_neg = i
        while i_neg in self.inter[u] or i_neg in self.coldstart:  # remove the cold start items from negative samples 这个neg_item即不在冷启动也不在用户的交互列表中
            i_neg = random.randrange(self.isz)  # 可以用于生成随机范围内的一个整数
        return u, i, i_neg

    def sample_train(self, user):
        items = self.mode_dataset[user]
        i_pos = random.sample(items, 1)
        i_neg = random.randrange(self.isz)
        while i_neg in self.inter[user] or i_neg in self.coldstart:
            i_neg = random.randrange(self.isz)
        return user, i_pos, i_neg

    def sample_evaluation(self, user):
        i_pos = self.mode_dataset[user]
        return user, i_pos

    def batch_generator(self, mode):
        if mode == 'Train':
            self.mode_dataset = self.training_dataset
        elif mode == 'Validation':
            self.mode_dataset = self.validation_dataset
        elif mode == 'Test':
            self.mode_dataset = self.evaluation_dataset
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'Train', 'Validation', or 'Test'.")
        self.shuffle_dict()  # 相当于把交互记录打乱
        # sz = len(self.mode_dataset)//self.bsz*self.bsz  # 这里就相当于进行了一个取整操作，使得sz正好是batch_size的整数倍

        users = list(self.mode_dataset.keys())
        if mode == 'Train':
            num_batches = len(users) // self.bsz + (1 if len(users) % self.bsz != 0 else 0)
        else:
            num_batches = len(users) // self.val_bsz + (1 if len(users) % self.val_bsz != 0 else 0)
        for st in range(num_batches):
            if mode == 'Train':
                batch_users = users[st * self.bsz:(st + 1) * self.bsz]
            else:
                batch_users = users[st * self.val_bsz:(st + 1) * self.val_bsz]
            # map(function, iterable)
            if mode == 'Train':
                samples = zip(*map(self.sample_train, batch_users))  # map函数对可迭代对象进行指定的function操作 zip就可以将u集中到一起，将i也集中到一起，将i_neg集中到一起
            elif mode == 'Validation' or mode == 'Test':
                samples = zip(*map(self.sample_evaluation, batch_users))
            else:
                raise ValueError(f"Unsupported mode '{mode}'. Expected 'Train', 'Validation', or 'Test'.")
            if mode == 'Train':
                user_ids, pos_items, neg_items = map(np.array, samples)

                user_ids = np.squeeze(user_ids)
                pos_items = np.squeeze(pos_items)
                neg_items = np.squeeze(neg_items)

                yield user_ids, pos_items, neg_items
            else:
                user_ids, pos_items = map(np.array, samples)

                user_ids = np.squeeze(user_ids)
                pos_items = np.squeeze(pos_items)

                yield user_ids, pos_items

            # yield map(np.array, samples)

    def test_generator(self):
        for u in range(0, self.usz):
            pos_items = self.pos_elements[self.pos_elements['u'] == u]['i'].tolist()
            neg_samples = list(set(range(self.isz)).difference(pos_items))
            samples = zip(*[(u, i) for i in neg_samples])
            yield map(np.array, samples)

