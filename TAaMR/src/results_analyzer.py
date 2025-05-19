import utils.read as read
import utils.write as write
from utils.sendmail import sendmail
from utils import get_server_name, cpu_count
from operator import itemgetter

import pandas as pd
import time
import multiprocessing as mp
import os


# Global Configuration
result_dir = '../rec_results/'
dataset_name = 'amazon_men/'
experiment_name = ''
tp_k_predictions = 1000
prediction_files_path = result_dir + dataset_name

K = 100
counter = 0
start_counter = time.time()
start = time.time()


def elaborate(class_frequency, user_id, user_positive_items, sorted_item_predictions):
    # Count the class occurrences for the user: user_id
    k = 0
    for item_index in sorted_item_predictions:  # 这里就是根据top1000类别中判断出来的用户的喜好，然后与这个用户原来的喜好列表1，1对比
        if item_index not in user_positive_items:
            item_original_class = item_classes[item_classes['ImageID'] == item_index]['ClassStr'].values[0]
            class_frequency[item_original_class] += 1
            k += 1
            if k == K:
                break
    if k < K:
        print('User: {0} has more than {1} positive rated items in his/her top K'.format(user_id, K))

    return user_id


def count_elaborated(r):
    global counter, start_counter, users_size
    counter += 1
    if (counter + 1) % 100 == 0:
        print('{0}/{1} in {2}'.format(counter + 1, users_size, time.time() - start_counter))
        start_counter = time.time()


if __name__ == '__main__':

    prediction_files = os.listdir(prediction_files_path)  # 这里存放的是这个数据集下面的存储的不同的攻击方法的检测结果

    with open('results_aw_amr', 'w') as f:
        for prediction_file in prediction_files:
            # if not prediction_file.startswith('Top') and not prediction_file.startswith('Plot'):
            if not prediction_file.startswith('Top') and not prediction_file.startswith('Plot'):
                print(prediction_file)
                for pkl_path_name in os.listdir(os.path.join(prediction_files_path, prediction_file)):
                    predictions = read.load_obj(os.path.join(prediction_files_path, prediction_file, pkl_path_name))

                    pos_elements = pd.read_csv('../data/{0}/pos.txt'.format(dataset_name), sep='\t', header=None)
                    pos_elements.columns = ['u', 'i']  # 'u'表示的是用户, 'i'表示的是商品
                    pos_elements.u = pos_elements.u.astype(int)
                    pos_elements.i = pos_elements.i.astype(int)
                    pos_elements = pos_elements.sort_values(by=['u', 'i'])  # 在偏好商品的txt文件中，根据用户和商品进行排序

                    item_classes = pd.read_csv('../data/{0}/original_images/classes.csv'.format(dataset_name)) # 这里是原来的商品的预测概率

                    # manager = mp.Manager()
                    # class_frequency = manager.dict()
                    class_frequency = dict()
                    for item_class in item_classes['ClassStr'].unique():  # 这里主要做的就是根据原来样本的分布，记录一下原来样本所属的类别有哪些
                        class_frequency[item_class] = 0

                    users_size = len(predictions)  # 这里表示的就是用户的数量

                    # p = mp.Pool(cpu_count()-1)
                    # p = mp.Pool(1)

                    for user_id, sorted_item_predictions in enumerate(predictions):
                        user_positive_items = pos_elements[pos_elements['u'] == user_id]['i'].to_list()  # pos_elements['u'] == user_id，就是热门商品列表中用户u喜好的商品
                        elaborate(class_frequency, user_id, user_positive_items, sorted_item_predictions)
                        count_elaborated(user_id)
                        # p.apply_async(elaborate, args=(class_frequency, user_id, user_positive_items, sorted_item_predictions,),
                        #               callback=count_elaborated)

                    # p.close()
                    # p.join()

                    print('END in {0} - {1}'.format(time.time() - start, max(class_frequency.values())))

                    # We need this operation to use the results in the Manager
                    novel = dict()
                    for key in class_frequency.keys():
                        novel[key] = class_frequency[key]

                    # print(novel.items())

                    N_USERS = pos_elements['u'].nunique()  # Return number of unique elements in the object. 这里是反馈矩阵中用户的个数
                    N = 50  # Top-N classes
                    class_str_length = 10

                    # Store class frequencies results
                    class_frequency_file_name = 'Top{0}/Top{0}_class_frequency_of_'.format(K) + pkl_path_name.split('.')[0]
                    write.save_obj(novel, prediction_files_path + class_frequency_file_name)

                    # 会得到一个top N 的字典，其中包含了商品类型和出现的频率
                    res = dict(sorted(novel.items(), key=itemgetter(1), reverse=True)[:N])  # novel.items()就是这个字典的key:item对，然后itemgetter(1)表示的是按照这个第1列来进行排序，reverse表示的是从高到低

                    res = {str(k): v / users_size for k, v in res.items()}

                    # res = {str(k)[:class_str_length]: v / N_USERS for k, v in res.items()}

                    keys = res.keys()
                    values = res.values()

                    ordered = pd.DataFrame(list(zip(keys, values)), columns=['x', 'y']).sort_values(by=['y'], ascending=False)

                    print('\nExperiment Name: {0}'.format(prediction_file))
                    print(ordered)

                    f.writelines('\nExperiment Name: {0}'.format(prediction_file))
                    f.writelines(ordered.to_string())

    # sendmail('Elaborate Predictions on {0}'.format(get_server_name()), 'Amazon Women')

