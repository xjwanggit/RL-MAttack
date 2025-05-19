import os.path
import pandas as pd
import numpy as np
import torch
import random
import tensorflow as tf
from collections.abc import Iterable
from tensorflow.keras.models import load_model
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def split_number(number, parts):
    result = []
    for _ in range(parts - 1):
        result.append(number // parts)
    # print(sum(result))
    result.append(number - sum(result))
    return result

def user_preference(pos_file_txt):
    global inter, usz, isz
    pos = np.loadtxt(pos_file_txt, dtype=int)
    # Number of users, Number of items
    usz, isz = np.max(pos, 0) + 1
    pos_elements = pd.read_csv(pos_file_txt, sep='\t', header=None)
    pos_elements.columns = ['u', 'i']
    pos_elements.u = pos_elements.u.astype(int)
    pos_elements.i = pos_elements.i.astype(int)
    coldstart = set(range(0, isz)) - set(pos[:, 1].tolist())
    pos = list(pos)
    inter = {}
    for u, i in pos:  # 这里就是把pos（pos代表的是用户和商品之间的交互记录）中的数据，存入到字典类型即 用户：交互商品1，交互商品2
        if u not in inter:
            inter[u] = set([])
        inter[u].add(i)


def preprocess_images(image_file_paths):
    # Define the image transformation for your model (resize, normalize, etc.)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of the model
        transforms.ToTensor(),  # Convert to tensor (this gives shape (C, H, W))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    image_tensors = []
    if isinstance(image_file_paths, Iterable):
        for image_path in image_file_paths:
            img = Image.open(image_path).convert('RGB')
            img_tensor = preprocess(img)
            image_tensors.append(img_tensor)
        # Stack all image tensors to create a batch (batch_size, C, H, W)
        batched_images = torch.stack(image_tensors)
        return batched_images
    else:
        img = Image.open(image_file_paths).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        return img_tensor


def extract_image_features(image_file_paths, feature_model):
    # Load and preprocess images
    batched_images = preprocess_images(image_file_paths)

    # Convert PyTorch tensor (batch_size, C, H, W) -> TensorFlow format (batch_size, H, W, C)
    batched_images = batched_images.permute(0, 2, 3, 1).numpy()  # Rearrange to (batch_size, H, W, C)

    # Extract image embeddings from the feature model in batches
    image_features = feature_model.predict(batched_images)

    return image_features

def generate_target(running_times, user_number, total_samples=100, flag=False, model='VBPR', release_memory=False):
    if flag:
        random.seed(running_times)
        np.random.seed(running_times)
    else:
        random.seed(running_times)  # Amazo_Men是0, Amazon_Women是10
        np.random.seed(running_times)
    target_user_items_dict = {}
    target_rank_filter = {}
    target_user = random.sample(range(usz), user_number)
    target_rank_score = torch.matmul(-user[target_user], item_image.T).detach().cpu().numpy()
    target_rank_idx = np.argsort(target_rank_score, axis=1)
    target_rank_dict = {user_index: target_rank_idx[i] for i, user_index in enumerate(target_user)}
    for user_index in target_user:
        target_user_pos = np.array(list(inter[user_index]), dtype=np.int_)
        target_indices_to_keep = np.logical_not(np.isin(target_rank_dict[user_index], target_user_pos))
        target_rank_filter[user_index] = target_rank_dict[user_index][target_indices_to_keep]
        top_first_part_idx = int(1/3 * len(target_rank_filter[user_index]))
        top_second_part_idx = int(2/3 * len(target_rank_filter[user_index]))
        sample_distribution = split_number(total_samples, 3)
        target_sample_first_part = np.random.choice(target_rank_filter[user_index][100:top_first_part_idx], size=sample_distribution[0], replace=False)
        target_sample_second_part = np.random.choice(target_rank_filter[user_index][top_first_part_idx:top_second_part_idx], size=sample_distribution[1], replace=False)
        target_sample_third_part = np.random.choice(target_rank_filter[user_index][top_second_part_idx:], size=sample_distribution[2], replace=False)
        favourite_item = random.choices(target_rank_filter[user_index][:100], k=1)[0]
        favourite_items = target_rank_filter[user_index][:100]
        item_set = np.concatenate((target_sample_first_part, target_sample_second_part, target_sample_third_part))
        user_item_dict = {
            'user_id': user_index,
            'interaction_items': target_user_pos,
            'first_part': target_sample_first_part,
            'second_part': target_sample_second_part,
            'third_part': target_sample_third_part,
            'item_set': item_set,
            'item_distribution': sample_distribution,
            'favourite_item': favourite_item,
            'favourite_items': favourite_items,
        }
        target_user_items_dict[user_index] = user_item_dict
    return target_user_items_dict


def load_embedding(file_path, model_name, adv, tpdv, image_file_path=None, batch_size=32):
    global user, item_image, normalize_feature_value, phi, item, ori_item_embed
    if model_name == 'VBPR':
        user, item, phi = np.load(os.path.join(f'{file_path}', f'{model_name}_epoch_best.npy'), allow_pickle=True)
        ori_item_embed = np.load(os.path.join(f'{file_path}', 'VBPR_features.npy'), allow_pickle=True)
        normalize_feature_value = np.max(np.abs(ori_item_embed))
        ori_item_embed = ori_item_embed / normalize_feature_value
        user = torch.tensor(user).to(**tpdv)
        item = torch.tensor(item).to(**tpdv)
        phi = torch.tensor(phi).to(**tpdv)
        ori_item_embed = torch.tensor(ori_item_embed).to(**tpdv)
        item_image = torch.matmul(ori_item_embed, phi) + item
    elif model_name == 'AMR':
        user, item, phi = np.load(os.path.join(f'{file_path}', f'{model_name}_epoch_best_AMR.npy'), allow_pickle=True)
        ori_item_embed = np.load(os.path.join(f"{file_path}", 'VBPR_features.npy'), allow_pickle=True)
        normalize_feature_value = np.max(np.abs(ori_item_embed))
        ori_item_embed = ori_item_embed / normalize_feature_value
        user = torch.tensor(user).to(**tpdv)
        item = torch.tensor(item).to(**tpdv)
        phi = torch.tensor(phi).to(**tpdv)
        ori_item_embed = torch.tensor(ori_item_embed).to(**tpdv)
        item_image = torch.matmul(ori_item_embed, phi) + item
    elif model_name == 'DVBPR':
        custom_objects = {'tf': tf}
        user = np.load(os.path.join(f'{file_path}', 'DVBPR_user_embedding_matrix_epoch_best.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(file_path, 'DVBPR_image_embeddings.npy')):
            item_image = np.load(os.path.join(file_path, 'DVBPR_image_embeddings.npy'), allow_pickle=True)
        else:
            feature_model = load_model(os.path.join(f"{file_path}", 'DVBPR_feature_model_epoch_best.h5'),
                                       custom_objects=custom_objects)
            item_image = np.zeros((len(image_file_path), feature_model.output_shape[-1]))
            total_batches = (len(image_file_path) - 1) // batch_size + 1
            halfway_point = total_batches // 2
            num_images = len(image_file_path)
            with tqdm(total=total_batches, desc="Processing Image Batches", unit="batch") as pbar:
                for batch in range(total_batches):
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, num_images)

                    # Extract features for the current batch of images
                    image_files_batch = image_file_path[start_idx:end_idx]
                    image_embeddings = extract_image_features(image_files_batch, feature_model)

                    # Store embeddings in the main array
                    item_image[start_idx:end_idx] = image_embeddings
                    if batch == halfway_point - 1:
                        np.save(os.path.join(f"{file_path}", 'DVBPR_image_embeddings_1.npy'),
                                item_image[:end_idx])
                        print(f"First half of image embeddings saved at batch {batch + 1}")
                    pbar.update(1)
            np.save(os.path.join(f"{file_path}", 'DVBPR_image_embeddings_2.npy'),
                    item_image[halfway_point * batch_size:])
            print("Second half of image embeddings saved")
            # Save the complete embedding matrix
            np.save(os.path.join(f"{file_path}", 'DVBPR_image_embeddings.npy'), item_image)
            print("All image embeddings saved")

            # Clear session after saving all embeddings
            tf.keras.backend.clear_session()
        user = torch.tensor(user).to(**tpdv)
        item_image = torch.tensor(item_image).to(**tpdv)

    elif model_name == 'DeepStyle':
        global category_bias, item_category, item_image_embed
        data = np.load(os.path.join(f'{file_path}', 'DeepStyle_embeddings_epoch_best.npz'))
        user = data["user_embeddings"]
        item = data["item_embeddings"]
        category_bias = data["category_embeddings"]
        phi = data['phi_matrix']
        data.close()
        if os.path.exists(os.path.join(file_path, 'DeepStyle_image_embeddings.npy')):
            item_image_embed = np.load(os.path.join(file_path, 'DeepStyle_image_embeddings.npy'), allow_pickle=True)
        else:
            custom_objects = {'tf': tf}
            feature_model = load_model(os.path.join(f"{file_path}", 'DeepStyle_feature_model_epoch_best.h5'),
                                       custom_objects=custom_objects)
            item_image = np.zeros((len(image_file_path), feature_model.output_shape[-1]))
            total_batches = (len(image_file_path) - 1) // batch_size + 1
            halfway_point = total_batches // 2
            num_images = len(image_file_path)
            with tqdm(total=total_batches, desc="Processing Image Batches", unit="batch") as pbar:
                for batch in range(total_batches):
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, num_images)

                    # Extract features for the current batch of images
                    image_files_batch = image_file_path[start_idx:end_idx]
                    image_embeddings = extract_image_features(image_files_batch, feature_model)

                    # Store embeddings in the main array
                    item_image[start_idx:end_idx] = image_embeddings
                    if batch == halfway_point - 1:
                        np.save(os.path.join(f"{file_path}", 'DeepStyle_image_embeddings_1.npy'),
                                item_image[:end_idx])
                        print(f"First half of image embeddings saved at batch {batch + 1}")
                    pbar.update(1)
            np.save(os.path.join(f"{file_path}", 'DeepStyle_image_embeddings_2.npy'),
                    item_image[halfway_point * batch_size:])
            print("Second half of image embeddings saved")
            # Save the complete embedding matrix
            np.save(os.path.join(f"{file_path}", 'DeepStyle_image_embeddings.npy'), item_image)
            print("All image embeddings saved")

            # Clear session after saving all embeddings
            tf.keras.backend.clear_session()
            item_image_embed = np.load(os.path.join(file_path, 'DeepStyle_image_embeddings.npy'), allow_pickle=True)
        user = torch.tensor(user).to(**tpdv)
        item = torch.tensor(item).to(**tpdv)
        category_bias = torch.tensor(category_bias).to(**tpdv)
        item_category = torch.tensor(
            pd.read_csv(f'{"/".join(image_file_path[0].split("/")[:-2])}/classes.csv')['ClassNum'].tolist(), dtype=torch.int64, device=tpdv['device'])
        phi = torch.tensor(phi).to(**tpdv)
        item_image_embed = torch.tensor(item_image_embed).to(**tpdv)
        item_image = item + torch.matmul(item_image_embed, phi) - category_bias[item_category]

def embedding_return(model_name):
    global user, normalize_feature_value, phi, item, ori_item_embed, item_image, category_bias, item_category, item_image_embed
    if model_name == 'VBPR' or model_name == 'AMR':
        return user, ori_item_embed, item, normalize_feature_value, phi
    elif model_name == 'DVBPR':
        return user, item_image
    elif model_name == 'DeepStyle':
        return user, item, category_bias, item_category, phi, item_image_embed
    else:
        raise ValueError(f'No recommender system model named {model_name}')



def ori_prediction(former_emb, latter_emb, item_idx, people_idx, tpdv, model='VBPR'):
    user_pos = torch.tensor(list(inter[people_idx]), dtype=torch.int).to(tpdv['device'])
    if isinstance(former_emb, tf.Tensor):
        former_emb = torch.tensor(former_emb.numpy()).to(**tpdv)
    elif not isinstance(former_emb, torch.Tensor):
        former_emb = torch.tensor(former_emb).to(**tpdv)
    elif former_emb.device != tpdv['device']:
        former_emb = former_emb.to(**tpdv)
    if model == 'VBPR' or model == 'AMR':
        former_item_embed = torch.matmul(former_emb / normalize_feature_value, phi) + item[item_idx, :]
    elif model == 'DVBPR':
        former_item_embed = former_emb
    elif model == 'DeepStyle':
        former_item_embed = torch.matmul(former_emb, phi) + item[item_idx] - category_bias[item_category[item_idx]]
    else:
        raise ValueError(f"No recommender system model named {model}")
    item_image_copy = item_image.clone()
    item_image_copy[item_idx, :] = former_item_embed  # 计算上一次的图片特征
    user_embed = user[people_idx, :].reshape(1, -1)  # 用户特征，这个是不变的
    rank_score_ori = torch.matmul(-user_embed, item_image_copy.T)
    rank = torch.argsort(rank_score_ori)
    indices_to_keep = torch.logical_not(torch.isin(rank, user_pos))
    rank_filter = rank[indices_to_keep][None, ...]
    ori_index = torch.where(rank_filter == int(item_idx))[1].item() + 1
    # print(f"原样本排名{ori_index}")
    if isinstance(latter_emb, tf.Tensor):
        latter_emb = torch.tensor(latter_emb.numpy()).to(**tpdv)
    elif not isinstance(latter_emb, torch.Tensor):
        latter_emb = torch.tensor(latter_emb).to(**tpdv)
    elif latter_emb.device != tpdv['device']:
        latter_emb = latter_emb.to(**tpdv)
    if model == 'VBPR' or model == 'AMR':
        latter_item_embed = torch.matmul(latter_emb / normalize_feature_value, phi) + item[item_idx, :]
    elif model == 'DVBPR':
        latter_item_embed = latter_emb
    elif model == 'DeepStyle':
        latter_item_embed = torch.matmul(latter_emb, phi) + item[item_idx] - category_bias[item_category[item_idx]]
    else:
        raise ValueError(f"No recommender system model named {model}")
    item_image_copy[item_idx, :] = latter_item_embed  # 计算这一次特征
    adv_rank_score = torch.matmul(-user_embed, item_image_copy.T)
    adv_rank = torch.argsort(adv_rank_score)
    indices_to_keep = torch.logical_not(torch.isin(adv_rank, user_pos))
    adv_rank_filtered = adv_rank[indices_to_keep][None, ...]
    adv_index = torch.where(adv_rank_filtered == int(item_idx))[1].item() + 1
    # print(f"现在样本排名为{adv_index}")
    score = ori_index - adv_index
    # if adv_index < ori_index:
    #     print(f'排名进步{score}')
    # else:
    #     print(f'排名退步{-score}')
    del former_emb, latter_emb, item_image_copy
    return score, adv_index, ori_index

def gradient_promotion(input, emb, item_idx, people_idx, tpdv):
    user_pos = torch.tensor(list(inter[people_idx]), dtype=torch.int).to(tpdv['device'])
    item_embed = torch.matmul(emb / normalize_feature_value, phi) + item[item_idx, :]
    item_image[item_idx, :] = item_embed  # 计算上一次的图片特征
    user_embed = user[people_idx, :].reshape(1, -1)  # 用户特征，这个是不变的
    rank_score_ori = torch.matmul(-user_embed, item_image.T)
    rank_score = rank_score_ori[0, item_idx]
    rank_score.backward()
    if input.grad is not None:
        print(input.grad)
    else:
        print('No gradient!')
    rank = torch.argsort(rank_score_ori)
    indices_to_keep = torch.logical_not(torch.isin(rank, user_pos))
    rank_filter = rank[indices_to_keep][None, ...]
    ori_index = torch.where(rank_filter == int(item_idx))[1].item() + 1
    # print(f"原样本排名{ori_index}")

    return ori_index

def release_resources():
    """Release global variables to free memory."""
    global user, item_image, normalize_feature_value, phi, item  # Declare them as global

    try:
        # Delete the variables if they exist
        del user, item_image
        print("Resources released successfully.")
    except NameError as e:
        print(f"Resource not found: {e}")

    # Free GPU memory if used
    torch.cuda.empty_cache()
