import torchvision.models as models
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image

def load_embedding(file_path, tpdv):
    global user, item_image, normalize_feature_value, phi, item
    user, item, phi = np.load(os.path.join(f'{file_path}', 'epoch4000.npy'), allow_pickle=True)
    ori_item_embed = np.load(os.path.join(f'{file_path}', 'features.npy'), allow_pickle=True)
    normalize_feature_value = np.max(np.abs(ori_item_embed))
    ori_item_embed = ori_item_embed / normalize_feature_value
    user = torch.tensor(user).to(**tpdv)
    item = torch.tensor(item).to(**tpdv)
    phi = torch.tensor(phi).to(**tpdv)
    ori_item_embed = torch.tensor(ori_item_embed).to(**tpdv)
    item_image = torch.matmul(ori_item_embed, phi) + item
    pass

def ori_prediction(former_emb, latter_emb, item_idx, people_idx, tpdv):
    former_emb = former_emb.view(1, -1)
    latter_emb = latter_emb.view(1, -1)
    former_item_embed = torch.matmul((former_emb / normalize_feature_value), phi) + item[item_idx, :]
    item_image[item_idx, :] = former_item_embed  # 计算上一次的图片特征
    user_embed = user[people_idx, :].reshape(1, -1)  # 用户特征，这个是不变的
    rank_score_ori = torch.matmul(-user_embed, item_image.T)
    rank = torch.argsort(rank_score_ori)
    ori_index = torch.where(rank == int(item_idx))[1].item() + 1
    # print(f"原样本排名{ori_index}")

    latter_item_embed = torch.matmul((latter_emb / normalize_feature_value), phi) + item[item_idx, :]
    item_image[item_idx, :] = latter_item_embed  # 计算这一次
    adv_rank_score = torch.matmul(-user_embed, item_image.T)
    adv_rank = torch.argsort(adv_rank_score)
    adv_index = torch.where(adv_rank == int(item_idx))[1].item() + 1
    # print(f"现在样本排名为{adv_index}")
    score = ori_index - adv_index
    # if adv_index < ori_index:
    #     print(f'排名进步{score}')
    # else:
    #     print(f'排名退步{-score}')
    del former_emb, latter_emb
    return score, adv_index, ori_index

def modify_numpy_array(arr):
    # 检查数组的维度
    if len(arr.shape) == 3:
        if arr.shape[0] == 3:  # 如果是 (c, h, w)
            return np.transpose(arr, (1, 2, 0))  # 转换为 (h, w, c)
        elif arr.shape[-1] == 3:  # 如果是 (h, w, c)，保持原样
            return arr
    # 如果不符合要求，返回 None 或者抛出异常，取决于你的需求
    return None

ori_path = r'E:\Reinforcement-Learning-Against-RecSys\utils_wcy\0.jpg'
adv_path = r'E:\Reinforcement-Learning-Against-RecSys\results\0_adv.npy'

transforms_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

tpdv = dict(dtype=torch.float64, device='cuda:0')
load_embedding(r'E:\Reinforcement-Learning-Against-RecSys\model\VBPR\Amazon', tpdv)
model = models.resnet50(pretrained=True).to(**tpdv)
model.eval()
feature_model = nn.Sequential(*list(model.children())[:-1]).to(**tpdv)

if ori_path[-3:] == 'jpg':
    ori_pic = Image.open(ori_path)
    ori_feature = feature_model(transforms_image(ori_pic)[None, ...].to(**tpdv))
elif ori_path[-3:] == 'npy':
    ori_pic = np.load(ori_path, allow_pickle=True)
    ori_feature = torch.tensor(ori_pic).to(**tpdv)
if adv_path[-3:] == 'jpg':
    adv_pic = Image.open(adv_path)
    adv_feature = feature_model(transforms_image(adv_pic)[None, ...].to(**tpdv))
elif adv_path[-3:] == 'npy':
    adv_pic = modify_numpy_array(np.load(adv_path, allow_pickle=True))
    adv_feature = feature_model(transforms_image(adv_pic)[None, ...].to(**tpdv))

score, adv_index, ori_index = ori_prediction(ori_feature, adv_feature, 0, 0, tpdv)
print(f"原排名:{ori_index},现在排名:{adv_index}")
