import os.path

import numpy as np

from utils_wcy.Configuration import args_parser
from utils_wcy.load_rec_generator import *
from utils_wcy.Attack import *
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import pandas as pd
from utils_wcy.Classification import Feature_Extract, Classify, Comput_SSIM
from itertools import count

def attack(data, attack_type):
    WSS_list = []
    WPS_list = []
    AE_list = []
    HR_1_list = []
    HR_10_list = []
    HR_100_list = []
    img_info = {'user_id': [], 'image_name': [], 'SSIM_Value': [], 'original_index': [], 'final_index': [], 'HR_1': [],
                'HR_10': [], "HR_100": [], "WPS": [], "WSS": [], "AE": []}
    dataset = param.file_path.split("/")[-3]
    # D:\Recommender System Project\TAaMR-master\data\{dataset}\original_images\classes.csv
    item_csv = pd.read_csv(rf"/home/lxh/wcy/TAaMR-master/data/{dataset}/original_images/classes.csv")
    for index, (user_id, user_info) in enumerate(data.items(), start=1):
        HR_1 = 0
        HR_10 = 0
        HR_100 = 0
        AE = 0  # Attack Effectiveness
        WPS_denominator = 0  # Weighted Promotion Score
        WPS_numerator = 0
        WPS = 0
        WSS = 0  # Weighted Stealthiness Score
        WSS_denominator = 0
        WSS_numerator = 0
        total_items = len(user_info['item_set'])
        pbar = tqdm(total=total_items, desc="Processing Items", unit="item")
        target = user_info['favourite_item']
        print(f"Processing user: {user_id}")
        target_index = torch.tensor(item_csv['ClassNum'][target], dtype=torch.long, device=device)
        for item in user_info['item_set']:
            pbar.set_description(f"Processing user {index}/{param.user_number}'s item {item}.jpg")
            file = os.path.join(param.file_path, file_lis[item])
            if attack_type.upper() == 'FGSM':
                for perturbation in count(1):
                    modify_image = Add_perturb(model, file, perturbation / 255, attack_type.upper(), target=target_index, tensor_file=True)
                    prediction_result = Classify(modify_image, model, tpdv)
                    prediction_index = np.argmax(prediction_result)
                    if prediction_index == target_index or perturbation == 32:
                        adv_image = to_pil_image(modify_image)
                        image_file_path = fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/user_id{user_id}'
                        if not os.path.exists(image_file_path):
                            os.makedirs(image_file_path)
                        save_img = os.path.join(image_file_path, f'attack_img_{item}.jpg')
                        adv_image.save(save_img)
                        ssim_value = Comput_SSIM(file, save_img)
                        if param.rec_arch == 'tensorflow_file':
                            feature_adv_former = Feature_Extract(file, rec_feature_model, tpdv,
                                                                 param.rec_arch,
                                                                 tensor_feature=True, is_tensorflow=True)
                            feature_adv_latter = Feature_Extract(save_img, rec_feature_model, tpdv,
                                                                 param.rec_arch,
                                                                 tensor_feature=True, is_tensorflow=True)
                            feature_adv_former = tf.stop_gradient(feature_adv_former)
                            feature_adv_latter = tf.stop_gradient(feature_adv_latter)
                            score, adv_index, ori_index = ori_prediction(feature_adv_former, feature_adv_latter,
                                                                         item, user_id,
                                                                         tpdv, model=param.model)
                        else:
                            score, adv_index, ori_index = ori_prediction(
                                Feature_Extract(file, rec_feature_model, tpdv, tensor_feature=True).detach(),
                                Feature_Extract(save_img, rec_feature_model, tpdv,
                                                tensor_feature=True).detach(),
                                item, user_id, tpdv, param.model
                                )
                        final_index = adv_index
                        break
            elif attack_type.upper() == 'PGD':
                for iteration_step in count(1):
                    modify_image = Add_perturb(model, file, 32 / 255, attack_type.upper(), iteration_number=iteration_step, target=target_index, tensor_file=True)
                    prediction_result = Classify(modify_image, model, tpdv)
                    prediction_index = np.argmax(prediction_result)
                    if prediction_index == target_index or iteration_step == 32:
                        adv_image = to_pil_image(modify_image)
                        image_file_path = fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/user_id{user_id}'
                        if not os.path.exists(image_file_path):
                            os.makedirs(image_file_path)
                        save_img = os.path.join(image_file_path, f'attack_img_{item}.jpg')
                        adv_image.save(save_img)
                        ssim_value = Comput_SSIM(file, save_img)
                        if param.rec_arch == 'tensorflow_file':
                            feature_adv_former = Feature_Extract(file, rec_feature_model, tpdv,
                                                                 param.rec_arch,
                                                                 tensor_feature=True, is_tensorflow=True)
                            feature_adv_latter = Feature_Extract(save_img, rec_feature_model, tpdv,
                                                                 param.rec_arch,
                                                                 tensor_feature=True, is_tensorflow=True)
                            feature_adv_former = tf.stop_gradient(feature_adv_former)
                            feature_adv_latter = tf.stop_gradient(feature_adv_latter)
                            score, adv_index, ori_index = ori_prediction(feature_adv_former, feature_adv_latter,
                                                                         item, user_id,
                                                                         tpdv, model=param.model)
                        else:
                            score, adv_index, ori_index = ori_prediction(
                                Feature_Extract(file, rec_feature_model, tpdv, tensor_feature=True).detach(),
                                Feature_Extract(save_img, rec_feature_model, tpdv,
                                                tensor_feature=True).detach(),
                                item, user_id, tpdv, param.model
                                )
                        final_index = adv_index
                        break


            if final_index < ori_index:
                AE += 1
                if item in user_info['first_part']:
                    WPS_numerator += (ori_index - final_index) * 0.2
                    WPS_denominator += (ori_index - 1) * 0.2
                    WSS_numerator += ssim_value * 0.5
                    WSS_denominator += 0.5
                elif item in user_info['second_part']:
                    WPS_numerator += (ori_index - final_index) * 0.3
                    WPS_denominator += (ori_index - 1) * 0.3
                    WSS_numerator += ssim_value * 0.3
                    WSS_denominator += 0.3
                elif item in user_info['third_part']:
                    WPS_numerator += (ori_index - final_index) * 0.5
                    WPS_denominator += (ori_index - 1) * 0.5
                    WSS_numerator += ssim_value * 0.2
                    WSS_denominator += 0.2
                else:
                    raise ValueError(f"{item} not in the user's item set")
            else:
                WPS_numerator += 0
                if item in user_info['first_part']:
                    WPS_denominator += (ori_index - 1) * 0.2
                    WSS_numerator += 0.5 * ssim_value
                    WSS_denominator += 0.5
                elif item in user_info['second_part']:
                    WPS_denominator += (ori_index - 1) * 0.3
                    WSS_numerator += 0.3 * ssim_value
                    WSS_denominator += 0.3
                elif item in user_info['third_part']:
                    WPS_denominator += (ori_index - 1) * 0.5
                    WSS_numerator += 0.2 * ssim_value
                    WSS_denominator += 0.2
            if final_index == 1:
                HR_1 += 1
                HR_10 += 1
                HR_100 += 1
            elif final_index < 11:
                HR_10 += 1
                HR_100 += 1
            elif final_index < 101:
                HR_100 += 1
            img_info['user_id'].append(f'{user_id}')
            img_info['image_name'].append(f'attack_img_{item}.jpg')
            img_info['SSIM_Value'].append(f'{ssim_value:.3f}')
            img_info['original_index'].append(f'{ori_index}')
            img_info['final_index'].append(f'{final_index}')
            img_info['HR_1'].append(None)
            img_info['HR_10'].append(None)
            img_info['HR_100'].append(None)
            img_info['AE'].append(None)
            img_info['WPS'].append(None)
            img_info['WSS'].append(None)
            pbar.update(1)
        WPS = WPS_numerator / WPS_denominator
        WSS = WSS_numerator / WSS_denominator
        AE = AE / total_items
        img_info['user_id'].append(f'{user_id}')
        img_info['image_name'].append(None)
        img_info['SSIM_Value'].append(None)
        img_info['original_index'].append(None)
        img_info['final_index'].append(None)
        img_info['HR_1'].append(f"{HR_1 / total_items}")
        img_info['HR_10'].append(f"{HR_10 / total_items}")
        img_info['HR_100'].append(f"{HR_100 / total_items}")
        img_info['AE'].append(f"{AE}")
        img_info['WSS'].append(f"{WSS}")
        img_info['WPS'].append(f"{WPS}")
        WPS_list.append(WPS)
        WSS_list.append(WSS)
        AE_list.append(AE)
        HR_1_list.append(HR_1/total_items)
        HR_10_list.append(HR_10/total_items)
        HR_100_list.append(HR_100/total_items)
        print(
            f'Attack_type:{attack_type}, HR_1:{HR_1}, HR_10:{HR_10}, HR_100:{HR_100}, WSS:{WSS}, WPS:{WPS}, AE:{AE}')
    df = pd.DataFrame(img_info)
    df.to_csv(
        fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/attack_img_info.csv',
        index=False)
    WSS_average = np.array(WSS_list).mean()
    WPS_average = np.array(WPS_list).mean()
    AE_average = np.array(AE_list).mean()
    HR_1_average = np.array(HR_1_list).mean()
    HR_10_average = np.array(HR_10_list).mean()
    HR_100_average = np.array(HR_100_list).mean()
    print(fr"final result that averaged on {param.user_number} users as followed:")
    print(f"HR_1_Average:{HR_1_average:%}")
    print(f"HR_10_Average:{HR_10_average:%}")
    print(f"HR_100_Average:{HR_100_average:%}")
    print(f"AE_Average:{AE_average:%}")
    print(f"WPS_Average:{WPS_average:%}")
    print(f"WSS_Average:{WSS_average:.4f}")
    print(
        f"processed Image info has been stored at:'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/attack_img_info.csv")


def Save_user_items_info(data_dict):
    rows = []
    for user_index, data in data_dict.items():
        rows.append({
            'user_id': user_index,
            'interaction_items': ', '.join(map(str, data['interaction_items'])),
            'first_part': ', '.join(map(str, data['first_part'])),
            'second_part': ', '.join(map(str, data['second_part'])),
            'third_part': ', '.join(map(str, data['third_part'])),
            'item_set': ', '.join(map(str, data['item_set'])),
            'item_distribution': ', '.join(map(str, data['item_distribution'])),
            'favourite_item': data['favourite_item']
        })

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(rows)
    file_path = fr'./results/{param.model}/{param.dataset}/{param.attack}/Subs_{param.arch}-Rec_{param.rec_arch}/Item_User_Info/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    csv_file_path = os.path.join(file_path, 'target_user_items.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")


def initialize_models(param, tpdv, custom_rec_model_path=None, custom_rec_model_type=None):
    """
    Initializes the feature extraction model and recommendation model.

    Parameters:
    - param: A dictionary containing the architecture names for both models (e.g., {'arch': 'resnet50', 'rec_arch': 'resnet18'}).
    - tpdv: The device configuration (e.g., {'device': 'cuda'}).
    - custom_rec_model_path (optional): Path to a custom-trained recommendation model if not using a pretrained model.
    - custom_rec_model_type (optional): Type of the custom-trained model (e.g., 'DVBPR').

    Returns:
    - feature_model: The feature extraction model.
    - rec_feature_model: The recommendation model.
    """

    # Initialize feature extraction model
    if 'arch' not in param:
        raise ValueError("Parameter 'arch' (feature extraction model architecture) is required.")
    print(f"Initializing classification model: {param['arch']}")

    model = eval(f"models.{param['arch']}(pretrained=True)").to(**tpdv)

    # Initialize recommendation model
    if custom_rec_model_path and custom_rec_model_type:
        print(f"Loading custom recommendation model: {custom_rec_model_type} from {custom_rec_model_path}")

        if custom_rec_model_type == 'DVBPR' or custom_rec_model_type == 'DeepStyle':
            # Custom handling for the DVBPR model loaded via TensorFlow or other frameworks
            import tensorflow as tf
            custom_objects = {'tf': tf}
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Set memory growth to avoid occupying all memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPUs detected: {gpus}")
                except RuntimeError as e:
                    print(f"Error configuring GPU memory: {e}")

            # Load the model on a specific GPU (e.g., 'GPU:0')
            with tf.device('/GPU:1'):  # Use '/GPU:1' or other device if needed
                rec_feature_model = tf.keras.models.load_model(custom_rec_model_path, custom_objects=custom_objects)
        else:
            raise ValueError(f"Unknown custom recommendation model type: {custom_rec_model_type}")

    elif 'rec_arch' in param:
        print(f"Initializing pretrained recommendation model: {param['rec_arch']}")
        rec_ic_model = eval(f"models.{param['rec_arch']}(pretrained=True)").to(**tpdv)
        rec_feature_model = nn.Sequential(*list(rec_ic_model.children())[:-1], nn.Flatten()).to(**tpdv)
    else:
        raise ValueError(
            "Parameter 'rec_arch' (recommendation model architecture) is required if not loading a custom model.")
    return model, rec_feature_model

def sort_key(filename):
    # Extract the numeric part of the filename between '0' and '.jpg'
    return int(filename.split('.')[0])

if __name__ == '__main__':
    global file_lis
    param = args_parser()
    to_pil_image = transforms.ToPILImage()
    device = f'cuda:{param.GPU}' if torch.cuda.is_available() else 'cpu'
    tpdv = dict(dtype=torch.float32, device=device)
    file_lis = os.listdir(param.file_path)
    file_lis = sorted(file_lis, key=sort_key)
    image_file_list = [f"{os.path.join(param.file_path, file)}" for file in file_lis]
    user_preference(param.pos_txt)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if param.dataset == 'Amazon_men':
        dataset_name = 'amazon_men'
    elif param.dataset == 'Amazon_Women':
        dataset_name = 'amazon_women'
    if param.model == 'VBPR':
        model_weight_path = os.path.join(f"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv)
        model, rec_feature_model = initialize_models({'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name':param.model}, tpdv)
    elif param.model == 'DVBPR':
        # D:\Recommender System Project\TAaMR-master\rec_model_weights\amazon_men\original_images
        model_weight_path = os.path.join(
            fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{dataset_name}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        model, rec_feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model},
            custom_rec_model_path=fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{dataset_name}/original_images/DVBPR_feature_model_epoch_best.h5",
            custom_rec_model_type='DVBPR', tpdv=tpdv)
        # custom_rec_model_path = fr"D:\Recommender System Project\TAaMR-master\rec_model_weights\amazon_men\original_images\DVBPR_feature_model_epoch_best.h5",
    elif param.model == 'DeepStyle':
        model_weight_path = os.path.join(
            fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        model, rec_feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model},
            custom_rec_model_path=os.path.join(model_weight_path, r"DeepStyle_feature_model_epoch_best.h5"),
            custom_rec_model_type='DeepStyle', tpdv=tpdv)
    if param.dataset == 'amazon_men':
        target_user_items_dict = generate_target(0, param.user_number)
    elif param.dataset == 'amazon_women':
        target_user_items_dict = generate_target(10, param.user_number)
    else:
        raise ValueError(f"There is no dataset named {param.dataset}")
    Save_user_items_info(target_user_items_dict)


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    DVBPR_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of the model
        transforms.ToTensor(),  # Convert to tensor (this gives shape (C, H, W))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    denormalize = transforms.Compose([
        transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std])
    ])
    to_Tensor = transforms.ToTensor()

    attack(target_user_items_dict, param.attack)