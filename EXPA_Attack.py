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
from utils_wcy.Classification import Feature_Extract, Comput_SSIM
from itertools import count


def save_perturbed_image_tf(attack_img_path, perturbation, image_file_path, item, user_id):
    # Load the original image and convert it to a NumPy array
    attack_img = Image.open(attack_img_path).convert('RGB')
    attack_img_resized = attack_img.resize((perturbation.shape[2], perturbation.shape[1]))
    attack_img_np = np.array(attack_img_resized) / 255.0  # Normalize the image to [0, 1] range
    # Convert the perturbation to a NumPy array (if it's a tensor)
    perturbation_np = perturbation.numpy()
    perturbation_np = perturbation_np.transpose(1, 2, 0)

    # Add the perturbation to the original image and clamp values between 0 and 1
    perturbed_img_np = np.clip(attack_img_np + perturbation_np, 0, 1)

    save_dir = os.path.join(image_file_path, f'user_id{user_id}')
    os.makedirs(save_dir, exist_ok=True)
    img_save_path = os.path.join(save_dir, f'attack_img_{item}.jpg')
    np_save_path = os.path.join(save_dir, f'attack_img_{item}.npy')

    np.save(np_save_path, perturbed_img_np)

    # Convert the perturbed image back to the [0, 255] range
    perturbed_img_np = (perturbed_img_np * 255).astype(np.uint8)

    # Convert the NumPy array back to a PIL image
    perturbed_img = Image.fromarray(perturbed_img_np)

    # Save the perturbed image
    perturbed_img.save(img_save_path)
    return img_save_path, np_save_path

def save_perturbed_image_torch(attack_img_path, perturbation, image_file_path, item):
    adv_img = to_Tensor(Image.open(attack_img_path)) + perturbation
    adv_img = torch.clamp(adv_img, 0, 1)
    adv_img_pil = to_pil_image(adv_img)
    os.makedirs(image_file_path, exist_ok=True)
    img_save_path = os.path.join(image_file_path, f'attack_img_{item}.jpg')
    np_save_path = os.path.join(image_file_path, f'attack_img_{item}.npy')
    adv_img_pil.save(img_save_path)
    np.save(np_save_path, adv_img.detach().numpy())
    return img_save_path, np_save_path



def attack(data, attack_type):
    WSS_list = []
    WPS_list = []
    AE_list = []
    HR_1_list = []
    HR_10_list = []
    HR_100_list = []
    img_info = {'user_id': [], 'image_name': [], 'SSIM_Value': [], 'original_index': [], 'final_index': [], 'HR_1': [], 'HR_10': [], "HR_100": [], "WPS": [], "WSS": [], "AE": []}
    flag = False
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
        for item in user_info['item_set']:
            final_index = 1e6
            pbar.set_description(f"Processing user {index}/{param.user_number}'s item {item}.jpg")
            attack_img = os.path.join(param.file_path, file_lis[item])
            target_img = os.path.join(param.file_path, file_lis[target])
            if param.arch == 'tensorflow_file':
                perturbation = tf.Variable(
                    tf.zeros_like(tf.convert_to_tensor(tf_preprocess(Image.open(attack_img))), dtype=tf.float32), trainable=True)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
                flag = True
            else:
                perturbation = torch.zeros_like(preprocess(Image.open(attack_img))).requires_grad_(True)
                optimizer = torch.optim.Adam([perturbation], lr=0.1)
            criterion = nn.MSELoss().to(**tpdv)
            count_step = 0
            for step in count(1):
                if flag:
                    with tf.GradientTape() as tape:
                        feature_adv_latter = Feature_Extract(attack_img, feature_model, tpdv, param.arch,
                                                             tensor_feature=True, perturbation=perturbation,
                                                             is_tensorflow=flag)

                        # Get the target features to match
                        feature_target = tf.stop_gradient(Feature_Extract(target_img, feature_model, tpdv, param.arch,
                                                         tensor_feature=True, is_tensorflow=flag))

                        # Compute the loss (MSE between the perturbed and target features)
                        loss = tf.reduce_mean(tf.square(feature_adv_latter - feature_target))

                    # Compute the gradients of the loss with respect to the perturbation
                    grads = tape.gradient(loss, perturbation)

                    # Apply the gradients to the perturbation using the optimizer
                    optimizer.apply_gradients([(grads, perturbation)])
                else:
                    feature_adv_latter = Feature_Extract(attack_img, feature_model, tpdv, param.arch, tensor_feature=True,
                                                         perturbation=perturbation, is_tensorflow=flag)
                    feature_target = Feature_Extract(target_img, feature_model, tpdv, param.arch, tensor_feature=True, is_tensorflow=flag).detach()
                    loss = criterion(feature_adv_latter, feature_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if param.rec_arch == 'tensorflow_file':
                        image_file_path = fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/user_id{user_id}'
                        perturbation_file, perturbation_npy = save_perturbed_image_torch(attack_img, perturbation, image_file_path, item)

                if param.rec_arch == 'tensorflow_file':
                    feature_adv_former = tf.stop_gradient(Feature_Extract(attack_img, rec_feature_model, tpdv, param.rec_arch,
                                                         tensor_feature=True, is_tensorflow=True, adv=param.adv))
                    if not flag:
                        feature_adv_latter = tf.stop_gradient(Feature_Extract(perturbation_file, rec_feature_model, tpdv, param.rec_arch,
                                                         tensor_feature=True, is_tensorflow=True, adv=param.adv))
                    score, adv_index, ori_index = ori_prediction(feature_adv_former, tf.stop_gradient(feature_adv_latter),
                                                                 item, user_id,
                                                                 tpdv, model=param.model)
                else:
                    score, adv_index, ori_index = ori_prediction(Feature_Extract(attack_img, rec_feature_model, tpdv, param.rec_arch, tensor_feature=True, is_tensorflow=flag, adv=param.adv).detach(),
                                                             Feature_Extract(attack_img, rec_feature_model, tpdv, param.rec_arch,
                                                                             tensor_feature=True,
                                                                             perturbation=perturbation,
                                                                             is_tensorflow=flag, adv=param.adv).detach(),
                                                             item, user_id, tpdv, model=param.model
                                                             )
                if adv_index != final_index and step < 20:
                    final_index = adv_index
                    count_step = 0
                else:
                    count_step += 1
                    try:
                        loss_value = loss.item()
                    except:
                        loss_value = loss.numpy()
                    if count_step == 5 or loss_value < 2e-2:
                        print(f"step:{step},loss:{loss_value:.2f},ori_index:{ori_index}, adv_index:{adv_index}")
                        if param.arch == 'tensorflow_file':
                            image_file_path = fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}'
                            save_img, _ = save_perturbed_image_tf(attack_img, perturbation, image_file_path, item, user_id)
                        else:
                            image_file_path = fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/user_id{user_id}'
                            save_img, _ = save_perturbed_image_torch(attack_img, perturbation, image_file_path, item)
                        ssim_value = Comput_SSIM(attack_img, save_img, model=param.model)

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
                                WSS_numerator += ssim_value * 0.5
                                WSS_denominator += 0.5
                            elif item in user_info['second_part']:
                                WPS_denominator += (ori_index - 1) * 0.3
                                WSS_numerator += ssim_value * 0.3
                                WSS_denominator += 0.3
                            elif item in user_info['third_part']:
                                WPS_denominator += (ori_index - 1) * 0.5
                                WSS_numerator += ssim_value * 0.2
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
                        break
            pbar.update(1)
        WPS = WPS_numerator / WPS_denominator
        WSS = WSS_numerator / WSS_denominator
        AE = AE / total_items
        img_info['user_id'].append(f'{user_id}')
        img_info['image_name'].append(None)
        img_info['SSIM_Value'].append(None)
        img_info['original_index'].append(None)
        img_info['final_index'].append(None)
        img_info['HR_1'].append(f"{HR_1/total_items}")
        img_info['HR_10'].append(f"{HR_10/total_items}")
        img_info['HR_100'].append(f"{HR_100/total_items}")
        img_info['AE'].append(f"{AE}")
        img_info['WSS'].append(f"{WSS}")
        img_info['WPS'].append(f"{WPS}")
        WPS_list.append(WPS)
        WSS_list.append(WSS)
        AE_list.append(AE)
        HR_1_list.append(HR_1/total_items)
        HR_10_list.append(HR_10/total_items)
        HR_100_list.append(HR_100/total_items)
        print(f'Attack_type:{attack_type}, HR_1:{HR_1}, HR_10:{HR_10}, HR_100:{HR_100}, WSS:{WSS}, WPS:{WPS}, AE:{AE}')
    df = pd.DataFrame(img_info)
    df.to_csv(
        fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/attack_img_info.csv', index=False)
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
    print(f"processed Image info has been stored at:'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/attack_img_info.csv")



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
    print(f"Initializing feature extraction model: {param['arch']}")
    if param['arch'] != 'tensorflow_file':
        feature_model = eval(f"models.{param['arch']}(pretrained=True)").to(**tpdv)
        if param['arch'] == 'vgg16':
            # For VGG16, we need to take both the 'features' and 'classifier' (excluding the last layer of classifier)

            # Extract the 'features' part (convolutional layers)
            feature_layers = list(feature_model.features)
            avgpool_layer = feature_model.avgpool
            flatten_layer = nn.Flatten()
            # Extract the 'classifier' part (fully connected layers) and exclude the last layer
            classifier_layers = list(feature_model.classifier[:-1])

            # Combine 'features' and 'classifier' layers
            combined_layers = feature_layers + [avgpool_layer, flatten_layer] + classifier_layers

            # Create the feature extraction model from the combined layers
            feature_model = nn.Sequential(*combined_layers).to(**tpdv)
        else:
            # For other models (like ResNet), remove only the last layer
            feature_model = nn.Sequential(*list(feature_model.children())[:-1]).to(**tpdv)

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
        rec_feature_model = nn.Sequential(*list(rec_ic_model.children())[:-1]).to(**tpdv)
    else:
        raise ValueError(
            "Parameter 'rec_arch' (recommendation model architecture) is required if not loading a custom model.")
    if param['arch'] == 'tensorflow_file':
        return rec_feature_model, rec_feature_model
    else:
        return feature_model, rec_feature_model


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
        
    if param.model == 'VBPR' or param.model == 'AMR':
        model_weight_path = os.path.join(f"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv)
        feature_model, rec_feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model}, tpdv)
    elif param.model == 'DVBPR':
        model_weight_path = os.path.join(fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        feature_model, rec_feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model},
            custom_rec_model_path=os.path.join(model_weight_path, r"DVBPR_feature_model_epoch_best.h5"),
            custom_rec_model_type='DVBPR', tpdv=tpdv)
    elif param.model == 'DeepStyle':
        model_weight_path = os.path.join(
            fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        feature_model, rec_feature_model = initialize_models(
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
    tf_preprocess = transforms.Compose([
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