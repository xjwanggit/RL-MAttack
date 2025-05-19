import numpy as np
import torch
from sklearn.decomposition import PCA
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils_wcy.Configuration import args_parser
from utils_wcy.load_rec_generator import *
import torch.nn as nn
from utils_wcy.Classification import Feature_Extract
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import MDS, Isomap
import re
from sklearn.manifold import TSNE

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

def pca_analysis():
    attack_type = ['RLRS', 'EXPA', 'c-SEMA', 'FGSM', 'PGD']
    embedding_info_all = dict()
    for attack_name in attack_type:
        embedding_info_all[attack_name] = {'first_scenario': [], 'second_scenario': [], 'third_scenario':[]}
    model_name = param.model
    specific_item_index = int(specific_item[param.model])
    favourite_item_index = int(favourite_item)
    result_save_path = f'/home/lxh/wcy/Reinforcement-Learning-Against-RecSys/results/{model_name}/{param.dataset}/'
    for attack_method in attack_type:
        scenario_list = os.listdir(os.path.join(result_save_path, attack_method))
        for scenario_name in scenario_list:
            match_sub = re.search(r'Subs_(.*?)-Rec', scenario_name)
            match_rec = re.search(r'-Rec_(.*)', scenario_name)
            if match_sub:
                match_name = match_sub.group(1)
            else:
                raise ValueError(f'The {scenario_name}\'s substitute name no match found in the directory {(os.path.join(result_save_path, attack_method))}')
            if match_rec:
                recommender_system_name = match_rec.group(1)
                if recommender_system_name == 'tensorflow_file':
                    flag = True
                else:
                    flag = False
            else:
                raise ValueError(f'The {scenario_name}\'s recommender system\'s name no match found in the directory {(os.path.join(result_save_path, attack_method))}')
            if attack_method == 'c-SEMA':
                item_image_path = os.path.join(result_save_path, attack_method, scenario_name, f"user_id{user_id}",
                                               f"composite_image_{specific_item_index}.jpg")
            else:
                item_image_path = os.path.join(result_save_path, attack_method, scenario_name, f"user_id{user_id}",
                                               f"attack_img_{specific_item_index}.jpg")
            if match_name.lower() == 'resnet50' or match_name.lower() == 'tensorflow_file':
                image_features = get_item_embedding(model_name, specific_item_index, tpdv, rec_feature_model, item_image_path=item_image_path, flag=flag)
                embedding_info_all[attack_method]['first_scenario'].append(image_features)
                if attack_method == 'c-SEMA':
                    embedding_info_all[attack_method]['second_scenario'].append(image_features)
                    embedding_info_all[attack_method]['third_scenario'].append(image_features)
            elif match_name.lower() == 'resnet18':
                image_features = get_item_embedding(model_name, specific_item_index, tpdv, rec_feature_model, item_image_path=item_image_path, flag=flag)
                embedding_info_all[attack_method]['second_scenario'].append(image_features)
                if attack_method == 'c-SEMA':
                    embedding_info_all[attack_method]['first_scenario'].append(image_features)
                    embedding_info_all[attack_method]['third_scenario'].append(image_features)
            elif match_name.lower() == 'vgg16':
                image_features = get_item_embedding(model_name, specific_item_index, tpdv, rec_feature_model, item_image_path=item_image_path, flag=flag)
                embedding_info_all[attack_method]['third_scenario'].append(image_features)
                if attack_method == 'c-SEMA':
                    embedding_info_all[attack_method]['first_scenario'].append(image_features)
                    embedding_info_all[attack_method]['second-scenario'].append(image_features)
            else:
                raise ValueError(f"No scenario name is {match_name.lower()}")
    original_item_embedding = get_item_embedding(model_name, specific_item_index, tpdv, rec_feature_model)
    favourite_item_embedding = get_item_embedding(model_name, favourite_item_index, tpdv, rec_feature_model)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    scenarios = ['first_scenario', 'second_scenario', 'third_scenario']
    origin = (0, 0)

    color_cycle = plt.cm.get_cmap('tab10')
    plt.gca().set_prop_cycle('color', [color_cycle(i) for i in range(10)])

    # Initialize plot for each scenario
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        info = []

        # Collect embeddings for the current scenario across all attack methods
        for attack_method in attack_type:
            if embedding_info_all[attack_method][scenario]:
                info.append(np.vstack(embedding_info_all[attack_method][scenario]))

        # Append original and favorite item embeddings
        info.extend([original_item_embedding, favourite_item_embedding])

        # Convert embeddings to numpy array
        item_embeddings = np.vstack(info)
        normalized_item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        normalized_user_embedding = user_embedding / np.linalg.norm(user_embedding)

        # Compute cosine similarities
        cos_similarities = np.dot(normalized_item_embeddings, normalized_user_embedding)

        # Compute vector moduli and normalize
        vector_moduli = np.linalg.norm(item_embeddings, axis=1)
        normalized_vector_moduli = vector_moduli / np.max(vector_moduli)

        # Track labels to avoid duplicates in the legend
        unique_labels = set()

        # Plot each attack method's embeddings
        for i, (cos_sim, modulus) in enumerate(zip(cos_similarities, normalized_vector_moduli)):
            # Define the label name
            if i < len(attack_type):
                attack_method = attack_type[i]
                label_name = 'RL-MAttack' if attack_method == 'RLRS' else attack_method
            else:
                special_labels = ['Original Embeddings', 'Favourite Embeddings']
                label_name = special_labels[i % len(attack_type)]

            # Only add unique labels for legend
            if label_name not in unique_labels:
                unique_labels.add(label_name)
                label_arg = label_name
            else:
                label_arg = None

            # Calculate angle and positions for arrows
            angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            x = modulus * np.cos(angle)
            y = modulus * np.sin(angle)

            # Get unique color from color cycle for each vector
            color = color_cycle(i % 10)

            # Draw vector arrow with a minimum arrowhead size for visibility
            ax.arrow(origin[0], origin[1], x, y, fc=color, ec=color, alpha=0.8,
                     linewidth=1.5, label=label_arg, head_width=0.04, head_length=0.06)

            # Draw a thicker, more visible dotted line to the x-axis
            ax.plot([x, x], [y, 0], linestyle='--', color=color, linewidth=1.0, alpha=0.7)

        # Draw user embedding as a fixed vector of length 1
        ax.arrow(origin[0], origin[1], 1.0, 0, fc='black', ec='black', alpha=0.8, linewidth=1.5,
                 label='User Embeddings', head_width=0.04, head_length=0.06)

        # Additional plot settings
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1.0)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(f"t-SNE of Embeddings in {scenario.capitalize()}")
        ax.grid(True)

    # Create a single legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), bbox_transform=plt.gcf().transFigure)

    plt.tight_layout()
    plt.savefig('1.jpg', dpi=300, bbox_inches='tight')


def get_item_embedding(model_name, specific_item_index, tpdv, feature_model=None, item_image_path=None, flag=False):
    """
    Generalized function to obtain the item embedding for a specific item index.

    Arguments:
    - param: Object containing model configuration parameters.
    - specific_item_index: Index of the target item.
    - feature_model: Feature model to use for feature extraction, depending on the attack method.
    - tpdv: Device configuration (e.g., GPU settings).
    - item_image_path: Optional; path to the image if feature extraction is required.

    Returns:
    - Embedding for the specified item.
    """
    if model_name == 'VBPR' or model_name == 'AMR':
        if item_image_path:
            # Extract features from the image if item_image_path is provided
            extracted_feature = Feature_Extract(item_image_path, feature_model, tpdv, param.arch,
                                                tensor_feature=True, is_tensorflow=flag)
            if flag:
                extracted_feature = extracted_feature.numpy()
            else:
                extracted_feature = extracted_feature.detach().cpu().numpy()
            item_embedding = (torch.matmul(torch.tensor(extracted_feature).to(**tpdv) / normalize_feature_value, phi)
                              + item[specific_item_index]).cpu().numpy()
        else:
            # Use the existing embeddings directly
            item_embedding = (torch.matmul(ori_item_embed[specific_item_index], phi)
                              + item[specific_item_index]).detach().cpu().numpy()

    elif model_name == 'DVBPR':
        if item_image_path:
            item_embedding = Feature_Extract(item_image_path, feature_model, tpdv, param.arch,
                                             tensor_feature=True, is_tensorflow=flag)
            if flag:
                item_embedding = item_embedding.numpy()
            else:
                item_embedding = item_embedding.detach().cpu().numpy()
        else:
            item_embedding = item_image[specific_item_index].detach().cpu().numpy()

    elif model_name == 'DeepStyle':
        if item_image_path:
            extracted_feature = Feature_Extract(item_image_path, feature_model, tpdv, param.arch,
                                                tensor_feature=True, is_tensorflow=flag)
            if flag:
                extracted_feature = extracted_feature.numpy()
            else:
                extracted_feature = extracted_feature.detach().cpu().numpy()
            item_embedding = (item[specific_item_index] + torch.matmul(torch.tensor(extracted_feature).to(**tpdv), phi)
                              - category_bias[item_category[specific_item_index]]).cpu().numpy()
        else:
            item_embedding = (item[specific_item_index] + torch.matmul(item_image_embed[specific_item_index], phi)
                              - category_bias[item_category[specific_item_index]]).detach().cpu().numpy()

    else:
        raise ValueError(f"No recommender system model named {model_name}")

    return item_embedding


if __name__ == '__main__':

    param = args_parser()
    device = f'cuda:{param.GPU}' if torch.cuda.is_available() else 'cpu'
    tpdv = dict(dtype=torch.float32, device=device)
    file_lis = os.listdir(param.file_path)
    file_lis = sorted(file_lis, key=sort_key)
    user_preference(param.pos_txt)
    image_file_list = [f"{os.path.join(param.file_path, file)}" for file in file_lis]

    if param.model == 'VBPR' or param.model == 'AMR':
        model_weight_path = os.path.join(f"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv)
        feature_model, rec_feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model}, tpdv)
        user, ori_item_embed, item, normalize_feature_value, phi = embedding_return(param.model)
    elif param.model == 'DVBPR':
        model_weight_path = os.path.join(fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        feature_model, rec_feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model},
            custom_rec_model_path=os.path.join(model_weight_path, r"DVBPR_feature_model_epoch_best.h5"),
            custom_rec_model_type='DVBPR', tpdv=tpdv)
        user, item_image = embedding_return(param.model)
    elif param.model == 'DeepStyle':
        model_weight_path = os.path.join(
            fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        feature_model, rec_feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model},
            custom_rec_model_path=os.path.join(model_weight_path, r"DeepStyle_feature_model_epoch_best.h5"),
            custom_rec_model_type='DeepStyle', tpdv=tpdv)
        user, item, category_bias, item_category, phi, item_image_embed = embedding_return(param.model)
    if param.dataset == 'amazon_men':
        target_user_items_dict = generate_target(0, param.user_number)
    elif param.dataset == 'amazon_women':
        target_user_items_dict = generate_target(10, param.user_number)
    else:
        raise ValueError(f"There is no dataset named {param.dataset}")
    user_id = list(target_user_items_dict.keys())[0]
    item_set = target_user_items_dict[user_id]['item_set']
    favourite_item = target_user_items_dict[user_id]['favourite_item']
    # specific_item = input('the index of the target item:')
    # specific_item = '23213'
    specific_item = {'DeepStyle': '23213', 'VBPR': '19927', 'DVBPR': '42526'}
    user_embedding = user[user_id].detach().cpu().numpy()
    # if param.model == 'VBPR' or param.model == 'AMR':
    #     original_item_embedding = (torch.matmul(ori_item_embed[specific_item_index] / normalize_feature_value, phi) + item[specific_item_index]).detach().cpu().numpy()
    # elif param.model == 'DVBPR':
    #     original_item_embedding = item_image[specific_item_index].detach().cpu().numpy()
    # elif param.model == 'DeepStyle':
    #     original_item_embedding = (item[specific_item_index] + torch.matmul(item_image_embed[specific_item_index], phi) - category_bias[item_category[specific_item_index]]).detach().cpu().numpy()
    pca_analysis()
