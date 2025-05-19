import numpy as np
import torch
from sklearn.decomposition import PCA
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r"/home/lxh/wcy/Reinforcement-Learning-Against-RecSys")
from utils_wcy.Configuration import args_parser
from utils_wcy.load_rec_generator import *
import torch.nn as nn
from utils_wcy.Classification import Feature_Extract
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import MDS, Isomap
import re
from sklearn.manifold import TSNE
import matplotlib.font_manager as fm


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.font_manager as fm

def draw_tsne(embedding_info_all, save_dir):
    # Legend mapping
    legend_mapping = {
        'RLRS': 'RL-MAttack',
        'c-SEMA': 'SEMA',
        'original': 'ORIGINAL',
        'favourite': 'FAVOUR',
    }

    # Scenarios and titles
    scenarios = ['first_scenario', 'second_scenario', 'third_scenario']
    scenario_titles = {
        'first_scenario': 'Scenario \u2160',
        'second_scenario': 'Scenario \u2161',
        'third_scenario': 'Scenario \u2162',
    }

    # Attack methods and colors
    attack_methods = ['RLRS', 'EXPA', 'SEMA', 'FGSM', 'PGD', 'original', 'favourite']
    colors = plt.cm.get_cmap('tab10', len(attack_methods))

    # Define a beautiful font (e.g., "DejaVu Serif")
    font_path = fm.findfont(fm.FontProperties(family='DejaVu Serif'))  # Change to "Georgia" if desired
    custom_font = fm.FontProperties(fname=font_path, size=16)

    for scenario in scenarios:
        plt.figure(figsize=(12, 10))  # Slightly larger figure for better visibility
        embeddings = []
        labels = []

        # Collect embeddings and their corresponding labels
        for attack_method in attack_methods:
            if embedding_info_all[attack_method][scenario]:
                # Add embeddings and labels for the current attack method
                method_embeddings = np.vstack(embedding_info_all[attack_method][scenario])
                embeddings.append(method_embeddings)
                # Use the mapped legend name for labels
                labels.extend([legend_mapping.get(attack_method, attack_method)] * method_embeddings.shape[0])

        # If there are no embeddings for the scenario, skip
        if not embeddings:
            print(f"No embeddings found for {scenario}. Skipping...")
            continue

        # Combine all embeddings into a single numpy array
        embeddings = np.vstack(embeddings)

        # Compute t-SNE
        tsne = TSNE(n_components=2)
        tsne_result = tsne.fit_transform(embeddings)

        # Plot t-SNE results
        for attack_method in attack_methods:
            indices = [i for i, label in enumerate(labels) if label == legend_mapping.get(attack_method, attack_method)]
            if indices:
                plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                            label=legend_mapping.get(attack_method, attack_method),
                            alpha=0.7, s=50,
                            color=colors(attack_methods.index(attack_method)))

        # Plot settings
        plt.title(scenario_titles[scenario], fontsize=20, fontproperties=custom_font, fontweight='bold')  # Bigger title
        plt.xlabel("t-SNE Dimension 1", fontsize=14, fontproperties=custom_font, labelpad=10)
        plt.ylabel("t-SNE Dimension 2", fontsize=14, fontproperties=custom_font, labelpad=10)
        plt.legend(loc='best', fontsize=16, frameon=True)

        # Beautify the grid
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
        plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3, color='gray')
        plt.minorticks_on()

        # Save the plot
        save_path = f"{save_dir}/{scenario}_tsne.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot for {scenario} saved at {save_path}")

# def draw_tsne(embedding_info_all, save_dir):
#     legend_mapping = {
#         'RLRS': 'RL-MAttack',
#         'c-SEMA': 'SEMA',
#         'original': 'ORIGINAL',
#         'favourite': 'FAVOUR',
#     }
#     scenarios = ['first_scenario', 'second_scenario', 'third_scenario']
#     attack_methods = ['RLRS', 'EXPA', 'c-SEMA', 'FGSM', 'PGD', 'original', 'favourite']
#     colors = plt.cm.get_cmap('tab10', len(attack_methods))
#
#     scenario_titles = {
#         'first_scenario': 'Scenario \u2160',
#         'second_scenario': 'Scenario \u2161',
#         'third_scenario': 'Scenario \u2162',
#     }
#     font_path = fm.findfont(fm.FontProperties(family='DejaVu Serif'))  # Change to "Georgia" if desired
#     custom_font = fm.FontProperties(fname=font_path)
#     for scenario in scenarios:
#         plt.figure(figsize=(10, 8))
#         embeddings = []
#         labels = []
#
#         # Collect embeddings and their corresponding labels
#         for attack_method in attack_methods:
#             if embedding_info_all[attack_method][scenario]:
#                 # Add embeddings and labels for the current attack method
#                 method_embeddings = np.vstack(embedding_info_all[attack_method][scenario])
#                 embeddings.append(method_embeddings)
#                 labels.extend([attack_method] * method_embeddings.shape[0])
#
#         # If there are no embeddings for the scenario, skip
#         if not embeddings:
#             print(f"No embeddings found for {scenario}. Skipping...")
#             continue
#
#         # Combine all embeddings into a single numpy array
#         embeddings = np.vstack(embeddings)
#
#         # Compute t-SNE
#         tsne = TSNE(n_components=2, random_state=42)
#         tsne_result = tsne.fit_transform(embeddings)
#
#         # Plot t-SNE results
#         for attack_method in attack_methods:
#             indices = [i for i, label in enumerate(labels) if label == attack_method]
#             if indices:
#                 plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
#                             label=attack_method, alpha=0.7, s=50,
#                             color=colors(attack_methods.index(attack_method)))
#
#         # Plot settings
#         plt.title(scenario_titles[scenario], fontsize=16, fontproperties=custom_font, fontweight='bold')
#         plt.xlabel("t-SNE Dimension 1", fontsize=12, fontproperties=custom_font, labelpad=10)
#         plt.ylabel("t-SNE Dimension 2", fontsize=12, fontproperties=custom_font, labelpad=10)
#         plt.legend(loc='best', fontsize=10, frameon=True)
#         # Beautify the grid
#         plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
#         plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3, color='gray')
#         plt.minorticks_on()
#
#         # Save the plot
#         save_path = f"{save_dir}/{scenario}_tsne.jpg"
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"t-SNE plot for {scenario} saved at {save_path}")

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

def tsne_analysis():
    attack_type = ['RLRS', 'EXPA', 'SEMA', 'FGSM', 'PGD']
    embedding_info_all = dict()
    for attack_name in attack_type:
        embedding_info_all[attack_name] = {'first_scenario': [], 'second_scenario': [], 'third_scenario':[]}
    embedding_info_all['original'] = {'first_scenario': [], 'second_scenario': [], 'third_scenario':[]}
    embedding_info_all['favourite'] = {'first_scenario': [], 'second_scenario': [], 'third_scenario':[]}
    model_name = param.model
    favourite_item_index = int(favourite_item)
    result_save_path = f'/home/lxh/wcy/Reinforcement-Learning-Against-RecSys/results/{model_name}/{param.dataset}/'
    done = True
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
            for image_name in os.listdir(os.path.join(result_save_path, attack_method, scenario_name, f"user_id{user_id}")):
                print(image_name)
                if not image_name.endswith('.jpg'):
                    continue
                specific_item_index = int(re.search(r'_(\d+).jpg', image_name).group(1))
                item_image_path = os.path.join(result_save_path, attack_method, scenario_name, f"user_id{user_id}", image_name)
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
                if done:
                    ori_embedding = get_item_embedding(model_name, specific_item_index, tpdv, rec_feature_model)
                    embedding_info_all['original']['first_scenario'].append(ori_embedding)
                    embedding_info_all['original']['second_scenario'].append(ori_embedding)
                    embedding_info_all['original']['third_scenario'].append(ori_embedding)
            done = False
    for favourite_item_index in target_user_items_dict[user_id]['favourite_items']:
        favourite_embedding = get_item_embedding(model_name, favourite_item_index, tpdv, rec_feature_model)
        embedding_info_all['favourite']['first_scenario'].append(favourite_embedding)
        embedding_info_all['favourite']['second_scenario'].append(favourite_embedding)
        embedding_info_all['favourite']['third_scenario'].append(favourite_embedding)

    draw_tsne(embedding_info_all, '.')


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
    user_embedding = user[user_id].detach().cpu().numpy()
    # if param.model == 'VBPR' or param.model == 'AMR':
    #     original_item_embedding = (torch.matmul(ori_item_embed[specific_item_index] / normalize_feature_value, phi) + item[specific_item_index]).detach().cpu().numpy()
    # elif param.model == 'DVBPR':
    #     original_item_embedding = item_image[specific_item_index].detach().cpu().numpy()
    # elif param.model == 'DeepStyle':
    #     original_item_embedding = (item[specific_item_index] + torch.matmul(item_image_embed[specific_item_index], phi) - category_bias[item_category[specific_item_index]]).detach().cpu().numpy()
    tsne_analysis()
