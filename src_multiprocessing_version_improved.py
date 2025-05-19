import math
import os
import re

import gym
from gym.spaces import Discrete, Box, Tuple
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from collections import deque, namedtuple
import torch.nn as nn
import random
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from utils_wcy.Configuration import args_parser
import torchvision.models as models
from utils_wcy.Classification import Classify, Comput_SSIM, Feature_Extract
from copy import copy, deepcopy
from utils_wcy.env_confi import configure
from utils_wcy.distributions import Categorical
from utils_wcy.Attack import *
from utils_wcy.load_rec_generator import *
import time
import wandb
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count

run_dir = f'./results/{param.model}/{param.dataset}/{param.attack}/Subs_{param.arch}-Rec_{param.rec_arch}'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)


class PopularImageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, image, target_user=None, item_index=None, render_mode=None):
        global flag
        self.item_index = item_index
        self.target_user = target_user
        flag = False
        self.ori_image = image
        self.modify_image = image
        self.image_pre = Classify(image, model, tpdv) # the classification of the image
        if param.rec_arch == 'tensorflow_file':
            self.feature_ori = Feature_Extract(image, feature_model, tpdv, model_arch=param.rec_arch, is_tensorflow=True, adv=param.adv)
        else:
            self.feature_ori = Feature_Extract(image, feature_model, tpdv, model_arch=param.rec_arch, is_tensorflow=False, adv=param.adv)
        self.feature_adv_former = self.feature_ori
        self.feature_adv_latter = self.feature_ori
        self.image_ssim = 1.0
        # self.image_ssim = Comput_SSIM(image, image)
        self.ori_state = np.concatenate((self.image_pre, np.array([self.image_ssim])[None, ...]), axis=1)
        # self.ori_state = torch.cat((self.image_pre, torch.tensor([[self.image_ssim]], device=device)), dim=1)
        self.adv_state = self.ori_state
        self.observation_space = gym.spaces.Space(shape=(self.ori_state.shape[1],))
        # self.action_space = Tuple((
        #     Discrete(out_features_num),  # First action: selecting an item index
        #     Box(low=1, high=32, shape=(1,), dtype=np.int)  # Second action: selecting perturbation magnitude
        # ))
        self.action_space = Tuple((
            Discrete(out_features_num),  # First action: selecting an item index
            Box(low=1, high=8, shape=(1,), dtype=np.int)  # Second action: selecting perturbation magnitude
        ))
        self.training_step = 0
        self.index_list = {'ori_index':[], 'adv_index':[]}
        self.eps = param.eps
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"ori_state": self.ori_state, "adv_state": self.adv_state}

    def _get_info(self, adv_index):
        return {"change_ratio": np.linalg.norm(self.adv_state.reshape(1, -1) - self.ori_state.reshape(1, -1), ord=2), 'ssim_value': self.image_ssim,
                'adv_index': adv_index}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state = self._get_obs()['ori_state']
        info = self._get_info(None)
        self.modify_image = self.ori_image
        self.feature_adv_former = self.feature_ori
        self.feature_adv_latter = self.feature_ori

        self.index_list = {'ori_index': [], 'adv_index': []}
        self.training_step = 0
        return state, info

    def step(self, action, use_ensemble=None, ensemble_attack=None):
        self.target, self.perturbation = action

        if not isinstance(self.target, torch.LongTensor):
            self.target = self.target.long()
        if use_ensemble and ensemble_attack is not None:
            self.modify_image = ensemble_attack.apply_weighted_ensemble(self.modify_image, (self.perturbation + 1) / 255, param.rec_arch)
            # self.modify_image = ensemble_attack.ensemble_perturb(
            #     self.modify_image, (self.perturbation + 1) / 255, self.target
            # )
        elif use_ensemble:
            self.modify_image = ensemble_perturb(model_list,
                self.modify_image, (self.perturbation + 1) / 255, self.target, param.rec_arch,
            )
        else:
            self.modify_image = Add_perturb(model, self.modify_image, (self.perturbation + 1) / 255,
                                            attack_method='FGSM', target=self.target)  # 会返回的是numpy类型

        # self.modify_image = add_intermediate_perturb(model, self.modify_image, 16/255, self.target, (self.perturbation.cpu() + 1)/255)
        # self.modify_image = add_intermediate_perturb(model, self.modify_image, 16/255, self.target, self.perturbation.cpu() + 1)
        # self.modify_image = Add_perturb(model, self.modify_image, 16/255, attack_method='PGD', iteration_number=2, eps_iter=(self.perturbation.cpu() + 1)/255 ,target=self.target)  # 会导致内存泄露

        self.image_ssim = Comput_SSIM(self.ori_image, self.modify_image, model=param.rec_arch)
        self.image_pre = Classify(self.modify_image, model, tpdv)  # 会导致内存泄露 这边需要考虑是否要进行正则化
        if param.rec_arch == 'tensorflow_file':
            self.feature_adv_latter = Feature_Extract(self.modify_image, feature_model, tpdv, model_arch=param.rec_arch, is_tensorflow=True, adv=param.adv)
        else:
            self.feature_adv_latter = Feature_Extract(self.modify_image, feature_model, tpdv, model_arch=param.rec_arch, is_tensorflow=False, adv=param.adv)

        self.adv_state = np.concatenate((self.image_pre, np.array([self.image_ssim])[None, ...]), axis=1)
        # self.adv_state = torch.cat((self.image_pre, torch.tensor([[self.image_ssim]], device=device)), dim=1)
        score, adv_index, ori_index = ori_prediction(self.feature_adv_former, self.feature_adv_latter, self.item_index, self.target_user, tpdv, param.model)
        self.feature_adv_former = self.feature_adv_latter
        self.index_list['ori_index'].append(ori_index)
        self.index_list['adv_index'].append(adv_index)
        reward = score + self.eps * math.tan(math.pi/2 * (self.image_ssim - 1))
        done = False
        truncated = False
        self.training_step += 1
        if adv_index == 1:
            done = True
            # print("商品以第一名被推荐！")
            # Save_img(self.modify_image)
        if not done and self.training_step == store_step:
            truncated = True
            # Save_img(self.modify_image)
        info = self._get_info(adv_index)
        return self.adv_state, reward, done, truncated, info


    def save_image(self, epoch):
        to_pil = ToPILImage()
        img_pil = to_pil(self.modify_image.squeeze())
        plt.figure()
        plt.imshow(img_pil, cmap='gray')
        plt.savefig(f'./result/process_img/process_{epoch}.png')
        plt.show()
        print('保存了一张图片')



class Actor(nn.Module):
    def __init__(self, feature_num, n_actions):
        super(Actor, self).__init__()
        self.shared_base = nn.Sequential(
            nn.Linear(feature_num, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Shared dropout layer
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Shared dropout layer
            nn.Linear(256, 128),  # Shared 128-unit layer
            nn.ReLU(),
            nn.Dropout(p=0.5)    # Shared dropout layer
        )
        # Action-specific layers for the first action
        self.action1_layers = nn.Sequential(
            nn.Linear(128, 128),  # Custom layers for action 1
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),  # Custom action 1 layer
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        # Action-specific layers for the second action (perturbation size)
        self.action2_layers = nn.Sequential(
            nn.Linear(128, 64),  # Custom layers for action 2
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),  # Custom action 2 layer
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.action_outs = nn.ModuleList([
            Categorical(128, n_actions[0], 0.01),   # Action 1 categorical distribution
            Categorical(32, n_actions[1], 0.01)           # Action 2 categorical distribution (perturbation size)
        ])



    def forward(self, x, action=None, deterministic=False):
        x = self.shared_base(x)
        actions = []
        action_log_probs = []
        action_prob_distribution = []
        entropy_list = []
        for index, action_out in enumerate(self.action_outs, start=1):  # 这个action_outs存放的就是动作的模块
            if index == 1:
                action_pred = self.action1_layers(x)
            else:
                action_pred = self.action2_layers(x)
                # Compute the action distribution
            action_dist = action_out(action_pred)
            entropy = action_dist.entropy()  # Entropy for exploration

            # If a specific action is provided (e.g., random), fetch log_prob and prob distribution
            if action is not None:
                selected_action = action[:, index - 1].unsqueeze(-1)  # Extract the corresponding dimension
                action_log_prob = action_dist.log_prob(selected_action)  # Log probability of the given action
            else:
                # Sample action from the distribution
                selected_action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_prob(selected_action)

            # Collect outputs
            actions.append(selected_action)
            action_log_probs.append(action_log_prob)
            action_prob_distribution.append(action_dist.probs)
            entropy_list.append(entropy)
        actions = torch.cat(actions, dim=-1)
        action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
        total_entropy = torch.cat(entropy_list, dim=-1).sum(dim=-1, keepdim=True)

        return actions, action_log_probs, action_prob_distribution, total_entropy


class Critic(nn.Module):
    def __init__(self, feature_num):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)  # Added another layer for more depth
        self.v_out = nn.Linear(128, 1)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

        # Layer normalization to stabilize training
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.layer_norm3 = nn.LayerNorm(128)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = self.layer_norm1(x)
        x = self.dropout(x)  # Applying dropout

        x = F.leaky_relu(self.fc2(x))
        x = self.layer_norm2(x)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.layer_norm3(x)
        x = self.dropout(x)

        # Output the state value
        state_value = self.v_out(x)

        return state_value

class AC():
    def __init__(self, feature_num, n_actions):
        self.parameters = configure()
        self.n_actions = n_actions
        self.eval_actor = Actor(feature_num=feature_num, n_actions=n_actions).to(device)
        self.eval_critic, self.target_critic = Critic(feature_num=feature_num).to(device), Critic(feature_num=feature_num).to(device)
        self.target_critic.load_state_dict(self.eval_critic.state_dict())


        # self.actor_optim = torch.optim.Adam(self.eval_actor.parameters(), lr=self.parameters.ac_lr)
        # self.critic_optim = torch.optim.Adam(self.eval_critic.parameters(), lr=self.parameters.cr_lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.eval_actor.parameters(), 'lr': self.parameters.ac_lr},
            {'params': self.eval_critic.parameters(), 'lr': self.parameters.cr_lr}
        ])
        self.loss_fn = torch.nn.MSELoss().to(device)

        self.memory_buffer = deque(maxlen=self.parameters.buffer)

    def choose_action(self, state):
        if random.random() < self.parameters.epsilon:
            # Select random action from each degree
            random_action_label_degree = torch.tensor([random.randint(0, self.n_actions[0]-1)]).unsqueeze(-1).to(device)
            random_action_perturbation_degree = torch.tensor([random.randint(0, self.n_actions[1]-1)]).unsqueeze(-1).to(device)
            random_action = torch.cat((random_action_label_degree, random_action_perturbation_degree), dim=-1)

            _, action_log_probs, action_prob_distribution, entropy_loss = self.eval_actor(state, action=random_action)
            return random_action, action_log_probs, action_prob_distribution, entropy_loss
        action, action_log_probs, action_prob_distribution, entropy_loss = self.eval_actor(state)
        return action, action_log_probs, action_prob_distribution, entropy_loss

    def update_epsilon(self):
        self.parameters.epsilon = max(0.01, 0.9 * self.parameters.epsilon)
        # self.parameters.epsilon = max(0.01, 0.95 * self.parameters.epsilon)

    def store_transition(self, *transition):
        observation, action, action_log, done, reward, next_state, entropy = transition
        self.memory_buffer.append(experience(observation, action, action_log, done, reward, next_state, entropy))

    def clear_transition(self):
        self.memory_buffer.clear()

    def update_critic(self, *training_experience):
        state, action, reward, state_new, mask = training_experience
        target_value = self.eval_critic(state).to(**tpdv)
        state_all = torch.cat((state, state_new[-1].unsqueeze(0)), dim=0)
        value_pred = self.target_critic(state_all).detach()
        returns = torch.zeros_like(reward).to(**tpdv)
        # target_value = reward + self.target_critic(state_new).detach() * self.parameters.gamma
        gae = 0
        for step in reversed(range(reward.shape[0])):
            td_delta = reward[step] + self.parameters.gamma * value_pred[step + 1] * mask[step + 1] - value_pred[step]
            gae = td_delta + self.parameters.gamma * self.parameters.gae_lambda * mask[step + 1] * gae
            returns[step] = gae + value_pred[step]
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        self.critic_loss = self.loss_fn(returns, target_value)
        # self.critic_optim.zero_grad()
        # critic_loss.backward()
        # self.critic_optim.step()
        return returns.detach() - target_value.detach()

    def update_actor(self, actions_log, theta_value):
        self.actor_loss = -torch.mean(torch.sum(actions_log * theta_value, dim=1, keepdim=True))
        pass
        # self.actor_optim.zero_grad()
        # actor_loss.backward()
        # self.actor_optim.step()

    def update_param(self):
        loss = self.actor_loss + self.critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_net(self, *training_experience):
        state, actions_log, returns, advantage, entropy = training_experience
        target_value = self.eval_critic(state).to(**tpdv)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        actions_log = actions_log.squeeze()
        target_value = target_value.squeeze()
        self.critic_loss = self.loss_fn(returns, target_value)
        self.actor_loss = -torch.mean(actions_log * advantage.detach()) - parameters.ent_coef * torch.mean(entropy)
        loss = self.actor_loss + self.critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), max_norm=0.5)
        self.optimizer.step()

    def update_network(self):
        for target_para, para in zip(self.target_critic.parameters(), self.eval_critic.parameters()):
            target_para.data.copy_((1 - self.parameters.tau) * target_para.data + self.parameters.tau * para.data)

    def save_network(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.eval_actor.state_dict(), file_path + f"actor_{timestamp}.pt")
        torch.save(self.eval_critic.state_dict(), file_path + f"critic_{timestamp}.pt")

    def load_network(self, file_path):
        file_list = os.listdir(file_path)
        for file in file_list:
            if 'actor' in file_list:
                self.eval_actor.load_state_dict(torch.load(os.path.join(file_path, file)))
            else:
                self.target_critic.load_state_dict(torch.load(os.path.join(file_path, file)))
                self.eval_critic.load_state_dict(torch.load(os.path.join(file_path, file)))

    def calculate_gae(self, episode_length):
        numpy_episode_length = np.array(episode_length)
        all_returns = []
        all_advantages = []
        all_td_errors = []

        # Iterate over each episode to calculate GAE for its transitions
        for index in range(len(episode_length)):
            if episode_length[index] == 0:
                continue
            if not episode_length:
                start_position = 0
                end_position = self.parameters.buffer
            elif index == 0:
                start_position = 0
                end_position = numpy_episode_length[0]
            elif index == len(episode_length) - 1 and numpy_episode_length.sum() != self.parameters.buffer:
                start_position = numpy_episode_length[:index].sum()
                end_position = self.parameters.buffer
            else:
                start_position = numpy_episode_length[:index].sum()
                end_position = numpy_episode_length[:index + 1].sum()

            # Fetch the transitions for the current episode
            episode_transitions = [self.memory_buffer[i] for i in range(start_position, end_position)]

            rewards = torch.tensor([t.reward for t in episode_transitions]).to(**tpdv)
            state_np = np.array([t.state for t in episode_transitions])
            next_state_np = np.array([t.state for t in episode_transitions])

            state = torch.tensor(state_np).to(**tpdv).squeeze()
            next_state = torch.tensor(next_state_np).to(**tpdv).squeeze()
            masks = torch.tensor([1 - t.done for t in episode_transitions]).to(**tpdv)
            values = self.eval_critic(state).squeeze()
            next_values = self.target_critic(next_state).detach().squeeze()

            if values.ndim == 0:  # Add batch dimension if the tensor is 0-dimensional
                values = values.unsqueeze(0)
            if next_values.ndim == 0:  # Add batch dimension for next_state if needed
                next_values = next_values.unsqueeze(0)

            td_errors = rewards + self.parameters.gamma * next_values * masks - values
            # Calculate GAE for the current episode
            returns, advantages = self._compute_gae(rewards, values, masks, next_values)
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_td_errors.append(td_errors)
        all_returns = torch.cat(all_returns)
        all_advantages = torch.cat(all_advantages)
        all_td_errors = torch.cat(all_td_errors)

        return all_returns, all_advantages, all_td_errors

    def _compute_gae(self, rewards, values, masks, next_value):
        advantages = torch.zeros_like(rewards).to(**tpdv)
        returns = torch.zeros_like(rewards).to(**tpdv)
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.parameters.gamma * next_value[step] * masks[step] - values[step]
            gae = delta + self.parameters.gamma * self.parameters.gae_lambda * masks[step] * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]

        return returns, advantages


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def save_action_prob(result_file_path, image_name, action_prob_distribution, epoch, training_steps):
    index_action_prob = {'target_index_prob': []}
    eps_action_prob = {'target_eps_prob':[]}
    index_action_prob['target_index_prob'] = action_prob_distribution[0].cpu().detach().numpy().squeeze().tolist()
    eps_action_prob['target_eps_prob'] = action_prob_distribution[1].cpu().detach().numpy().squeeze().tolist()
    df_index_action_prob = pd.DataFrame(index_action_prob)
    df_eps_action_prob = pd.DataFrame(eps_action_prob)
    file_path = os.path.join(result_file_path, 'action_distribution')
    # file_path = f'./results/{param.dataset}/{param.attack}/Subs_{param.arch}-Rec_{param.rec_arch}/action_distribution'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    index_csv_path = os.path.join(file_path, f"{image_name}_{epoch}_index_action_prob_{training_steps}.csv")
    eps_csv_path = os.path.join(file_path, f"{image_name}_{epoch}_eps_action_prob_{training_steps}.csv")
    df_index_action_prob.to_csv(index_csv_path)
    df_eps_action_prob.to_csv(eps_csv_path)


def check_exists(path):
    file_lis = os.listdir(path)
    pattern = r'(\d+)_adv'
    exist_file_name = []
    for file in file_lis:
        if file.endswith('.jpg'):
            match = re.search(pattern, file)
            exist_file_name.append(int(match.group(1)))
        else:
            continue
    return exist_file_name

def load_models():
    """Load a list of pre-trained models to be used for generating perturbations."""
    models_list = []

    # # Load ResNet18
    # resnet18 = models.resnet18(pretrained=True).eval().to(f"cuda:{param.GPU}")
    # models_list.append(resnet18)
    #
    # # Load ResNet50
    # resnet50 = models.resnet50(pretrained=True).eval().to(f"cuda:{param.GPU}")
    # models_list.append(resnet50)
    #
    # # Load VGG16
    # vgg16 = models.vgg16(pretrained=True).eval().to(f"cuda:{param.GPU}")
    # models_list.append(vgg16)

    # Load CNN-F model
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to True for the first GPU (or the one specified)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[param.GPU]} for TensorFlow model")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Load the CNN-F model (TensorFlow/Keras)
    with tf.device(f"/GPU:{param.GPU}"):
        cnn_f_model = load_model(rf'/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images/{param.model.upper()}_classifier.h5')

    # Add the CNN-F model to the list of models
    models_list.append(cnn_f_model)

    return models_list


def process_item(items, user_id, user_info, param, eps_discount, result_file_path, file_lis):
    global model, feature_model, out_features_num, tpdv, device, store_step, experience, parameters, model_list
    parameters = configure()
    if param.use_ensemble:
        model_list = load_models()
        # from class_object.WeightedEnsembleAttack import WeightedEnsembleAttack
        # ensemble_attack = WeightedEnsembleAttack(models)
    experience = namedtuple("experience", "state action action_log done reward next_state entropy")
    store_step = parameters.training_step
    device = f'cuda:{param.GPU}' if torch.cuda.is_available() else 'cpu'
    tpdv = dict(dtype=torch.float32, device=device)
    file_lis = sorted(file_lis, key=sort_key)
    image_file_list = [f"{os.path.join(param.file_path, file)}" for file in file_lis]
    user_preference(param.pos_txt)
    if param.model == 'VBPR' or param.model == 'AMR':
        model_weight_path = os.path.join(f"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv)
        model, feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model}, tpdv)
    elif param.model == 'DVBPR':
        model_weight_path = os.path.join(fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        model, feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model},
            custom_rec_model_path=os.path.join(model_weight_path, r"DVBPR_feature_model_epoch_best.h5"),
            custom_rec_model_type='DVBPR', tpdv=tpdv)
    elif param.model == 'DeepStyle':
        model_weight_path = os.path.join(
            fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        model, feature_model = initialize_models(
            {'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name': param.model},
            custom_rec_model_path=os.path.join(model_weight_path, r"DeepStyle_feature_model_epoch_best.h5"),
            custom_rec_model_type='DeepStyle', tpdv=tpdv)
    if isinstance((list(model.children())[-1]), nn.Sequential):
        out_features_num = (list(model.children())[-1])[-1].out_features
    else:
        out_features_num = (list(model.children())[-1]).out_features
    user_id_list = []
    item_id_list = []
    ssim_value_list = []
    ori_index_list = []
    final_index_list = []
    if param.continue_running and os.path.exists(result_file_path):
        exists_file = [f for f in os.listdir(result_file_path) if f.endswith('.npy')]
    else:
        exists_file = None
    for item in items:
        user_id_list.append(user_id)
        item_id_list.append(item)
        training_recode = {'epoch': [], 'rewards': []}
        total_training_steps = 0
        initial_ssim = 0
        initial_rank = 1e5
        update_time = 0
        num_steps = 0
        env = PopularImageEnv(os.path.join(param.file_path, file_lis[item]), item_index=item, target_user=user_id)
        feature_num = env.observation_space.shape[0]
        first_action_num = env.action_space[0].n
        second_action_num = int(env.action_space[1].high[0])
        n_actions = (first_action_num, second_action_num)
        ac = AC(feature_num, n_actions)
        pbar = tqdm(total=parameters.epoch, desc="Processing Items", unit="epoch", ncols=160,
                    postfix=dict(original_index=None, best_index=None, best_ssim=None))
        # Set epsilon based on item part
        if item in user_info['first_part']:
            env.eps = eps_discount[0]
        elif item in user_info['second_part']:
            env.eps = eps_discount[1]
        elif item in user_info['third_part']:
            env.eps = eps_discount[2]
        action_trajectories = []
        episode_length = []
        for epoch in range(parameters.epoch):
            if exists_file is not None and param.continue_running:
                attacked_file_name = f"attack_img_{file_lis[item].split('.')[0]}.npy"
                if attacked_file_name in exists_file:
                    attacked_file_path = os.path.join(result_file_path, attacked_file_name)
                    perturbed_image = np.load(attacked_file_path)
                    initial_ssim = Comput_SSIM(env.ori_image, perturbed_image, param.rec_arch)
                    if param.rec_arch == 'tensorflow_file':
                        feature_adv_latter = Feature_Extract(perturbed_image, feature_model, tpdv,
                                                             model_arch=param.rec_arch, is_tensorflow=True, adv=param.adv)
                    else:
                        feature_adv_latter = Feature_Extract(perturbed_image, feature_model, tpdv,
                                                             model_arch=param.rec_arch, is_tensorflow=False, adv=param.adv)

                    _, adv_index, ori_index = ori_prediction(env.feature_adv_former, feature_adv_latter,
                                                             env.item_index, env.target_user, tpdv, param.model)
                    initial_rank = adv_index
                    pbar.set_postfix(original_index=f"{ori_index}", best_index=f"{initial_rank}",
                                     best_ssim=f"{initial_ssim}")
                    break
            observation, info = env.reset()
            total_reward = 0.0
            current_epoch_actions = []
            record_step = 0
            for training_step in range(parameters.training_step):
                record_step += 1
                total_training_steps += 1
                num_steps += 1
                observation = check(observation).to(**tpdv)
                action, action_log_probs, action_prob_distribution, entropy_loss = ac.choose_action(observation)

                current_epoch_actions.append(action.cpu().numpy())
                if param.save_prob and (total_training_steps % 1 == 0):
                    save_action_prob(result_file_path, image_name=item, action_prob_distribution=action_prob_distribution,
                                     epoch=epoch, training_steps=total_training_steps)
                if param.use_ensemble:
                    observation_new, reward, terminated, truncated, info = env.step(action[0], param.use_ensemble)
                # if param.use_ensemble and epoch > 200:
                #     target_image = env.modify_image
                #     perturbed_image = ensemble_attack.generate_individual_perturbations(target_image, (action[0][1] + 1) /255, action[0][0])
                #     for index, image in enumerate(perturbed_image):
                #         if param.rec_arch == 'tensorflow_file':
                #             feature_adv_latter = Feature_Extract(image, feature_model, tpdv,
                #                                                       model_arch=param.rec_arch, is_tensorflow=True,
                #                                                       adv=param.adv)
                #         else:
                #             feature_adv_latter = Feature_Extract(image, feature_model, tpdv,
                #                                                       model_arch=param.rec_arch, is_tensorflow=False,
                #                                                       adv=param.adv)
                #         score, _, _ = ori_prediction(env.feature_adv_former, feature_adv_latter,
                #                                                      env.item_index, env.target_user, tpdv)
                #         ensemble_attack.update_counts(index, score > 0)
                #     observation_new, reward, terminated, truncated, info = env.step(action[0], param.use_ensemble, ensemble_attack)
                else:
                    observation_new, reward, terminated, truncated, info = env.step(action[0])

                ac.store_transition(observation.cpu().numpy(), action, action_log_probs, terminated, reward,
                                    observation_new, entropy_loss)
                observation = observation_new
                done = terminated

                # Update model if buffer is full
                if len(ac.memory_buffer) > parameters.buffer:
                    update_time += 1
                    returns, advantages, td_errors = ac.calculate_gae(episode_length)
                    # Compute sampling probabilities based on TD errors (absolute value)
                    td_errors_abs = torch.abs(td_errors)  # Take absolute value of TD errors
                    sampling_probs = td_errors_abs / torch.sum(td_errors_abs)  # Normalize to get probabilities

                    # Sample transitions based on probabilities
                    sampled_indices = torch.multinomial(sampling_probs, parameters.batch_size, replacement=False)
                    experiences = [ac.memory_buffer[i] for i in sampled_indices]

                    returns = torch.tensor([returns[i] for i in sampled_indices]).to(**tpdv)
                    advantages = torch.tensor([advantages[i] for i in sampled_indices]).to(**tpdv)
                    states = torch.tensor(np.stack([e.state for e in experiences if e is not None])).to(**tpdv)
                    actions_log = torch.stack([e.action_log for e in experiences if e is not None]).to(**tpdv)
                    entropy = torch.stack([e.entropy for e in experiences if e is not None])


                    ac.update_net(states, actions_log, returns, advantages, entropy)
                    ac.clear_transition()
                    record_step = 0
                    ac.update_network()
                    episode_length.clear()
                    # if update_time % parameters.update_net == 0:
                    #     ac.update_network()
                total_reward += reward
                ori_index = env.index_list['ori_index'][0]

                if info['adv_index'] < initial_rank:
                    Save_img(env.modify_image, user_id, item, mode=None)
                    initial_rank = info['adv_index']
                    initial_ssim = info['ssim_value']
                    pbar.set_postfix(original_index=f"{ori_index}", best_index=f"{initial_rank}",
                                     best_ssim=f"{initial_ssim}")
                elif info['adv_index'] == initial_rank and info['ssim_value'] > initial_ssim:
                    Save_img(env.modify_image, user_id, item, mode=None)
                    initial_ssim = info['ssim_value']
                    pbar.set_postfix(original_index=f"{ori_index}", best_index=f"{initial_rank}",
                                     best_ssim=f"{initial_ssim}")

                if done or truncated:
                    break
            ac.update_epsilon()
            episode_length.append(record_step)
            pbar.update(1)
            training_recode['epoch'].append(epoch + 1)
            training_recode['rewards'].append(round(total_reward, 2))
            action_trajectories.append(current_epoch_actions)
            if len(action_trajectories) > 3:
                action_trajectories.pop(0)  # Remove the first (oldest) trajectory

            # Check if the last three epochs have identical action trajectories
            if len(action_trajectories) == 3 and epoch > 280:
                if np.array_equal(action_trajectories[-1], action_trajectories[-2]) and np.array_equal(
                        action_trajectories[-2], action_trajectories[-3]):
                    print(f"Agent's policy has converged after epoch {epoch + 1}. Stopping training.")
                    break  # Exit the epoch loop if the policy has converged
        env.close()
        # ensemble_attack.reset()
        if not param.continue_running:
            df_trRec = pd.DataFrame(training_recode)
            csv_file = os.path.join(result_file_path, f'training_csv_file')
            if not os.path.exists(csv_file):
                os.makedirs(csv_file)
            df_trRec.to_csv(os.path.join(csv_file, f'{item}_TrainingInfo.csv'), index=False)
        pbar.close()
        ssim_value_list.append(initial_ssim)
        ori_index_list.append(ori_index)
        final_index_list.append(initial_rank)

    # Return relevant metrics for this item
    return {
        'user_id': user_id_list,
        'item': item_id_list,
        'ssim_value': ssim_value_list,
        'ori_index': ori_index_list,
        'final_index': final_index_list
    }

def train(data, attack_type):
    global total_training_steps
    WSS_list = []
    WPS_list = []
    AE_list = []
    HR_1_list = []
    HR_10_list = []
    HR_100_list = []
    eps_discount = parameters.eps
    img_info = {'user_id': [], 'image_name': [], 'SSIM_Value': [], 'original_index': [], 'final_index': [], 'HR_1': [], 'HR_10': [], "HR_100": [], "WPS": [], "WSS": [], "AE": []}

    if parameters.num_workers < cpu_count():
        num_workers = parameters.num_workers
    else:
        num_workers = cpu_count()
    pbar = tqdm(total=len(data), desc="Processing Users", unit="user")

    with Pool(processes=num_workers) as pool:
        for index, (user_id, user_info) in enumerate(data.items(), start=1):
            print(f"Processing user: {user_id}")
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

            result_file_path = rf'./results/{param.model}/{param.dataset}/{param.attack}/Subs_{param.arch}-Rec_{param.rec_arch}/user_id{user_id}'
            total_items = len(user_info['item_set'])
            items = user_info['item_set']
            tasks = np.array_split(items, num_workers)

            batch_tasks = []
            for task_items in tasks:
                batch_tasks.append((task_items, user_id, user_info, param, eps_discount, result_file_path, file_lis))
            results_list = pool.starmap(process_item, batch_tasks)

            for result in results_list:
                final_index_list = result['final_index']
                ssim_value_list = result['ssim_value']
                ori_index_list = result['ori_index']
                item_list = result['item']
                for pos in range(len(final_index_list)):
                    final_index = final_index_list[pos]
                    ssim_value = ssim_value_list[pos]
                    ori_index = ori_index_list[pos]
                    item = item_list[pos]
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
            HR_1_list.append(HR_1 / total_items)
            HR_10_list.append(HR_10 / total_items)
            HR_100_list.append(HR_100 / total_items)
            print(
                f'Attack_type:{attack_type}, HR_1:{HR_1}, HR_10:{HR_10}, HR_100:{HR_100}, WSS:{WSS}, WPS:{WPS}, AE:{AE}')
            df = pd.DataFrame(img_info)
            df.to_csv(
                fr'./results/{param.model}/{param.dataset}/{attack_type}/Subs_{param.arch}-Rec_{param.rec_arch}/attack_img_info_real_time.csv',
                index=False)
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
        rec_feature_model = nn.Sequential(*list(rec_ic_model.children())[:-1]).to(**tpdv)
    else:
        raise ValueError(
            "Parameter 'rec_arch' (recommendation model architecture) is required if not loading a custom model.")
    return model, rec_feature_model

def sort_key(filename):
    # Extract the numeric part of the filename between '0' and '.jpg'
    return int(filename.split('.')[0])


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    if param.use_wandb:
        run = wandb.init(config=configure(),
                         project='RL-Attack-Rec',
                         name=f"{param.dataset}_{param.attack}_SubstituteRec-{param.arch}_Rec-{param.rec_arch}",
                         job_type="training",
                         dir=run_dir,
                         reinit=True)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    current_path = os.path.dirname(os.path.realpath(__file__))

    value_pred = []
    param = args_parser()
    parameters = configure()
    store_step = parameters.training_step
    device = f'cuda:{param.GPU}' if torch.cuda.is_available() else 'cpu'
    tpdv = dict(dtype=torch.float32, device=device)
    file_lis = os.listdir(param.file_path)
    file_lis = sorted(file_lis, key=sort_key)
    image_file_list = [f"{os.path.join(param.file_path, file)}" for file in file_lis]
    user_preference(param.pos_txt)

    if param.model == 'VBPR' or param.model == 'AMR':
        model_weight_path = os.path.join(f"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv)
        # model, feature_model = initialize_models({'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name':param.model}, tpdv)
    elif param.model == 'DVBPR':
        model_weight_path = os.path.join(fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
        # model, feature_model = initialize_models({'arch': param.arch, 'rec_arch': param.rec_arch, 'model_name':param.model}, custom_rec_model_path=fr"D:\Recommender System Project\TAaMR-master\rec_model_weights\amazon_men\original_images\DVBPR_feature_model_epoch_best.h5", custom_rec_model_type='DVBPR', tpdv=tpdv)
    elif param.model == 'DeepStyle':
        model_weight_path = os.path.join(fr"/home/lxh/wcy/TAaMR-master/rec_model_weights/{param.dataset}/original_images")
        load_embedding(model_weight_path, param.model, param.adv, tpdv, image_file_path=image_file_list, batch_size=128)
    if param.dataset == 'amazon_men':
        target_user_items_dict = generate_target(0, param.user_number, model=param.model, release_memory=True)
    elif param.dataset == 'amazon_women':
        target_user_items_dict = generate_target(10, param.user_number, release_memory=True)
    else:
        raise ValueError(f"There is no dataset named {param.dataset}")
    Save_user_items_info(target_user_items_dict)

    release_resources()
    train(target_user_items_dict, param.attack)
