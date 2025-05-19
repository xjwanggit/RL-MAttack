import os

import gym
from PIL import Image
import numpy as np
from model_architecture.LeNet_Model import LeNet
import torch
import torchvision
from torch.utils.data import DataLoader
from recommendation.recommender_utils.Solver import Solver
from collections import deque, namedtuple
from confirguration import configure
import torch.nn as nn
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torchvision.models as models

from cnn.models.dataset import *
from cnn.models.model import *
from utils.read import *
from utils.write import *
import pandas as pd


class PopularImageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, image, render_mode=None):
        global flag
        flag = False
        # 如何是4维的含有这个batch_size的就删掉batch_size
        if len(image.shape) == 4:
            self.raw_image_pixel = image.squeeze(axis=0).view(-1)
        else:
            self.raw_image_pixel = image.view(-1)
        self.perturbation = torch.zeros(self.raw_image_pixel.shape, dtype=torch.float32, device=device)
        self.modify_image_pixel = self.raw_image_pixel + self.perturbation
        self.raw_rank = raw_rank



        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"raw_image": self.raw_image_pixel, "modify_image": self.modify_image_pixel}

    def _get_info(self):
        return {"perturbation": torch.linalg.norm(self.modify_image_pixel.reshape(1, -1) - self.raw_image_pixel.reshape(1, -1), ord=2)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.raw_rank = raw_rank
        observation = self._get_obs()['raw_image']
        self.modify_image_pixel = self.raw_image_pixel.detach()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.perturbation = action

        self.modify_image_pixel = self.modify_image_pixel + self.perturbation

        # An episode is done iff the agent has reached the target
        present_rate = -(np.matmul(np.expand_dims(self.modify_image_pixel.detach().cpu().numpy(), axis=0), phi).squeeze(0) + item_feature).dot(user_embedding)
        new_rating = np.append(preference_rating, present_rate)
        present_rank = np.max(np.where(np.sort(new_rating) == present_rate)[0])
        # print(f"现在的评分为:{-present_rate},现在的排名为:{present_rank}")
        terminal_state = torch.tensor(self.modify_image_pixel, dtype=torch.float32)
        terminal_index = torch.argmax(model.classify(terminal_state))
        if terminal_index != raw_index:
            if present_rank == 0:
                reward = 1000
                terminated = True
            elif present_rank == self.raw_rank:
                reward = 1
                terminated = False
            else:
                reward = self.raw_rank - present_rank
                terminated = False
            if raw_rank > present_rank:
                data['item_rank'].append(present_rank)
                data['item_index'].append(terminal_index.item())
                data['reward'].append(reward)
        else:
            reward = -1
            terminated = False
        reward = torch.tensor(reward)
        self.raw_rank = present_rank
        observation = self._get_obs()["modify_image"]
        info = self._get_info()


        return observation, reward, terminated, False, info

    def save_image(self):
        to_pil = ToPILImage()
        img_pil = to_pil(self.modify_image_pixel.squeeze())
        plt.figure()
        plt.axis("off")
        plt.imshow(img_pil)
        plt.savefig(f'process_{args.index}.png')
        plt.show()
        print('保存了一张图片')



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_num, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.out(x)) * 0.01

        return actions

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_num + n_actions, 512)
        self.fc2 = nn.Linear(512, 512)
        self.q_out = nn.Linear(512, 1)

    def forward(self, state, aciton):
        x = torch.cat([state, aciton], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)

        return q_value

class AC():
    def __init__(self):
        self.parameters = configure()
        self.eval_actor, self.target_actor = Actor().to(device), Actor().to(device)
        self.eval_critic, self.target_critic = Critic().to(device), Critic().to(device)
        self.target_actor.load_state_dict(self.eval_actor.state_dict())
        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.eval_actor.parameters(), lr=self.parameters.ac_lr)
        self.critic_optim = torch.optim.Adam(self.eval_critic.parameters(), lr=self.parameters.cr_lr)
        self.loss_fn = torch.nn.MSELoss().to(device)

        self.memory_buffer = deque(maxlen=self.parameters.buffer)


    def choose_action(self, state):
        if np.random.rand() < self.parameters.epsilon:
            action = torch.tensor(np.random.uniform(-1, 1, n_actions), device=device) * 0.01
        else:
            inputs = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action = self.eval_actor(inputs).squeeze(0)
        return action

    def update_epsilon(self):
        self.parameters.epsilon = max(0.05, 0.9 * self.parameters.epsilon)
        # self.parameters.epsilon = max(0.01, 0.95 * self.parameters.epsilon)

    def store_transition(self, *transition):
        observation, action, reward, next_state = transition
        if len(self.memory_buffer) == self.parameters.buffer:
            self.memory_buffer.popleft()
        self.memory_buffer.append(experience(observation, action, reward, next_state))

    def update_critic(self, *training_experience):
        state, action, reward, state_new = training_experience
        new_action = self.target_actor(state).detach()
        target_value = reward + self.target_critic(state_new, new_action).detach() * self.parameters.gamma
        value = self.eval_critic(state, action)
        loss = self.loss_fn(value, target_value)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def update_actor(self, states):
        loss = -torch.mean(self.eval_critic(states, self.eval_actor(states)))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def update_network(self):
        for target_para, para in zip(self.target_actor.parameters(), self.eval_actor.parameters()):
            target_para.data.copy_((1 - self.parameters.tau) * target_para.data + self.parameters.tau * para.data)

        for target_para, para in zip(self.target_critic.parameters(), self.eval_critic.parameters()):
            target_para.data.copy_((1 - self.parameters.tau) * target_para.data + self.parameters.tau * para.data)

def train(image):
    env = PopularImageEnv(image)
    parameters = configure()
    ac = AC()
    num_steps = 0
    best_reward = -1e5
    for epoch in range(parameters.epoch):
        observation, info = env.reset()
        total_reward = 0.0
        for training_step in range(parameters.training_step):
            num_steps += 1
            action = ac.choose_action(observation)
            observation_new, reward, terminated, truncated, info = env.step(action)
            ac.store_transition(observation, action, reward, observation_new)
            observation = observation_new
            done = terminated
            # if len(ac.memory_buffer) > parameters.batch_size:
            if num_steps % parameters.update == 0 and len(ac.memory_buffer) > parameters.batch_size:
                experiences = random.sample(ac.memory_buffer, parameters.batch_size)

                states = torch.stack([e.state for e in experiences if e is not None]).to(device)
                states = torch.tensor(states, dtype=torch.float32)

                actions = torch.stack([e.action for e in experiences if e is not None]).to(device)
                actions = torch.tensor(actions, dtype=torch.float32)

                rewards = torch.stack([e.reward for e in experiences if e is not None]).view(parameters.batch_size, 1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32)

                next_states = torch.stack([e.next_state for e in experiences if e is not None]).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32)

                ac.update_critic(states, actions, rewards, next_states)
                ac.update_actor(states)
                ac.update_network()
            total_reward += reward
            if done:
                terminal_state = torch.tensor(env.modify_image_pixel, dtype=torch.float32)
                terminal_index = torch.argmax(model.classify(terminal_state))
                print(f"epoch:{epoch+1}结束,达到最终状态,被识别为{terminal_index}")
                break
        print(f'epoch:{epoch+1}, reward:{total_reward}')
        if (epoch + 1) % 20 == 0:
            df = pd.DataFrame(data)
            df.to_csv('./result.csv')
            ac.update_epsilon()


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = dict()
    data['item_rank'] = []
    data['item_index'] = []
    data['reward'] = []
    args = configure()
    params = np.load(args.weight_path, allow_pickle=True)
    user_embedding = params[0][25]  # 选择了1号用户
    item_embedding = params[1]  # （82630， 64）
    item_feature = np.copy(item_embedding[args.index])  # 这个是目标商品特征
    phi = params[2]  # 用于处理图像的的嵌入特征的，将其从2048转换为64

    image_feature = np.load(args.image_feature_path, allow_pickle=True)  # 这个是图像的特征
    image_feature = image_feature / np.max(np.abs(image_feature))  # 这其中np.max()会选择整个矩阵中最大的值，相当于进行了归一化处理吧
    # print(image_feature[args.index] @ phi)
    emb_image = np.matmul(image_feature, phi)
    item_embedding += emb_image

    preference_rating = np.matmul(item_embedding, -user_embedding)
    item_rating = preference_rating[args.index]
    raw_rank = np.where(np.sort(preference_rating) == item_rating)[0].item()
    print(f"原始评分为:{-item_rating},排名为:{raw_rank}")

    experience = namedtuple("experience", "state action reward next_state")


    img_classes = read_imagenet_classes_txt('../../data/imagenet_classes.txt')
    target_data = CustomDataset(root_dir='../../data/amazon_men/original_images/images/',
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    denormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    model = Model(model=models.resnet50(pretrained=True), gpu=0) # 首先定义了一个对象
    model.set_out_layer(drop_layers=1)  # drop_layers=1，即保留模型除最后一层的结构，从而获得模型的特征
    model.set_classify_model(save_layers=1)

    image = target_data[args.index][0].to(device)
    target = torch.tensor(args.target_classes, device=device)
    feature = torch.tensor(image_feature[args.index], device=device, dtype=torch.float32)
    raw_index = torch.argmax(model.classify(feature))
    print(f"原始类别为:{raw_index}")
    # print(-1 *( feature.unsqueeze(0).cpu().numpy() @ phi + item_feature).dot(user_embedding))
    feature_num = feature.shape[0]
    n_actions = feature_num
    to_pil = ToPILImage()
    show_original_image = to_pil(denormalize(target_data[args.index][0]))
    plt.axis("off")
    plt.imshow(show_original_image)
    plt.savefig(f'original_{args.index}.png')
    plt.show()
    plt.close()
    train(feature)
