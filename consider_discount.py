import math
import os

import gym
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

run_dir = f'./logging/{param.dataset}/{param.attack}/Subs-{param.arch}_Rec-{param.rec_arch}'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)


class PopularImageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, image, render_mode=None):
        global flag
        flag = False
        self.ori_image = image
        self.modify_image = image
        self.image_pre = Classify(image, model, tpdv)
        self.feature_ori = Feature_Extract(image, feature_model, tpdv)
        self.feature_adv_former = self.feature_ori
        self.feature_adv_latter = self.feature_ori
        self.image_ssim = Comput_SSIM(image, image)
        self.ori_state = np.concatenate((self.image_pre, np.array([self.image_ssim])[None, ...]), axis=1)
        # self.ori_state = torch.cat((self.image_pre, torch.tensor([[self.image_ssim]], device=device)), dim=1)
        self.adv_state = self.ori_state
        self.observation_space = gym.spaces.Space(shape=(self.ori_state.shape[1],))
        self.action_space = gym.spaces.Discrete(out_features_num)
        self.training_step = 0
        self.index_list = {'ori_index':[], 'adv_index':[]}
        self.eps = param.eps
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"ori_state": self.ori_state, "adv_state": self.adv_state}

    def _get_info(self):
        return {"change_ratio": np.linalg.norm(self.adv_state.reshape(1, -1) - self.ori_state.reshape(1, -1), ord=2)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state = self._get_obs()['ori_state']
        info = self._get_info()
        self.modify_image = self.ori_image
        self.feature_adv_former = self.feature_ori
        self.feature_adv_latter = self.feature_ori

        self.index_list = {'ori_index': [], 'adv_index': []}
        self.training_step = 0
        return state, info

    def step(self, action):
        self.target, self.perturbation = action[0, 0], action[0, 1]

        self.modify_image = Add_perturb(model, self.modify_image, self.perturbation / 255, 'FGSM', self.target)  # 会导致内存泄露

        self.image_ssim = Comput_SSIM(self.ori_image, self.modify_image)
        self.image_pre = Classify(self.modify_image, model, tpdv)  # 会导致内存泄露
        self.feature_adv_latter = Feature_Extract(self.modify_image, feature_model, tpdv)

        self.adv_state = np.concatenate((self.image_pre, np.array([self.image_ssim])[None, ...]), axis=1)
        # self.adv_state = torch.cat((self.image_pre, torch.tensor([[self.image_ssim]], device=device)), dim=1)
        score, adv_index, ori_index = ori_prediction(self.feature_adv_former, self.feature_adv_latter, int(self.ori_image.split('\\')[-1].split('.')[0]), target_user, tpdv)
        self.feature_adv_former = self.feature_adv_latter
        self.index_list['ori_index'].append(ori_index)
        self.index_list['adv_index'].append(adv_index)
        reward = score + self.eps * math.tan(math.pi/2 * (self.image_ssim - 1))
        done = False
        truncated = False
        self.training_step += 1
        if adv_index == 1:
            done = True
            print("商品以第一名被推荐！")
            # Save_img(self.modify_image)
        if not done and self.training_step == store_step:
            truncated = True
            # Save_img(self.modify_image)
        info = self._get_info()
        return self.adv_state, reward, done, truncated, info



        # An episode is done iff the agent has reached the target
        # self.raw_image = self.raw_image_pixel.view((1, channel, height, weight))
        # self.modify_image = self.modify_image_pixel.view((1, channel, height, weight))
        # original_output = model(self.raw_image)
        # present_output = model(self.modify_image)
        # present_class = torch.argmax(present_output)
        # l2_norm = self._get_info()['perturbation']
        # if present_class != target[0]:
        #     print(f"原图像的分类类别为:{target[0]},现在对抗样本被识别为:{present_class}")
        #     terminated = True
        #     reward = 1 / l2_norm
        # else:
        #     terminated = False
        #     reward = 10 * (- torch.sigmoid(present_output[0][target[0]]) + torch.sigmoid(original_output[0][target[0]])) - 100 * l2_norm
        # observation = self._get_obs()["modify_image"]
        # info = self._get_info()
        # return observation, reward, terminated, False, info

    def save_image(self, epoch):
        to_pil = ToPILImage()
        img_pil = to_pil(self.modify_image.squeeze())
        plt.figure()
        plt.imshow(img_pil, cmap='gray')
        plt.savefig(f'./result/process_img/process_{epoch}.png')
        plt.show()
        print('保存了一张图片')



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(feature_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        action_dims = [n_actions, 32]
        action_outs = []
        for action_dim in action_dims:
            action_outs.append(Categorical(256, action_dim, 0.01))
        self.action_outs = nn.ModuleList(action_outs)


    def forward(self, x, deterministic=False):
        x = self.actor(x)
        actions = []
        action_log_probs = []
        for action_out in self.action_outs:  # 这个action_outs存放的就是四个动作的模块
            action_dist = action_out(x)
            action = action_dist.mode() if deterministic else action_dist.sample()
            action_log_prob = action_dist.log_probs(action)  # 然后这里做的就是把这个动作的概率的对数选出来
            actions.append(action)
            action_log_probs.append(action_log_prob)  # 这个action_log_prob暂时不知道啥意思；现在知道了，就是动作的对数概率
        actions = torch.cat(actions, dim=-1)
        action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)

        return actions, action_log_probs

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_out = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_value = self.v_out(x)

        return state_value

class AC():
    def __init__(self):
        self.parameters = configure()
        self.eval_actor, self.target_actor = Actor().to(device), Actor().to(device)
        self.eval_critic, self.target_critic = Critic().to(device), Critic().to(device)
        self.target_actor.load_state_dict(self.eval_actor.state_dict())
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
        action, action_log_probs = self.eval_actor(state)
        return action, action_log_probs

    def update_epsilon(self):
        self.parameters.epsilon = max(0.05, 0.9 * self.parameters.epsilon)
        # self.parameters.epsilon = max(0.01, 0.95 * self.parameters.epsilon)

    def store_transition(self, *transition):
        observation, action, action_log, done, reward, next_state = transition
        if len(self.memory_buffer) == self.parameters.buffer:
            self.memory_buffer.clear()
        self.memory_buffer.append(experience(observation, action, action_log, done, reward, next_state))

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
        state, actions_log, reward, state_new, mask = training_experience
        target_value = self.eval_critic(state.data).to(**tpdv)
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

        self.actor_loss = -torch.mean(torch.sum(actions_log * (returns - target_value).detach(), dim=1, keepdim=True))
        loss = self.actor_loss + self.critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def update_network(self):
        for target_para, para in zip(self.target_actor.parameters(), self.eval_actor.parameters()):
            target_para.data.copy_((1 - self.parameters.tau) * target_para.data + self.parameters.tau * para.data)

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
                self.target_actor.load_state_dict(torch.load(os.path.join(file_path, file)))
                self.eval_actor.load_state_dict(torch.load(os.path.join(file_path, file)))
            else:
                self.target_critic.load_state_dict(torch.load(os.path.join(file_path, file)))
                self.eval_critic.load_state_dict(torch.load(os.path.join(file_path, file)))



def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def train():
    # env = PopularImageEnv(image)
    parameters = configure()
    ac = AC()
    # best_reward = -1e5

    sample_distribution_idx = 0
    record_epoch = 0
    HR_1 = 0
    HR_10 = 0
    HR_100 = 0
    HR_200 = 0

    eps_discount = [5000, 3000, 1000, 500, 100]
    flag = False
    for eps in eps_discount:
        idx = -1
        while idx < test_sample:
            if not flag:
                idx += 1
                data = sample_data[idx]
            else:
                flag = False
                pass
        # for idx, data in enumerate(sample_data[:test_sample]):

            training_recode = {'epoch': [], 'rewards': [], 'SSIM': []}

            result_dict = {'ImageName': [], 'initial_index': [], 'final_index': [], 'SSIM': []}

            initial_ssim = 1
            initial_rank = 1e5
            update_time = 0
            num_steps = 0
            ac = AC()
            env = PopularImageEnv(os.path.join(param.file_path, file_lis[data]))
            env.eps = eps
            judge_exception = 0
            for epoch in range(parameters.epoch):
                if flag:
                    break
                record_epoch += 1
                observation, info = env.reset()
                total_reward = 0.0
                for training_step in range(parameters.training_step):
                    num_steps += 1
                    observation = check(observation).to(**tpdv)
                    action, action_log_probs = ac.choose_action(observation.detach())
                    observation_new, reward, terminated, truncated, info = env.step(action)
                    if truncated:
                        ac.store_transition(observation.cpu().numpy(), action, action_log_probs, truncated, reward,observation_new)
                    else:
                        ac.store_transition(observation.cpu().numpy(), action, action_log_probs, terminated, reward, observation_new)
                    observation = observation_new
                    done = terminated
                    # if len(ac.memory_buffer) > parameters.batch_size:
                    if num_steps % parameters.update == 0 and len(ac.memory_buffer) >= parameters.batch_size:
                        update_time += 1
                        # experiences = random.sample(ac.memory_buffer, parameters.batch_size)
                        experiences = ac.memory_buffer


                        states = torch.tensor(np.stack([e.state for e in experiences if e is not None])).to(**tpdv)

                        # actions = torch.stack([e.action[0] for e in experiences if e is not None]).to(**tpdv)
                        actions_log = torch.stack([e.action_log for e in experiences if e is not None]).to(**tpdv)

                        # rewards = torch.stack([e.reward for e in experiences if e is not None]).view(parameters.batch_size, 1).to(**tpdv)

                        rewards = torch.stack([torch.tensor(e.reward)for e in experiences if e is not None]).view(parameters.batch_size, 1, 1).to(**tpdv)

                        next_states = torch.tensor(np.stack([e.next_state for e in experiences if e is not None])).to(**tpdv)

                        done_signal = torch.stack([torch.tensor(e.done) for e in experiences if e is not None]).to(**tpdv)

                        mask = torch.ones(done_signal.shape[0] + 1, *done_signal.shape[1:]).to(**tpdv)
                        mask[1:] = torch.ones_like(done_signal).to(**tpdv) - done_signal

                        ac.update_net(states, actions_log, rewards, next_states, mask)
                        # theta_value = ac.update_critic(states, actions_log, rewards, next_states, mask)
                        # ac.update_actor(actions_log, theta_value)
                        # ac.update_param()
                        if update_time % parameters.update_net == 0:
                            ac.update_network()
                    total_reward += reward
                    if len(env.index_list['adv_index']) >= 2:
                        if env.index_list['adv_index'][-1] == env.index_list['adv_index'][-2] and initial_rank < 10:
                            judge_exception += 1
                            if judge_exception == 6:
                                flag = True
                                print("智能体异常，开始重新训练")
                        else:
                            judge_exception = 0
                    if done or truncated or flag:
                        # if total_reward >= best_reward:
                        #     best_reward = total_reward
                        break
                if (env.index_list['adv_index'][-1] - env.index_list['ori_index'][0]) > 0:
                    print(f'epoch:{epoch+1}, reward:{total_reward}, 退步了{env.index_list["adv_index"][-1] - env.index_list["ori_index"][0]}, SSIM:{env.image_ssim}, 原排名:{env.index_list["ori_index"][0]}, 现排名:{env.index_list["adv_index"][-1]}')
                else:
                    print(f'epoch:{epoch+1}, reward:{total_reward}, 进步了{-env.index_list["adv_index"][-1] + env.index_list["ori_index"][0]}, SSIM:{env.image_ssim}, 原排名:{env.index_list["ori_index"][0]}, 现排名:{env.index_list["adv_index"][-1]}')
                if env.index_list['adv_index'][-1] < initial_rank:
                    Save_img_discount_Exp(env.modify_image, file_lis[data], eps)
                    initial_rank = env.index_list['adv_index'][-1]
                    initial_ssim = env.image_ssim
                elif env.index_list['adv_index'][-1] == initial_rank and env.image_ssim > initial_ssim:
                    Save_img_discount_Exp(env.modify_image, file_lis[data], eps)
                    initial_ssim = env.image_ssim
                if param.use_wandb:
                    run.log({
                        'rewards': round(total_reward, 2)
                    }, step=record_epoch)
                training_recode['epoch'].append(epoch + 1)
                training_recode['rewards'].append(round(total_reward, 2))
                training_recode['SSIM'].append(round(env.image_ssim, 2))
            result_dict['ImageName'].append(str(data) + '.jpg')
            result_dict['initial_index'].append(env.index_list["ori_index"][0])
            result_dict['final_index'].append(initial_rank)
            result_dict['SSIM'].append(initial_ssim)
            print(f"最终保存图片{file_lis[data]}，排名为：{initial_rank},最终SSIM值为{initial_ssim}")
            if initial_rank == 1:
                HR_1 += 1
                HR_10 += 1
                HR_100 += 1
                HR_200 += 1
            elif initial_rank <= 10:
                HR_10 += 1
                HR_100 += 1
                HR_200 += 1
            elif initial_rank <= 100:
                HR_100 += 1
                HR_200 += 1
            elif initial_rank <= 200:
                HR_200 += 1
            df_result = pd.DataFrame(result_dict)
            df_trRec = pd.DataFrame(training_recode)
            if not os.path.exists(f'./results/{param.dataset}/{param.attack}/discount_reason/Substi_{param.arch}-Rec_{param.rec_arch}/eps_{eps}'):
                os.makedirs(f'./results/{param.dataset}/{param.attack}/discount_reason/Substi_{param.arch}-Rec_{param.rec_arch}/eps_{eps}')
            df_result.to_csv(f'./results/{param.dataset}/{param.attack}/discount_reason/Substi_{param.arch}-Rec_{param.rec_arch}/eps_{eps}/ImageInfo{idx}.csv', index=False)
            df_trRec.to_csv(f'./results/{param.dataset}/{param.attack}/discount_reason/Substi_{param.arch}-Rec_{param.rec_arch}/eps_{eps}/TrainingInfo{idx}.csv', index=False)
        # print(f'最终攻击效果,HR@1:{(HR_1 / len(sample_data)):.1%},HR@10:{(HR_10 / len(sample_data)):.1%},HR@100:{(HR_100 / len(sample_data)):.1%}, HR@200:{(HR_200 / len(sample_data)):.1%}')




        # if (epoch + 1) % 20 == 0:
        #     ac.update_epsilon()
    # with open('./result/result_reward.txt', 'w') as f:
    #     f.write("\n".join(result_dict['reward_lis']))
    #
    # with open('./result/result_index', 'w') as f:
    #     f.write("\n".join(result_dict['target_index']))

def sort_key(filename):
    # Extract the numeric part of the filename between '0' and '.jpg'
    return int(filename.split('.')[0])



if __name__ == '__main__':
    test_sample = 3

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
    store_step = configure().training_step
    device = f'cuda:{param.GPU}' if torch.cuda.is_available() else 'cpu'
    tpdv = dict(dtype=torch.float32, device=device)
    file_lis = os.listdir(param.file_path)
    file_lis = sorted(file_lis, key=sort_key)
    user_preference(param.pos_txt)
    load_embedding(param.weight, tpdv)
    target_user, sample_data, sample_distribution = generate_target(0)
    experience = namedtuple("experience", "state action action_log done reward next_state")

    # txt_file_path = fr'./results/{param.dataset}/{param.attack}/Subs_{param.arch}-Rec_{param.rec_arch}/Item_User_Info'
    # if not os.path.exists(txt_file_path):
    #     os.makedirs(txt_file_path)
    # file_name = []
    # for idx in sample_data:
    #     file_name.append(file_lis[idx])
    # file_name = "目标文件名为:" + ",".join(file_name) + '\n'
    # user_name = "目标用户名为:" + str(target_user) + '\n'
    # distribution_info = "样本分布情况:" + ",".join([str(dis) for dis in sample_distribution]) + '\n'
    # with open(os.path.join(txt_file_path, 'info.txt'), 'w') as f:
    #     f.writelines([file_name, user_name, distribution_info])

    # this place define the feature model of the reinforcement learning
    model = eval(f"models.{param.arch}(pretrained=True)")
    model = model.to(**tpdv)

    if isinstance((list(model.children())[-1]), nn.Sequential):
        out_features_num = (list(model.children())[-1])[-1].out_features
    else:
        out_features_num = (list(model.children())[-1]).out_features

    rec_ic_model = eval(f"models.{param.rec_arch}(pretrained=True)")
    feature_model = nn.Sequential(*list(rec_ic_model.children())[:-1]).to(**tpdv)

    env = PopularImageEnv(os.path.join(param.file_path, file_lis[0]))
    # img, info = env.reset()
    # img = img.to(**tpdv)
    feature_num = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # show the original picture
    # pic = Image.open(param.file_path)
    # pic.show()
    train()
