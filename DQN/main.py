import random
from collections import namedtuple
import time

import gym
from gym import wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

np.random.seed(283)

ENV = 'MountainCar-v0'
GAMMA = 0.9
MAX_STEP = 200
NUM_EPISODES = 10000
BATCH_SIZE = 64
CAPACITY = 10000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        # ニューラルネットワーク
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, 32))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('fc4', nn.Linear(32, 32))
        self.model.add_module('relu4', nn.ReLU())
        self.model.add_module('fc5', nn.Linear(32, num_actions))
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # モデルを推論モードに
        self.model.eval()
        state_action_values = self.model(state_batch).gather(1, action_batch)
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + GAMMA * next_state_values
        # モデルを訓練モードに切り替え
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        # ε-greedy法で徐々に最適行動を採用するようにする
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # 推論モードに
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action


class Agent:
    def __init__(self, num_states, num_actions):

        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)

class Environment:

    def __init__(self):
        self.env = gym.make(ENV)
        self.env = wrappers.Monitor(self.env, './log/', force=True, video_callable=(lambda episode: episode % 10 == 0))
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        before = time.time()
        # 全エピソードループ
        for episode in range(NUM_EPISODES):
            # エピソード毎に環境を初期化
            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEP):
                self.env.render()
                action = self.agent.get_action(state, episode)
                observation_next, reward, done, _ = self.env.step(action.item())
                print('Episode: {0}, step: {1}, reward: {2}'.format(episode, step, reward))
                # 報酬を与える
                if done:
                    state_next = None
                    # 直前10エピソードで立てた平均ステップ数を格納
                    episode_10_list = np.hstack((episode_10_list[1:], step+1))
                    reward = torch.FloatTensor([reward])
                    if step < MAX_STEP - 1:
                        complete_episodes += 1
                        break
                    else:
                        complete_episodes = 0
                        break
                else:
                    reward = torch.FloatTensor([reward])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()
                state = state_next
                if done or step==MAX_STEP:
                    print('{0}エピソード: {1}ステップで終了 - reward: {2}'.format(episode, step, reward))
                    break
            # 10回連続で成功したら終わりにする
            if complete_episodes == 10:
                after = time.time()
                exe_time = after - before
                print('10回連続成功')
                print('Execution Time: {0} sec'.format(round(exe_time, 5)))
                return

cartpole_env = Environment()
cartpole_env.run()