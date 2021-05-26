import random
from collections import namedtuple
from itertools import count
import math
import time

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from .. import save

np.random.seed(283)

# 学習に使う変数を整理
ENV = 'MountainCar-v0'
# 報酬割引率
GAMMA = 0.9
# 1試行（1エピソード）の最大ステップ数
MAX_STEP = 200
# 最大試行回数（エピソード数）
NUM_EPISODES = 10000
# バッチサイズ32
BATCH_SIZE = 64
# キャパ
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

# エージェントが行う行動を与えられた状態によって判断する部分（深層強化学習（DQN）を行う部分）
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
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)
        # 最適化手法
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    # 結合パラメータを学習する部分
    def replay(self):

        # 最初にメモリサイズを確認する
        # 指定したバッチサイズより小さい場合は何もしない
        if len(self.memory) < BATCH_SIZE:
            return
        # ミニバッチ用のデータを取得（ランダム）
        transitions = self.memory.sample(BATCH_SIZE)
        # transitions は (state, action, next_state, reward) * BATCH_SIZE
        # (state * BATCH_SIZE, action * BATCH_SIZE, next_state * BATCH_SIZE, reward * BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # 教師信号となるQ(s_t, a_t)を求める
        # モデルを推論モードに
        self.model.eval()
        # self.model(state_batch)は，2つのQ値を出力する
        # [torch.FloatTensor of size BATCH_SIZE * 2]になってるので
        # 実行したアクション（a_t）に対応するQ値をaction_batchで行った行動a_tのindexを使って取得する
        state_action_values = self.model(state_batch).gather(1, action_batch)
        # CartPole がdoneになっていない，かつ，next_stateがあるかをチェックするためのマスクを作成
        # non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        # maxQ(s_t+1, a)を求める
        next_state_values = torch.zeros(BATCH_SIZE)
        # 次の状態があるindexの最大Q値を求める
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        # Q学習の行動価値関数更新式からQ(S_t, a_t)を求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values
        # モデルを訓練モードに切り替え
        self.model.train()
        # 二乗誤差の代わりにHuber関数を使う
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # 勾配をリセット
        self.optimizer.zero_grad()
        # 誤差逆伝搬
        loss.backward()
        # ニューラルネットワークの重み更新
        self.optimizer.step()

    # 現在の状態に応じて行動を決定する
    def decide_action(self, state, episode):
        # ε-greedy法で徐々に最適行動を採用するようにする
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # 推論モードに
            self.model.eval()
            # ネットワークの出力の最大値のindexを取得
            # view関数で行列サイズを（1 * 1）に調整
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            # 右，左ランダムに行動する
            # actionは[torch.LongTensor of size 1 * 1]
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

# エージェントクラス
class Agent:
    def __init__(self, num_states, num_actions):
        # Brainクラスをインスタンス化
        self.brain = Brain(num_states, num_actions)
    # Q関数の更新
    def update_q_function(self):
        self.brain.replay()
    # アクションを決定する
    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action
    # 状態を保存
    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)

# CartPoleを実行する環境クラス
class Environment:

    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):

        episode_10_list = np.zeros(10) # 10試行分の立ち続けた平均ステップ数の出力に使う
        complete_episodes = 0  # 旗まで連続して到達した数
        is_episode_final = False  # 最終試行フラグ
        frames = []  # 動画用に画像を格納する変数
        before = time.time()

        # 全エピソードループ
        for episode in range(NUM_EPISODES):
            # エピソード毎に環境を初期化
            observation = self.env.reset()
            state = observation
            # numpyからpytorchのテンソルに変換
            state = torch.from_numpy(state).type(torch.FloatTensor)
            # size を 1*4 に変換
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEP):
                # 最終試行はframesに画像を追加しておく
                if is_episode_final:
                    frames.append(self.env.render(mode='rgb_array'))
                # 最初の行動を決める
                action = self.agent.get_action(state, episode)
                # 最初の行動から次の状態を求める
                observation_next, _, done, _ = self.env.step(action.item())
                # 報酬を与える
                if done:
                    # 次の状態はないのでNoneを代入
                    state_next = None
                    # 直前10エピソードで立てた平均ステップ数を格納
                    episode_10_list = np.hstack((episode_10_list[1:], step+1))
                    if step < 199:
                        # 規定回数までに到達できたら+200 ←特に深い理由はない(見栄えの問題)
                        reward = torch.FloatTensor([200])
                        # 連続成功回数を+1
                        complete_episodes += 1
                    else:
                        # 立ったまま終了した場合は報酬-1
                        reward = torch.FloatTensor([-1.0])
                        # 連続成功回数をリセット
                        complete_episodes = 0
                else:
                    # 途中の報酬は-1
                    reward = torch.FloatTensor([-1.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)
                # Q関数をニューラルネットで更新
                self.agent.update_q_function()
                # 状態を次の状態に更新
                state = state_next
                # エピソード終了時
                if done:
                    print('{0}エピソード: {1}ステップで終了 - reward: {2}'.format(episode, step, reward))
                    break

            # 最終エピソードの場合は動画を保存
            # if is_episode_final:
            #     save_as_gif(frames)
            #     break
            # 10回連続で成功したら、次のエピソードで終わりにする
            if complete_episodes >= 10:
                after = time.time()
                exe_time = after - before
                print('10回連続成功')
                print('Execution Time: {0} sec'.format(round(exe_time, 5)))
                is_episode_final = True

cartpole_env = Environment()
cartpole_env.run()