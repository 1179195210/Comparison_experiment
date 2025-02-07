import pandas as pd
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# DRQN 最优代码

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_cost = []

class ImprovedDQN(nn.Module):
    def __init__(self, n_actions, input_shape):
        super(ImprovedDQN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm1d(64)

        conv_output_size = self._get_conv_output(input_shape)

        self.lstm1 = nn.LSTM(input_size=conv_output_size, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()[1:]))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), 1, -1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        x = self.fc(x)
        return x


class DQNAgent:
    def __init__(self, n_actions, input_shape, learning_rate=0.0005, gamma=0.99, epsilon=1.0,
                 replace_target_iter=400, memory_size=800, batch_size=800, epsilon_decrement=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = epsilon_decrement
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0

        self.memory = deque(maxlen=self.memory_size)
        self.eval_net = ImprovedDQN(n_actions, input_shape).to(device)
        self.target_net = ImprovedDQN(n_actions, input_shape).to(device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.loss_func = nn.SmoothL1Loss()

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            actions_value = self.eval_net(state)
            action = torch.argmax(actions_value).item()
        return action

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        if len(self.memory) < self.batch_size:
            return 0

        batch_memory = self.memory
        state_batch = torch.tensor([x[0] for x in batch_memory], dtype=torch.float32).to(device)
        action_batch = torch.tensor([x[1] for x in batch_memory], dtype=torch.int64).view(-1, 1).to(device)
        reward_batch = torch.tensor([x[2] for x in batch_memory], dtype=torch.float32).view(-1, 1).to(device)
        next_state_batch = torch.tensor([x[3] for x in batch_memory], dtype=torch.float32).to(device)

        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)
        loss_cost.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decrement

        return loss.item()

# 初始化 DQNAgent
n_actions = 1000
input_shape = (1, 1000)

agent = DQNAgent(n_actions, input_shape)

dataframe = pd.read_csv('experience669210.csv')
dataframe['0'] = dataframe['Unnamed: 0'] * 10
dataframe['1002'] = dataframe['0']
dataframe = dataframe.drop(dataframe.columns[0], axis=1)
print(dataframe)

# 固定的四步状态、动作和奖励
states = np.array(dataframe.iloc[:800, :1000])  # 初始化状态
state_to_index = {tuple(row): idx for idx, row in enumerate(states)}
next_states = np.array(dataframe.iloc[:800, 1002:])
actions = dataframe.iloc[:800, 1000].tolist()  # 固定的动作
rewards = dataframe.iloc[:800, 1001].tolist()  # 固定的奖励

# 存储转换为经验池
for i in range(800):
    state = states[i]
    next_state = next_states[i]
    agent.store_transition(state, actions[i], rewards[i], next_state)
reward_list = []

# 训练
start_time = time.time()
for epoch in range(1000):
    loss = agent.learn()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")
        state = states[0].copy()
        for step in range(4):
            action = agent.choose_action(state)
            state[action] = action * 10
            state[0] = (step+1) * 10
            print(f"Step: {step}, Selected action: {action}")
            if step == 3:
                end_time = time.time()
                time_diff = end_time - start_time
                print(f"Step: {step}, Selected action: {action}", "时间为", time_diff)
                element_to_find = action
                indices = [index for index, element in enumerate(actions) if int(element) == element_to_find]
                if not indices:
                    reward = 0
                    reward_list.append(reward)
                    break
                selected_elements = [rewards[i] for i in indices]
                reward = sum(selected_elements) / len(selected_elements)
                reward_list.append(reward)

print(reward_list)
# 保存模型
torch.save({
    'eval_net_state_dict': agent.eval_net.state_dict(),
    'target_net_state_dict': agent.target_net.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'epsilon': agent.epsilon,
}, 'drqn_model390.pth')

print("模型已成功保存为 drqn_model.pth")