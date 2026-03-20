import random
from collections import deque
from math import gamma

import torch
import wandb
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, bert_output_dim=768, gru_hidden_dim=256, action_dim=11):
        super(QNetwork, self).__init__()

        # BiGRU层提取全局语义
        self.bigru = nn.GRU(
            input_size=bert_output_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention层用于捕捉局部特征
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 残差连接路径：[CLS] 经过 projection 融合
        self.residual_proj = nn.Linear(bert_output_dim, gru_hidden_dim * 2)

        # 输出层：全连接，映射到动作空间的Q值
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        """
        输入:
            x: Tensor, [batch_size, seq_len, 768]  — BERT的token向量序列
        输出:
            q_values: Tensor, [batch_size, action_dim]
        """
        bigru_out, _ = self.bigru(x)  # [B, T, 512]

        # Attention: 权重加权表示
        attn_weights = torch.softmax(self.attention(bigru_out), dim=1)  # [B, T, 1]
        attn_output = torch.sum(attn_weights * bigru_out, dim=1)  # [B, 512]

        # 残差融合：使用BERT的[CLS]或第一个token
        residual = self.residual_proj(x[:, 0, :])  # [B, 512]

        combined = attn_output + residual  # [B, 512]

        # 输出Q值
        q_values = self.fc(combined)  # [B, action_dim]
        return q_values


# class QNetwork(nn.Module):
#     def __init__(self, state_dim=768, action_dim=11, hidden_dim=1024):
#         super().__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)  # 输出每个动作的Q值


def compute_q_loss(q_network, target_network, batch, i_episode, gamma=0.99):
    # 确保 batch 是一个可解包的元组，而不是列表
    if isinstance(batch, list):
        batch = zip(*batch)  # 将列表转换为元组

    states, actions, rewards, next_states = batch

    # 确保数据是 PyTorch 张量
    states = torch.cat(states,dim=0)
    actions = torch.tensor(actions, dtype=torch.long)
    next_states = torch.cat(next_states,dim=0)

    # 计算当前 Q 值
    q_values = q_network(states).squeeze(1)
    current_q = q_values.gather(1, actions.unsqueeze(1))

    # 计算目标 Q 值
    with torch.no_grad():
        next_q = target_network(next_states).squeeze(1).max(dim=1)[0]
        # 假设 rewards 是一个元组，先提取其中的张量
        rewards = rewards[0]  # 提取第一个张量
        # rewards = rewards.squeeze(1)

        target_q = rewards + gamma * next_q
    #wandb.log({"current_q": current_q.mean().item(), "target_q": target_q.mean().item(),"custom_epoch": i_episode})
    # 计算损失
    loss = F.mse_loss(current_q.squeeze(), target_q)
    return loss


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
