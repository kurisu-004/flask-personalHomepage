# DT网络结构

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DecisionTransformerModel, DecisionTransformerConfig
import numpy as np

class DTDataCollator:
    max_len =  512 # 用于训练的序列长度，所有的对局都不会超过这个长度

    def __init__(self, dataset):
        self.act_dim = len(dataset[0]["action"][0])
        self.state_dim = len(dataset[0]["state"][0])
        self.dataset = dataset
        # print("act_dim:", self.act_dim)
        # print("state_dim:", self.state_dim)

    def __call__(self, features):

        batch_size = len(features)

        s, a, r, rtg, timesetps, mask = [], [], [], [], [], []
        for feature in features:
            s.append(np.array(feature["state"]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["action"]).reshape(1, -1, self.act_dim))
            rewards = np.zeros((1, len(feature["state"]), 1))
            rewards[:, -1, 0] = feature["rtg"][0]
            r.append(rewards)
            rtg.append(np.array(feature["rtg"]).reshape(1, -1, 1))
            timesetps.append(np.arange(len(feature["state"])).reshape(1, -1))


            # padding and mask
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1)
            timesetps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesetps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesetps = torch.from_numpy(np.concatenate(timesetps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        # 打印形状和数据类型
        # print("states shape:", s.shape, "dtype:", s.dtype)
        # print("actions shape:", a.shape, "dtype:", a.dtype)
        # print("rewards shape:", r.shape, "dtype:", r.dtype)
        # print("returns_to_go shape:", rtg.shape, "dtype:", rtg.dtype)
        # print("timesteps shape:", timesetps.shape, "dtype:", timesetps.dtype)
        # print("attention_mask shape:", mask.shape, "dtype:", mask.dtype)
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesetps,
            "attention_mask": mask,
        }

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)

        action_preds = output[1]
        action_targets = kwargs['actions']
        attention_mask = kwargs['attention_mask']
        act_dim = action_preds.shape[-1]

        # Reshape the predictions and targets
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.argmax(dim=-1).reshape(-1)[attention_mask.reshape(-1) > 0]

        # Calculate the loss
        loss = F.cross_entropy(action_preds, action_targets)
        

        return {"loss": loss}

    
    def original_forward(self, **kwargs):
        return super().forward(**kwargs)
    
    def get_action(self, **kwargs):
        output = super().forward(**kwargs)
        return output[1].argmax(dim=-1)

