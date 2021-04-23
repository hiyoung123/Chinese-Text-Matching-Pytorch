#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_max(x):
    return torch.mean(x, dim=1), torch.max(x, dim=1)[0]


def submul(x1, x2):
    mul = x1 * x2
    sub = x1 - x2
    return torch.cat([sub, mul], -1)


class EnhancedRCNN(nn.Module):
    def __init__(self, config):
        super(EnhancedRCNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=False)
        self.gru = nn.GRU(config.embed_dim, config.hidden_size, batch_first=True, bidirectional=True)
        self.BN = nn.BatchNorm1d(config.embed_dim)
        self.nin = NIN(config.embed_dim, self.embeds_dim)
        self.fc_sub = FCSubtract(config.embed_dim * 16 + (config.max_seq_len - 1) * 6, config.linear_size)
        self.fc_mul = FCMultiply(config.embed_dim * 16 + (config.max_seq_len - 1) * 6, config.linear_size)
        self.dense = nn.Sequential(
            nn.Linear(config.linear_size*2, 2),
            nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):

        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)

        return x1_align, x2_align

    def forward(self, input):
        sent1, sent2 = input['seq1'], input['seq2']
        mask1, mask2 = sent1.eq(0), sent2.eq(0)
        a = self.BN(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        b = self.BN(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # RNN encoder: BiGRU
        o1, _ = self.gru(a)
        o2, _ = self.gru(b)

        # CNN encoder: NIN
        n1 = self.nin(a.transpose(1, 2))
        n2 = self.nin(b.transpose(1, 2))

        # Soft Attention Align
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Interaction Modeling
        q1_combined = torch.cat([o1, q1_align, submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, submul(o2, q2_align)], -1)
        v1_avg, v1_max = mean_max(q1_combined)
        v2_avg, v2_max = mean_max(q2_combined)

        o_a = torch.cat([v1_avg, n1, v1_max], -1)
        o_b = torch.cat([v2_avg, n2, v2_max], -1)

        # Similarity Modeling
        res_sub = self.fc_sub(o_a, o_b)
        res_mul = self.fc_mul(o_a, o_b)

        res = torch.cat((res_sub, res_mul), dim=1)
        similarity = self.dense(res)

        return similarity


class NIN(nn.Module):
    def __init__(self, input_dim, conv_dim):
        super(NIN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(input_dim, conv_dim, kernel_size=2),
            nn.ReLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(input_dim, conv_dim, kernel_size=3),
            nn.ReLU(),
        )

    def forward(self, x):
        avg1, max1 = mean_max(self.cnn1(x))
        avg2, max2 = mean_max(self.cnn2(x))
        avg3, max3 = mean_max(self.cnn3(x))
        return torch.cat((avg1, max1, avg2, max2, avg3, max3), dim=1)


class FCSubtract(nn.Module):
    def __init__(self, D_in, D_out):
        super(FCSubtract, self).__init__()
        self.dense = nn.Linear(D_in, D_out)

    def forward(self, input_1, input_2):
        res_sub = torch.sub(input_1, input_2)
        res_sub_mul = torch.mul(res_sub, res_sub)
        out = self.dense(res_sub_mul)
        return F.relu(out)


class FCMultiply(nn.Module):
    def __init__(self, D_in, D_out):
        super(FCMultiply, self).__init__()
        self.dense = nn.Linear(D_in, D_out)

    def forward(self, input_1, input_2):
        res_mul = torch.mul(input_1, input_2)
        out = self.dense(res_mul)
        return F.relu(out)