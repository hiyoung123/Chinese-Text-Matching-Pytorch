#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIM(nn.Module):

    def __init__(self, config):
        super(ESIM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=False)

        self.lstm1 = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm2 = nn.LSTM(config.hidden_size*8, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)

        self.BN = nn.BatchNorm1d(config.embed_dim)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size * 8),
            nn.Linear(config.hidden_size * 8, config.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.linear_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_size, config.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.linear_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_size, 2),
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

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat([p1, p2], 1)

    def forward(self, input):
        sent1, sent2 = input['seq1'], input['seq2']
        mask1, mask2 = sent1.eq(0), sent2.eq(0)
        x1 = self.BN(self.embedding(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.BN(self.embedding(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)

        return similarity, similarity