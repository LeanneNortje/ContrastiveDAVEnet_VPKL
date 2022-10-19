#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoringAttentionModule(nn.Module):
    def __init__(self, args):
        super(ScoringAttentionModule, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.image_attention_encoder = nn.Sequential(
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.audio_attention_encoder = nn.Sequential(
            # nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        # self.image_embedding_encoder = nn.Sequential(
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU(),
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU()
        # )
        # self.audio_embedding_encoder = nn.Sequential(
        #     # nn.LayerNorm(512),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU(),
        #     # nn.LayerNorm(512),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU()
        # )
        # self.similarity_encoder = nn.Sequential(
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU(),
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, 1),
        #     nn.ReLU()
        # )
        self.pool_func = nn.AdaptiveAvgPool2d((1, 1))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, image_embedding, audio_embeddings, audio_nframes):

        aud_att = torch.bmm(image_embedding, audio_embeddings)
        aud_att, _ = aud_att.max(dim=1)
        score = aud_att.mean(dim=-1).unsqueeze(1)

        return score

    def encode(self, image_embedding, audio_embeddings, audio_nframes): 
        
        # im = self.pool_func(image_embedding.transpose(1, 2).unsqueeze(2)).squeeze(-1).squeeze(-1)#.transpose(1, 2)
        # aud = self.pool_func(audio_embeddings.unsqueeze(2)).squeeze(-1).squeeze(-1)#.transpose(1, 2)
        # print(im.size(), aud.size())
        # intermediate_score = torch.bmm(im.unsqueeze(1), aud.unsqueeze(1).transpose(1, 2)).squeeze(1)#self.cos(im, aud)#.unsqueeze(1)
        # print(intermediate_score.size())
        # im = self.image_embedding_encoder(im)
        # aud = self.audio_embedding_encoder(aud)

        aud_att = torch.bmm(image_embedding, audio_embeddings)
        aud_att, _ = aud_att.max(dim=1)
        score = aud_att.mean(dim=-1).unsqueeze(1)
        # # intermediate_score = 1 - torch.sigmoid(aud_att.mean(dim=-1))
        # aud_att = aud_att.unsqueeze(1)
        # aud_context = torch.bmm(aud_att, audio_embeddings.transpose(1, 2)).squeeze(1)
        
        # im_att = torch.bmm(audio_embeddings.transpose(1, 2), image_embedding.transpose(1, 2))
        # im_att, _ = im_att.max(dim=1)
        # im_att = im_att.unsqueeze(1)
        # im_context = torch.bmm(im_att, image_embedding).squeeze(1)

        # score = self.cos(aud_context, im_context).unsqueeze(1)

        return score, aud_att.unsqueeze(1)

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.margin = args["margin"]
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = nn.MSELoss()

    def forward(self, anchor, positives, negatives):
        # loss = 0
        # N = anchor.size(0)

        # for p in range(positives.size(1)):
        #     for n in range(negatives.size(1)):
        #         loss += (-(anchor.squeeze(1) - positives[:, p, :]) + (anchor.squeeze(1) + negatives[:, n, :]) + 1).clamp(min=0).mean() 
        #     samples = torch.cat([positives[:, p, :].unsqueeze(1), negatives], dim=1)
        #     # sim = torch.bmm(anchor, samples.transpose(1, 2)).squeeze(1) 
        #     sim = []
        #     for i in range(anchor.size(0)):
        #         sim.append(self.cos(anchor[i, :, :].repeat(samples.size(1), 1), samples[i, :, :]).unsqueeze(0))
        #     sim = torch.cat(sim, dim=0)
        #     labels = torch.zeros(sim.size(0), dtype=torch.long, device=sim.device)
        #     loss = F.cross_entropy(sim, labels)

        # N = anchor.size(0)
        # loss = 0
        # for n in range(negatives.size(1)): 
        #     loss += (negatives[:, n, :] - anchor.squeeze(1) + 1.0).clamp(min=0).mean()

        N = anchor.size(0)
        loss = 0
        for p in range(positives.size(1)):
            sim = [positives[:, p, :], anchor.squeeze(1)]
            labels = [torch.ones((anchor.size(0), 1), device=anchor.device), torch.ones((anchor.size(0), 1), device=anchor.device)]
            for n in range(negatives.size(1)): 
                sim.append(negatives[:, n, :])
                labels.append(-1 * torch.ones((anchor.size(0), 1), device=anchor.device))
                # loss += (self.cos(negatives[:, n, :], anchor.squeeze(1)) - self.cos(positives[:, p, :], anchor.squeeze(1)) + 2.0).clamp(min=0).mean()
                # print(self.cos(negatives[:, n, :], anchor.squeeze(1)), self.cos(positives[:, p, :], anchor.squeeze(1)))
            sim = torch.cat(sim, dim=1)
            labels = torch.cat(labels, dim=1)
            loss += self.criterion(sim, labels) 

        return loss

    # def encode(self, anchor, positives, negatives):

    #     samples = torch.cat([positives, negatives], dim=1)
    #     # sim = torch.bmm(anchor, samples.transpose(1, 2)).squeeze(1)
    #     sim = []
    #     for i in range(anchor.size(0)):
    #         sim.append(self.cos(anchor[i, :, :].repeat(samples.size(1), 1), samples[i, :, :]).unsqueeze(0))
    #     sim = torch.cat(sim, dim=0)
    #     labels = torch.zeros(sim.size(0), dtype=torch.long, device=sim.device)

    #     return sim, labels