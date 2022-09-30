import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import random

class MyDice(nn.Module):
    def __init__(self, types, whole_slice_oars = []):
        super(MyDice, self).__init__()
        self.smooth = 0.0001
        self.balance = True
        self.whole_slice_oars = whole_slice_oars
        self.threshold = 0.2
        self.process = 0.8
        self.amplitude = None
        self.record_dice = torch.zeros([types], dtype=torch.float32).cuda()
        self.record_val_dice = torch.zeros([types], dtype=torch.float32).cuda()


    def dice(self, pred, truth):
        channel = truth.shape[1]
        batchs = truth.shape[0]
        slices = truth.shape[2]

        if channel == 1:
            loss = 0
            dice = 0
            for i in range(batchs):
                a_f = pred[i, 0]
                b_f = truth[i, 0]
                if b_f.max() > 0:
                    dice_i = (2 * torch.sum(a_f * b_f) + self.smooth) / \
                             (torch.sum(a_f * a_f) + torch.sum(b_f * b_f) + self.smooth)
                else:
                    dice_i = (2 * torch.sum(a_f * b_f) + self.smooth) / \
                             (torch.sum(a_f * a_f) + torch.sum(b_f * b_f) + self.smooth)
                loss += (1.0 - dice_i)
                dice += dice_i

            loss = loss / batchs
            dice = dice / batchs
            c_dice = [dice.item()]

        else:
            dice = torch.zeros([channel], dtype=torch.float32).cuda()
            for i in range(0, channel):
                intersection = 0
                compilation = 0
                for j in range(batchs):
                    for k in range(1, slices-1):
                        T = truth[j, i, k]
                        P = pred[j, i + 1, k]
                        if T.max() > 0:
                            intersection += torch.sum(T * P)
                            compilation += torch.sum(T * T) + torch.sum(P * P)

                dice[i] = ( (2 * intersection + self.smooth) / (compilation + self.smooth))

            loss = 1.0 - dice.mean()
            c_dice = np.array([i.item() for i in dice])

        return loss, c_dice


    def cross_entropy3D(self, pred, truth):
        loss = F.binary_cross_entropy(pred, truth)
        return loss


    def diceloss(self, pred, truth, process):
        channel = truth.shape[1]
        batchs = truth.shape[0]

        if channel == 1:
            pred_s = pred
        else:
            pred_s = pred[:, 1::]

        dice = torch.zeros([channel], dtype=torch.float32).cuda()
        loss = torch.zeros([channel], dtype=torch.float32).cuda()
        for i in range(0, channel):
            weight = 0
            channel_dice = 0
            for j in range(batchs):
                if truth[j, i].max() < 0:
                    continue

                a_f = pred_s[j, i].contiguous()
                if truth[j, i].max() > 0:
                    b_f = truth[j, i].contiguous()
                    alpha = 1.0
                    exist_oar = True
                    dice_i = (2 * torch.sum(a_f * b_f) + self.smooth) / (torch.sum(a_f * a_f) + torch.sum(b_f * b_f) + self.smooth)
                else:
                    b_f = truth[j, i].contiguous()
                    alpha = 0.5
                    dice_i = (2 * torch.sum(a_f * b_f) + self.smooth) / (torch.sum(a_f * a_f) + torch.sum(b_f * b_f) + self.smooth)
                    exist_oar = False

                dice[i] += dice_i
                loss[i] += (1.0 - dice_i) * alpha
                if exist_oar and process >= self.process:
                   if random.uniform(0,1) > dice_i:
                       alpha = 0.02

                weight += alpha

            dice[i] = dice[i] / batchs
            loss[i] = loss[i] / weight

        loss = loss.sum()

        return loss, dice

    def forward(self, output, true_masks, process=0.0):
        [loss, pdice] = self.diceloss(output, true_masks, process)
        dice = np.array([i.item() for i in pdice])

        return loss, dice

