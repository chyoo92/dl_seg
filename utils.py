import torch
import torch.nn as nn

## segmentation loss
class SorensenDiceLoss(nn.Module):
    def __init__(self, channelwise=True):
        super().__init__()
        self.channelwise = channelwise

    def forward(self, prediction, target):
        # 분자: 예측과 타겟의 곱의 합
        intersection = torch.sum(prediction * target, dim=tuple(range(2, prediction.dim())))
        # 분모: 예측과 타겟 각각의 합
        cardinality = torch.sum(prediction + target, dim=tuple(range(2, prediction.dim())))
        
        dice_score = (2. * intersection) / (cardinality + 1e-6)
        
        if self.channelwise:
            return (1. - dice_score).mean()
        else:
            return 1. - dice_score.mean()