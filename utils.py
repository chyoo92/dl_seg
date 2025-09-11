import torch
import torch.nn as nn
import torch.nn.functional as F

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

#### for evaluation code
    
def pad_to_size(self, tensor, size):
    """텐서를 지정된 크기로 패딩합니다."""
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (size - h) if h < size else 0
    pad_w = (size - w) if w < size else 0
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

def unpad(self, tensor, original_shape):
    """패딩된 텐서를 원래 크기로 되돌립니다."""
    h_orig, w_orig = original_shape[-2], original_shape[-1]
    h_pad, w_pad = tensor.shape[-2], tensor.shape[-1]
    
    pad_h = h_pad - h_orig
    pad_w = w_pad - w_orig

    pad_top = pad_h // 2
    pad_bottom = h_pad - (pad_h - pad_top)
    pad_left = pad_w // 2
    pad_right = w_pad - (pad_w - pad_left)

    return tensor[..., pad_top:pad_bottom, pad_left:pad_right]


class GetAffs(nn.Module):
    """
    세그멘테이션 맵으로부터 어피니티(affinity) 맵을 계산하는 클래스.
    """
    def __init__(self, offsets, dtype, device):
        super(GetAffs, self).__init__()
        self.offsets = offsets
        self.dtype = dtype
        self.device = device

    @property
    def dim(self):
        return len(self.offsets[0])

    def aff_shift_kernels_(self, kernel, dim, offset):
        if dim == 2:
            assert len(offset) == 2
            kernel[0, 0, 1, 1] = -1.
            s_x = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            kernel[0, 0, s_x, s_y] = 1.
        else:
            raise NotImplementedError
        return kernel

    def segmentation_to_affinity(self, segmentation, offset):
        assert segmentation.size(1) == 1, str(segmentation.size(1))
        assert self.dim == 2

        shift_kernels = self.aff_shift_kernels_(
            torch.zeros(1, 1, 3, 3, device=self.device, dtype=self.dtype),
            self.dim,
            offset
        )
        
        abs_offset = tuple(max(1, abs(off)) for off in offset)
        
        spatial_gradient = F.conv2d(input=segmentation,
                                    weight=shift_kernels,
                                    dilation=abs_offset,
                                    padding=abs_offset)

        binarized_affinities = torch.where(
            spatial_gradient == 0,
            torch.ones_like(spatial_gradient),
            torch.zeros_like(spatial_gradient)
        )
        return binarized_affinities

    def get_affs(self, segmentation):
        affinities = []
        for offset in self.offsets:
            affinity = self.segmentation_to_affinity(segmentation, offset)
            affinities.append(affinity)
        
        # --- 수정된 부분 ---
        # (N, B, C, H, W) 형태의 리스트를 (N, H, W) 형태의 텐서로 변환
        # B=1, C=1 이므로 squeeze()로 차원을 제거합니다.
        # return(torch.stack(affinities, dim=0)) # 기존 코드
        return torch.stack(affinities, dim=0).squeeze(1).squeeze(1) # 수정된 코드    