## disco original code - https://github.com/EKirschbaum/DISCo/tree/master
## GASP & inferno version issue, original code is not working.
## inferno Unet model code change to pytorch
## GASP change simple code

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py as h5
from collections import deque

#### Unet model code change to pytorch
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels) 
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    표준 U-Net 아키텍처 구현.
    원본 코드의 파라미터(depth=5, initial_features=64)를 기반으로 재구성했습니다.
    """
    def __init__(self, in_channels, out_channels, initial_features=64, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        f = initial_features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(f*8, f*16 // factor)
        
        self.up1 = Up(f*16, f*8 // factor, bilinear)
        self.up2 = Up(f*8, f*4 // factor, bilinear)
        self.up3 = Up(f*4, f*2 // factor, bilinear)
        self.up4 = Up(f*2, f, bilinear)
        self.outc = OutConv(f, out_channels)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return self.final_activation(logits)


class DISCoNet(nn.Module):
    """
    DISCoNet 모델의 전체 아키텍처.
    """
    def __init__(self, device):
        super(DISCoNet, self).__init__()
        self.device = device
        
        corr_channels = 15
        combi_channels = corr_channels + 1
        out_channels = 6
        
        self.conv1 = nn.Conv3d(corr_channels, 2 * corr_channels, (4, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(2 * corr_channels, 4 * corr_channels, (4, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(4 * corr_channels, corr_channels, (4, 3, 3), padding=(0, 1, 1))
        self.relu = nn.ReLU()
        
        self.unet = UNet(in_channels=combi_channels, out_channels=out_channels, 
                         initial_features=64).to(self.device)
                             
    def forward(self, corrs, summ):
        corrs = torch.transpose(corrs, 1, 2)
        
        corrs = self.relu(self.conv1(corrs))
        corrs = self.relu(self.conv2(corrs))
        corrs = self.relu(self.conv3(corrs))[:, :, 0]
            
        combi_input = torch.cat([summ, corrs], dim=1)
            
        out = self.unet(combi_input)

        return out

####### GASP code

# 참고: 이 파일은 원래 GASP라는 라이브러리를 사용하여 어피니티 맵으로부터
# 최종 세그멘테이션을 생성했습니다. 사용자 요청에 따라 해당 라이브러리 의존성을 제거하고,
# 어피니티 맵과 임계값(threshold)을 이용한 간단한 연결 요소(Connected Components)
# 알고리즘으로 대체했습니다. 이 방법은 원본보다 정교함이 떨어질 수 있습니다.

def segment_with_connected_components(affs, fg_mask, threshold=0.9):
    """
    어피니티 맵과 전경 마스크를 사용하여 연결 요소 레이블링을 수행합니다.
    BFS(너비 우선 탐색) 알고리즘을 기반으로 합니다.

    Args:
        affs (np.ndarray): (C, H, W) 모양의 어피니티 맵. 
                           C=2로 가정 (상, 좌 방향).
        fg_mask (np.ndarray): (H, W) 모양의 전경 마스크. 0 또는 1 값을 가짐.
        threshold (float): 두 픽셀을 연결된 것으로 간주할 어피니티 임계값.

    Returns:
        np.ndarray: (H, W) 모양의 최종 세그멘테이션(레이블) 맵.
    """
    H, W = fg_mask.shape
    segmentation = np.zeros_like(fg_mask, dtype=np.int64)
    label_counter = 1

    coords = np.argwhere(fg_mask > 0)

    for r, c in coords:
        if segmentation[r, c] == 0:
            segmentation[r, c] = label_counter
            q = deque([(r, c)])

            while q:
                curr_r, curr_c = q.popleft()
                if curr_r > 0 and segmentation[curr_r - 1, curr_c] == 0 and \
                   fg_mask[curr_r - 1, curr_c] > 0 and affs[0, curr_r, curr_c] > threshold:
                    segmentation[curr_r - 1, curr_c] = label_counter
                    q.append((curr_r - 1, curr_c))
                
                if curr_r < H - 1 and segmentation[curr_r + 1, curr_c] == 0 and \
                   fg_mask[curr_r + 1, curr_c] > 0 and affs[0, curr_r + 1, curr_c] > threshold:
                    segmentation[curr_r + 1, curr_c] = label_counter
                    q.append((curr_r + 1, curr_c))

                if curr_c > 0 and segmentation[curr_r, curr_c - 1] == 0 and \
                   fg_mask[curr_r, curr_c - 1] > 0 and affs[1, curr_r, curr_c] > threshold:
                    segmentation[curr_r, curr_c - 1] = label_counter
                    q.append((curr_r, curr_c - 1))
                
                if curr_c < W - 1 and segmentation[curr_r, curr_c + 1] == 0 and \
                   fg_mask[curr_r, curr_c + 1] > 0 and affs[1, curr_r, curr_c + 1] > threshold:
                    segmentation[curr_r, curr_c + 1] = label_counter
                    q.append((curr_r, curr_c + 1))
            
            label_counter += 1
            
    return segmentation

def get_segmentation(prediction_file, mode, usedata):
    """
    모델 예측 결과(어피니티)를 HDF5 파일에서 읽어 최종 세그멘테이션을 생성합니다.
    
    Args:
        prediction_file (str): 예측 결과가 저장된 HDF5 파일 경로 (확장자 제외).
        mode (str): 'disco' 또는 'discos'.
        usedata (str): 'discos' 모드에서 사용할 데이터 그룹.

    Returns:
        str: 최종 세그멘테이션 결과가 저장된 파일 경로.
    """
    out_file = prediction_file + '_segmentation'
    
    if mode == 'disco':        
        with h5.File(prediction_file + '.h5', "r") as f:
            for name in f.keys():
                pred = f[name][...]
                fg = pred[-2]
                bg = pred[-1]
                affs = pred[[0, 1]]
                
                fg_mask = (fg > bg).astype(np.uint8)
                
                final_segmentation = segment_with_connected_components(affs, fg_mask, threshold=0.9)
                
                final = np.stack([final_segmentation, fg], axis=0)
                with h5.File(out_file + ".h5", "a") as g:
                    g.create_dataset(name, data=final, compression="gzip")
                        
    elif mode == 'discos':        
        with h5.File(prediction_file + '.h5', "r") as f:
            f_u = f[usedata]
            for name in f_u.keys():
                pred = f_u[name][...]
                fg = pred[-2]
                bg = pred[-1]
                affs = pred[[0, 1]]

                fg_mask = (fg > bg).astype(np.uint8)
                
                final_segmentation = segment_with_connected_components(affs, fg_mask, threshold=0.9)
                
                final = np.stack([final_segmentation, fg], axis=0)
                with h5.File(out_file + ".h5", "a") as g:
                    if usedata not in g:
                        grp = g.create_group(usedata)
                    else:
                        grp = g[usedata]
                    grp.create_dataset(name, data=final, compression="gzip")
                        
    return out_file