
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import os

from utils import GetAffs


class calciumdataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.isInit = False

        self.data_length = 0
        self.names = []
        self.videos = []
        self.labels = []
        self.segs = []
        self.summs = []
        
        
        


    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        if not self.isInit: self.initialize()

        video = self.videos[idx]
        summ = torch.tensor(self.summs[idx].astype(np.float32), dtype=self.dtype, device=self.device)
        
        seg = torch.tensor(self.segs[idx].astype(np.float32), dtype=self.dtype, device=self.device)
        label = torch.tensor(self.labels[idx].astype(np.float32), dtype=self.dtype, device=self.device)
        transform_params = self.get_transform_params(seg)

        new_summ = self.transform_summlike(summ, transform_params, norm=True, comp_affs=False)
        corrs = self.transform_video(video, transform_params) 
        
        data = self.transform_summlike(seg, transform_params, norm=False, comp_affs=True)
        label = self.transform_summlike(label, transform_params, norm=False, comp_affs=False)

        print(data.shape)
        print(label.shape)
        return data, label, corrs, new_summ





    def initialize(self):
        # 3. 실제 데이터 로딩 직전에 딱 한 번만 실행될 작업들
        assert not self.isInit, "initialize() should only be called once."
        
        print("="*40)
        print("Dataset Initializing...")
        # print(f"Found {len(self.samples)} samples.")
        print("="*40)
        
        # 여기서 모든 샘플 파일이 실제로 존재하는지 확인하는 등의
        # 무거운 검증 작업을 수행할 수도 있습니다.
        
        self.isInit = True
    
    

    def addSample(self, fName,device,predict):

        self.sec_length=100
        self.sec_size=128
        self.n_secs=10
        
        self.predict = predict
        if os.path.isdir(fName):
            fName = os.path.join(fName, "*.h5")
        
        for fName in glob.glob(fName):
            if not fName.endswith(".h5"): continue
            print('--------------------------------')
            print('loading ' + fName)
            print('--------------------------------')
            name = fName.split('/')[-1][:-3]

            path = os.path.dirname(fName)

            with h5py.File(fName) as f:
                if 'input' not in f: continue
                event = f['input'].astype(np.int16)

            with h5py.File(path+'/'+'BF_labels.h5','r') as f: label = f[name][...]
            with h5py.File(path+'/'+'gt_segmentations.h5','r') as f: seg = f[name][...]
            with h5py.File(path+'/'+'summary_images.h5','r') as f: summ = f[name][...]
            

            summ = torch.tensor(summ[np.newaxis,...].astype(np.float32), dtype=torch.float, device=device)
            seg = torch.tensor(seg[np.newaxis,...].astype(np.float32), dtype=torch.float, device=device)
            label = torch.tensor(label[np.newaxis,...].astype(np.float32), dtype=torch.float, device=device)

            transform_params = self.get_transform_params(seg)

            new_summ = self.transform_summlike(summ, transform_params, norm=True, comp_affs=False)
            # print(event.shape,'sfasdfasdfasfd')
            print(event,'asdfasfd')
            
            corrs = self.transform_video(event, transform_params) 
            
            data = self.transform_summlike(seg, transform_params, norm=False, comp_affs=True)
            label = self.transform_summlike(label, transform_params, norm=False, comp_affs=False)

            print(data.shape)
            print(label.shape)
            print(new_summ.shape)
            print(seg.shape)
           
            self.names.append(name)
            self.videos.append(event)
            self.labels.append(label[np.newaxis, ...])
            self.segs.append(seg[np.newaxis, ...])
            self.summs.append(summ[np.newaxis, ...])

        self.data_length = len(self.names) ### file number



    def get_transform_params(self, seg_o):
        """데이터 증강 및 크롭을 위한 파라미터를 생성합니다."""
        hflip, vflip, rots = False, False, 0
        maxpool_size = 6
        height, width = seg_o.size(1), seg_o.size(2)

        if not self.predict:
            # 학습 모드: 랜덤 증강 및 128x128 크롭 보장
            if np.random.random() > 0.5: vflip = True
            if np.random.random() > 0.5: hflip = True
            rots = np.random.choice(4, 1)[0]
            maxpool_size = np.random.choice(np.arange(3, 10), 1)[0]

            if height < self.sec_size or width < self.sec_size:
                x_start, y_start = 0, 0
                x_stop, y_stop = height, width
            else:
                x_start = np.random.randint(0, height - self.sec_size + 1)
                y_start = np.random.randint(0, width - self.sec_size + 1)
                x_stop = x_start + self.sec_size
                y_stop = y_start + self.sec_size
        else:
            # 예측 모드: 전체 이미지 사용
            x_start, y_start = 0, 0
            x_stop, y_stop = height, width

        return [hflip, vflip, rots, maxpool_size, x_start, x_stop, y_start, y_stop]
        
    def transform_summlike(self, summ, transform_params, norm=True, comp_affs=False):
        """요약 이미지, 레이블, 세그멘테이션에 동일한 변환을 적용합니다."""
        hflip, vflip, rots, _, x_start, x_stop, y_start, y_stop = transform_params
        
        summ = summ[:, x_start:x_stop, y_start:y_stop]
        
        if vflip: summ = summ.flip(1)
        if hflip: summ = summ.flip(2)
        if rots == 1: summ = summ.transpose(1, 2).flip(1)
        if rots == 2: summ = summ.flip(1).flip(2)
        if rots == 3: summ = summ.transpose(1, 2).flip(2)
            
        if norm:
            s_stds = torch.std(summ)
            if s_stds > 0: summ = (summ - torch.mean(summ)) / s_stds

        if comp_affs:
            if summ.dim() == 3:
                summ = summ.unsqueeze(0) # 배치 차원 추가
            # 최종 입력이 4D인지 확인
            if summ.dim() != 4:
                raise ValueError(f"Shape error before get_affs: expected 4D, got {summ.shape}")
            summ = self.GA.get_affs(summ).squeeze(0)

        return summ

    def transform_video(self,video,transform_params):
        hflip,vflip,rots,maxpool_size,x_start,x_stop,y_start,y_stop = transform_params
        sec_length = int(video.shape[0] / self.n_secs) if self.n_secs > 0 else video.shape[0]
        maxpool = nn.MaxPool3d(kernel_size=(maxpool_size,1,1))
        
        if not self.predict:
            sec_starts = np.random.choice(video.shape[0]-sec_length,self.n_secs) if video.shape[0]>sec_length else [0]*self.n_secs
        else:
            sec_starts = np.arange(int(video.shape[0]/sec_length)) * sec_length if sec_length > 0 else [0]
        
        corrs = []
        video_crop = video[:, x_start:x_stop, y_start:y_stop]

        for start_frame in sec_starts:
            end_frame = min(start_frame + sec_length, video.shape[0])
            video_sec = video_crop[start_frame:end_frame, :, :]

            if video_sec.size == 0: continue
            video_resized = torch.tensor(video_sec.astype(np.float32),dtype=self.dtype,device=self.device)
            
            if video_resized.numel() == 0: continue
            if video_resized.size(0) < maxpool_size: continue

            video_resized = maxpool(video_resized.view(1,1,video_resized.size(0),video_resized.size(1),video_resized.size(2)))[0,0]
            
            if vflip: video_resized = video_resized.flip(1)
            if hflip: video_resized = video_resized.flip(2)
            if rots == 1: video_resized = video_resized.transpose(1,2).flip(1)
            if rots == 2: video_resized = video_resized.flip(1).flip(2)
            if rots == 3: video_resized = video_resized.transpose(1,2).flip(2)
                
            corrs.append(self.get_corrs(video_resized))
        
        if not corrs:
             h = x_stop - x_start
             w = y_stop - y_start
             return torch.zeros((self.n_secs, 15, h, w), dtype=self.dtype, device=self.device)

        corrs = torch.stack(corrs)
        stds = torch.std(corrs)
        if stds > 0: corrs = (corrs - torch.mean(corrs)) / stds
        
        return corrs