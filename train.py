
import argparse
from tqdm import tqdm
import csv, yaml, sys
import torch
import os
sys.path.append("./module")

from model.allModel import *
from datasets import dataset_main
from utils import SorensenDiceLoss

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', action='store', type=str) #### config file name
parser.add_argument('-o', '--output', action='store', type=str) #### setting output folder name ==ex) vertex_test==
parser.add_argument('--device', action='store', type=int, default=None) #### gpu device number 0,1,2,3.... only one gpu training
parser.add_argument('--transfer_learning', action='store', type=int, default=0) #### 0 first training / bigger than 1 transfer learning

parser.add_argument('--model', action='store',type=str)
parser.add_argument('--predict', action='store',type=int, default=None)

#### training parameter
parser.add_argument('--nDataLoaders', action='store', type=int, default=4)
parser.add_argument('--epoch', action='store', type=int, default=3000)
parser.add_argument('--batch', action='store', type=int, default=20)
parser.add_argument('--learningRate', action='store', type=float, default=1e-4)
parser.add_argument('--randomseed', action='store', type=int, default=12345)


args = parser.parse_args()

dataset_module = dataset_main  #### main dataset code
Dataset = dataset_module.calciumdataset

if args.device is not None:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### config file load
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
if args.nDataLoaders: config['training']['nDataLoaders'] = args.nDataLoaders
if args.epoch: config['training']['epoch'] = args.epoch
if args.batch: config['training']['batch'] = args.batch
if args.learningRate: config['training']['learningRate'] = args.learningRate
if args.randomseed: config['training']['randomSeed'] = args.randomseed


#### result folder
result_path = 'result_seg/' + args.output
if not os.path.exists(result_path): os.makedirs(result_path)

with open(result_path + '/' + args.output+'.txt', "w") as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

#### dataset 
dset = Dataset()





for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(sampleInfo['path'],device = device, predict=args.predict)
dset.initialize() 

print(dset,'dset')
print(len(dset))
#### split events
lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
print(len(trnDset),len(valDset),len(testDset),'sadfsdfasf')
print(trnDset,valDset,testDset,'sadfasfsdf')


kwargs = {'batch_size':config['training']['batch'],'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':True}

trnLoader = torch.utils.data.DataLoader(trnDset, shuffle=True, **kwargs)
valLoader = torch.utils.data.DataLoader(valDset, shuffle=False, **kwargs)
testLoader = torch.utils.data.DataLoader(testDset, shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

model_name = args.model

#### model load
if args.transfer_learning == 0:
    # model = model_name(fea = args.fea, \
    #                 cla = args.cla, \
    #                 cross_head = args.cross_head, \
    #                 cross_dim = args.cross_dim, \
    #                 self_head = args.self_head, \
    #                 self_dim = args.self_dim, \
    #                 n_layers = args.n_layers, \
    #                 num_latents = args.num_latents, \
    #                 dropout_ratio = args.dropout_ratio, \
    #                 batch = config['training']['batch'], \
    #                 device = device)
    model = model_name(batch = config['training']['batch'], \
                    device = device)
else:
    model = torch.load(result_path+'/model.pth',map_location='cuda')
    # model.load_state_dict(torch.load(result_path+'/model_state_dict_rt.pth',map_location='cuda'),strict=False)
    optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])
    checkpoint_path = result_path + '/checkpoint_' + str(args.transfer_learning) + '.pth'  #### You can train by inserting a checkpoint file number at the desired point.
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    optm.load_state_dict(checkpoint['optimizer_state_dict'])



  
torch.save(model, os.path.join(result_path, 'model.pth'))

crit = SorensenDiceLoss(channelwise=True).to(device)

    
if args.transfer_learning == 0:    
    optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])


bestState, bestLoss = {}, 1e9

train = {'loss':[], 'val_loss':[]}
    
nEpoch = config['training']['epoch']





for epoch in range(nEpoch):

    trn_loss = 0.
    nProcessed = 0

    ###############################################
    ################## training ###################
    ###############################################

    for i, batch_set in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        model = model.cuda()
        data, label, corrs, new_summ = batch_set

        foreground = label.to(device)
        affs = (1. - data).to(device)
        target = torch.cat([affs, foreground], dim=1)
        
        corrs = corrs.to(device)
        summ = new_summ.to(device)
        
        if summ.dim() == 3:
            summ = summ.unsqueeze(1)

        pred = model(corrs,summ)
        
        loss = crit(pred, target)

        loss.backward()
        optm.step()
        optm.zero_grad()

        
        ibatch = len(foreground)
        nProcessed += ibatch
        trn_loss += loss.item()*ibatch
        
        del foreground, affs, target, corrs, summ

    trn_loss /= nProcessed 
    print(trn_loss,'trn_loss')

    
    torch.save(model.state_dict(), os.path.join(result_path, 'model_state_dict_rt.pth'))


    ###############################################
    ################## validation #################
    ###############################################

    model.eval()
    val_loss = 0.
    nProcessed = 0
    with torch.no_grad():
        for i, batch_set in enumerate(tqdm(valLoader)):

            data, label, corrs, new_summ = batch_set

            foreground = label.to(device)
            affs = (1. - data).to(device)
            target = torch.cat([affs, foreground], dim=1)
            
            corrs = corrs.to(device)
            summ = new_summ.to(device)
            
            
            if summ.dim() == 3:
                summ = summ.unsqueeze(1)




            pred = model(corrs,summ)
            
            loss = crit(pred, target)
            
            ibatch = len(foreground)
            nProcessed += ibatch
            val_loss += loss.item()*ibatch
            
            del foreground, affs, target, corrs, summ
                
                
        val_loss /= nProcessed
        print(val_loss,'val_loss')
        
        if bestLoss > val_loss:
            bestState = model.to('cpu').state_dict()
            bestLoss = val_loss
            torch.save(bestState, os.path.join(result_path, 'weight.pth'))


        
        train['loss'] = [trn_loss]
        train['val_loss'] = [val_loss]

        
        file_path = os.path.join(result_path, 'train.csv')
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # 파일이 새로 생성된 경우에만 헤더를 작성
            if not file_exists or os.path.getsize(file_path) == 0:
                keys = train.keys()
                writer.writerow(keys)

            # 데이터 추가
            keys = train.keys()
            for row in zip(*[train[key] for key in keys]):
                writer.writerow(row)

        
    checkpoint = {
        'epoch': int(args.transfer_learning)+epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optm.state_dict(),
        'loss': loss.item(),
    }
    
    torch.save(checkpoint, f'{result_path}/checkpoint_{int(args.transfer_learning)+epoch+1}.pth')
    torch.save(model.to('cpu').state_dict(), f'{result_path}/weight_{int(args.transfer_learning)+epoch+1}.pth')
    
bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join(result_path, 'weightFinal.pth'))
