import fed
import argparse
from model.backbone import AVNet, result_level_fusion, late_fusion
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


import copy
import os

from aggregator.aggregator import fedAvg
from dataset import ravdess

import sys

# SEED = 42
SEED = 272

torch.manual_seed(SEED)
np.random.seed(SEED)

class Client(fed.Client):
    def __init__(self, model, optimizer, scheduler, dataloader,
                 device, local_epoch=5):
        super(Client, self).__init__()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        
        self.local_epoch = local_epoch
        self.cached_model = None
        self.device = device
        
    def train(self):
        epoch = self.local_epoch
        device = self.device
        
        criterion = nn.CrossEntropyLoss()
        critic = nn.CosineSimilarity(dim=1)
        
        self.model.train()
        
        train_res = []
        
        
        for _ in range(epoch):
            _loss = torch.tensor(.0).to(device)
            for _, (a, v, label) in enumerate(self.dataloader):
                
                a = a.to(device); v = v.to(device); label = label.to(device)
                
                pred,  embedding_a,  embedding_v  = self.model(a, v)
                
                loss = criterion(pred, label)
                
 
                # loss -= critic(embedding_a, embedding_v).mean().to(device)

                if self.cached_model is not None:
                    with torch.no_grad():
                        self.cached_model.eval()
                        _, _embedding_a, _embedding_v  = self.cached_model(a, v)
                
                    soft_loss = (-(
                        critic(embedding_a, _embedding_a) +
                        critic(embedding_v, _embedding_v) +
                        critic(embedding_a, _embedding_v) +
                        critic(embedding_v, _embedding_a)
                    )/4).mean().to(device)
                
                    loss += soft_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                
                _loss += loss
            train_res.append(_loss / len(self.dataloader.dataset))
                
        return train_res
            
    
    def load_model(self, model_param):
        self.cached_model = copy.deepcopy(self.model)
        self.model.load_state_dict(model_param)
    
    def get_model(self):
        return self.model.state_dict()
    
    
    def state(self):
        return {
            'model' : self.model.state_dict(),
            'optimizer' : self.model.state_dict(),
            'scheduler' : self.scheduler.state_dict()
        }
    
    def load_state(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        
class cosine_baseline(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train(self):
        epoch = self.local_epoch
        device = self.device
        
        criterion = nn.CrossEntropyLoss()
        critic = nn.CosineSimilarity(dim=1)
        
        
        self.model.train()
        
        train_res = []
        
        
        for _ in range(epoch):
            _loss = torch.tensor(.0).to(device)
            for _, (a, v, label) in enumerate(self.dataloader):
                
                a = a.to(device); v = v.to(device); label = label.to(device)
                
                pred,  embedding_a,  embedding_v  = self.model(a, v)
                
                loss = criterion(pred, label)
                loss -= critic(embedding_a, embedding_v).mean().to(device)

                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                
                _loss += loss
            train_res.append(_loss / len(self.dataloader.dataset))
                
        return train_res
    
    
class baseline(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train(self):
        epoch = self.local_epoch
        device = self.device
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        train_res = []
        
        
        for _ in range(epoch):
            _loss = torch.tensor(.0).to(device)
            for _, (a, v, label) in enumerate(self.dataloader):
                
                a = a.to(device); v = v.to(device); label = label.to(device)
                
                pred,  _,  _  = self.model(a, v)
                
                loss = criterion(pred, label)

                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                
                _loss += loss
            train_res.append(_loss / len(self.dataloader.dataset))
                
        return train_res
    
class freeze_Client(Client):
    def __init__(self, **kwargs):
        self.freeze_config = kwargs.pop(
            'freeze_config',
            {
                'audio' : False,
                'visual': False
            }
        )
        super().__init__(**kwargs)
    
    def train(self):
        epoch = self.local_epoch
        device = self.device
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        freeze_config = self.freeze_config
        
        train_res = []
        
        if isinstance(self.model, nn.DataParallel):
            _model = self.model.module
        else:
            _model = self.model
        
        _model.requires_grad_(True)
                
        # unfreeze modality feature extractor
        if freeze_config['audio']:
            print('audio freeze')
            _model.audio_net.requires_grad_(False)
        if freeze_config['visual']:
            print('visual freeze')
            _model.visual_net.requires_grad_(False)
        
        for _ in range(epoch):
            _loss = torch.tensor(.0).to(device)
            for _, (a, v, label) in enumerate(self.dataloader):
                
                a = a.to(device); v = v.to(device); label = label.to(device)
                
                pred,  _,  _  = self.model(a, v)
                
                loss = criterion(pred, label)

                self.optimizer.zero_grad()
                
                loss.backward()
                
                self.optimizer.step()
                
                _loss += loss
            train_res.append(_loss / len(self.dataloader.dataset))
                
        return train_res
    
    def load_model(self, downloads):
        _state_dict = downloads['model_param']
        self.freeze_config = downloads.pop(
            'freeze_config',
            {
                'audio' : False,
                'visual' : False
            }
        )
        self.model.load_state_dict(_state_dict)
        
        



class Server(fed.Server):
    def __init__(self, model, dataloader, device, weights):
        super(Server, self).__init__()
        
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        
        self.weights = weights
        self.uploaded_params = None
        self.aggregated_params = None
    
    def aggregate(self):
        self.aggregated_params = fedAvg([v['model_param'] for v in self.uploaded_params], self.weights)
    
    def get_model(self):
        return self.aggregated_params
    
    
    def load_model(self, params):
        self.aggregated_params = params
        
    def upload_model(self, params):
        self.uploaded_params = params
        
    def validate(self):
        accuracy = 0
        
        self.model.load_state_dict(self.aggregated_params)
        self.model.eval()
        for a, v, label in self.dataloader:
            a = a.to(self.device); v = v.to(self.device); label = label.to(self.device)
            pred, _, _ = self.model(a, v)
            pred = torch.argmax(pred, dim=1, keepdim=False)
            accuracy += (pred == label).sum()
        print('acc: {}'.format(accuracy))

        return accuracy/len(self.dataloader.dataset)
            
    def state(self):
        return {
            'model' : self.aggregated_params
        }
        
    def load_state(self, param):
        self.aggregated_params = param['model']
        
        
class freeze_Server(Server):
    def __init__(self, *args, **kwargs):
        self.freeze_rate = kwargs.pop('freeze_rate', 0.3)
        self.allow_freeze_all = kwargs.pop('allow_freeze_all', False)
        self.cold_start_round = kwargs.pop('cold_start_round', 20)
        
        self.round_cnt = 0
        
        super().__init__(*args, **kwargs)
        
    def validate(self):
        self.round_cnt += 1
        return super().validate()
        
    def get_model(self, idx):
        _seed = 100
        _distribute = {}
        _distribute['model_param'] = self.aggregated_params
        _freeze_config = {
            'audio' : False,
            'visual' : False
        }
        
        if self.round_cnt > self.cold_start_round:
            
            if not self.allow_freeze_all:
                if np.random.binomial(1, self.freeze_rate) == 1:
                    if np.random.rand() < 0.5: _freeze_config['audio'] = True
                    else: _freeze_config['visual'] = True
            else:
                pass
        _distribute['freeze_config'] = _freeze_config

        return _distribute
        
                
            
            
            
         


            
            
            
        
        



if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    '''
        --csv-path: name of directory storing csv file for each of the client,
        which should have the following structure:
                .
                ├── test.csv         
                ├── 0.csv               ... client *.pt files are 0 indexed
                ├──  ⋮
                └── {num_client-1}.csv
                
                
        --data-path: source directory for `ravdess` dataset, which should have the following structure
                .
                ├── image
                └── audio
    
    '''
    
    parser = argparse.ArgumentParser()
    # federated framework settings
    parser.add_argument('--round', default=100, type=int)
    parser.add_argument('--local-epoch', default=5, type=int)
    
    parser.add_argument('--num-frame', default=1, type=int)
    parser.add_argument('--num-client', default=10, type=int)
    parser.add_argument('--num-thread', default=0, type=int)
    parser.add_argument('--num-workers', default=5, type=int)
    
    parser.add_argument('--disable-cuda', action='store_true')
    
    parser.add_argument('--mode', default='new', choices=['new', 'resume'])
    parser.add_argument('--resume-ckpt-path', default=None, type=str)

    
    # training arguments
    parser.add_argument('--csv-path', default=None, type=str)
    parser.add_argument('--data-path', default=None, type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--lr_decay_step', default=20, type=int)
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
    parser.add_argument('--ckpt-path', default='./fed_ckpt', type=str)
    
    
    parser.add_argument('--use-tensorboard', action='store_true')
    parser.add_argument('--tensorboard-path', default='./runs', type=str)
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--gpu-ids', default=None, type=str)
    
    
    args = parser.parse_args()
    
    if args.mode == 'resume' and args.resume_ckpt_path is None:
        raise 'In resume mode, checkpoint path to load the model should be specified'
    
    
    
    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')
        
    print(device)
    


    local_train_config = {
        'device' : device,
        'epoch' : args.local_epoch,
        'batch_size' : args.batch_size,
        'csv_path' : args.csv_path,
        'data_path' : args.data_path,
        'num_workers' : args.num_workers,
        'num_frame' : args.num_frame
    }
    
    model_config = {
        'output_dim' : 8
    }
    
    optimizer_config = {
        'lr'    : args.lr,
        'momentum' : 0.9,
        'weight_decay' : 1e-4
    }
    
    scheduler_config = {
        'lr_decay_step' : args.lr_decay_step * args.local_epoch,
        'lr_decay_ratio' : args.lr_decay_ratio
    }
    
    aggregator_config = {
        'weights' : None
    }
    
    clients = []
    weights = np.zeros(args.num_client)
    server = None
    
    
    if args.mode == 'new':
    
        for i in range(args.num_client):
            _model = late_fusion(output_dim=8)
            
            _avaliable_cuda = None
            if args.data_parallel:
                if args.gpu_ids is None:
                    _avaliable_cuda = [i for i in torch.cuda.device_count()]
                else:
                    _avaliable_cuda = [int(i) for i in args.gpu_ids.split(',')]
                _model = nn.DataParallel(_model, device_ids=_avaliable_cuda)
            
            
            _optimizer = torch.optim.SGD(_model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config['momentum'],
                weight_decay=optimizer_config['weight_decay']
            )
            _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, 
                step_size=scheduler_config['lr_decay_step'],
                gamma=scheduler_config['lr_decay_ratio']
            )
            _csv_path = os.path.join(local_train_config['csv_path'], f'{i}.csv')
            _dataloader = DataLoader(
                ravdess.ravdess_dataset(
                    src_path=local_train_config['data_path'],
                    csv_path=_csv_path,
                    frame_num=local_train_config['num_frame']
                ),
                batch_size=local_train_config['batch_size'],
                shuffle=True,
                pin_memory=True,
                num_workers=args.num_workers
            )
            
            weights[i] = len(_dataloader.dataset)
            
            
            
            clients.append(
                freeze_Client(model=_model, optimizer=_optimizer, scheduler=_scheduler,
                    dataloader=_dataloader, local_epoch=local_train_config['epoch'],
                    device=torch.device(f'cuda:{_avaliable_cuda[0]}'))
            )
        
        
        
        weights = weights / weights.sum()
        
        _val_csv_path = os.path.join(
            local_train_config['csv_path'], 'test.csv' 
        )
        
        _val_dataloader = DataLoader(
            ravdess.ravdess_dataset(
                src_path=local_train_config['data_path'],
                csv_path=_val_csv_path,
                frame_num=local_train_config['num_frame']
            ),
            batch_size=32
        )
        
        
        _model = late_fusion(output_dim=8)
        
        if args.data_parallel:
            if args.gpu_ids is None:
                _avaliable_cuda = [i for i in torch.cuda.device_count()]
            else:
                _avaliable_cuda = [int(i) for i in args.gpu_ids.split(',')]
            _model = nn.DataParallel(_model, device_ids=_avaliable_cuda)
        
        
        
        server = freeze_Server(
            model=_model,
            cold_start_round=30,
            freeze_rate=0.4,
            dataloader=_val_dataloader,
            device=torch.device(f'cuda:{_avaliable_cuda[0]}'),
            weights=weights
        )
        
        # server = Server(
        #     model=_model,
        #     dataloader=_val_dataloader,
        #     device=torch.device(f'cuda:{_avaliable_cuda[0]}'),
        #     weights=weights
        # )
        
        
        
        
        framework = fed.fed_framework(
            clients=clients, server=server,
            personalize=True,
            num_client=args.num_client, num_thread=args.num_thread,
            global_round=args.round,
            tensorboard_path=args.tensorboard_path,
            use_tensorboard=args.use_tensorboard,
            ckpt_path=args.ckpt_path
        )
        framework.run()
    
    elif args.mode == 'resume':
        pass
    
    else:
        raise 'wrong framework mode, which can only be chosen from `new` & `resume`'
    
    
