import fed
import argparse
from model.backbone import AVNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


import copy
import os

from aggregator.aggregator import fedAvg
from dataset import ravdess





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
        
        for _ in range(epoch):
            for _, (a, v, label) in enumerate(self.dataloader):
                
                a = a.to(device); v = v.to(device); label = label.to(device)
                
                pred,  embedding_a,  embedding_v  = self.model(a, v)
                
                loss = criterion(pred, label)
                
                loss -= critic(embedding_a, embedding_v).mean()

                if self.cached_model is not None:
                    with torch.no_grad():
                        self.cached_model.eval()
                        _, _embedding_a, _embedding_v  = self.cached_model(a, v)
                
                    loss += (-(
                        critic(embedding_a, _embedding_a) +
                        critic(embedding_v, _embedding_v) +
                        critic(embedding_a, _embedding_v) +
                        critic(embedding_v, _embedding_a)
                    )/4).mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
            
    
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
        self.model.eval()
        for a, v, label in self.dataloader:
            a = a.to(self.device); v = v.to(self.device); label = label.to(self.device)
            pred, _, _ = self.model(a, v)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            accuracy += (pred == label).sum()
        
        return accuracy/len(self.dataloader.dataset)
            
    def state(self):
        return {
            'model' : self.aggregated_params
        }
        
    def load_state(self, param):
        self.aggregated_params = param['model']


            
            
            
        
        



if __name__ == '__main__':
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
    
    
    parser.add_argument('--use-tensorboard', action='store_true')
    
    
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
    
    for i in range(args.num_client):
        _model = AVNet(output_dim=8)
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
            num_workers=32,
            pin_memory=True
        )
        
        weights[i] = len(_dataloader.dataset)
        
        
        
        clients.append(
            Client(model=_model, optimizer=_optimizer, scheduler=_scheduler,
                   dataloader=_dataloader, local_epoch=local_train_config['epoch'],
                   device=local_train_config['device'])
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
    
    
    
    server = Server(
        model=AVNet(output_dim=8),
        dataloader=_val_dataloader,
        device=device,
        weights=weights
    )
     
    
    framework = fed.fed_framework(
         clients=clients, server=server,
         num_client=args.num_client, num_thread=args.num_thread,
         global_round=args.round
     )
    framework.run()
    
    
