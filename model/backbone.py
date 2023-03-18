import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class residual(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1,
                 use_1x1_conv=False):
        super(residual, self).__init__()
        
        padding = int((kernel_size-1)/2)
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.conv3 = None
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=stride)
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        
        if self.conv3 is not None:
            y += self.conv3(x)
        else: y += x
        
        return F.relu(y)
    
class res_block(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(res_block, self).__init__()
        use_1x1_conv = (input_channels != output_channels) or (stride != 1)
        self.layer1 = residual(input_channels, output_channels, 
                               kernel_size=3, stride=stride, 
                               use_1x1_conv=use_1x1_conv)
        
        self.layer2 = residual(output_channels, output_channels, 
                               kernel_size=3)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    

class ResNet_18(nn.Module):
    def __init__(self, input_channels):
        super(ResNet_18, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, padding=3,
                               stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_blk1 = res_block(64, 128, stride=2)
        self.res_blk2 = res_block(128, 256, stride=2)
        self.res_blk3 = res_block(256, 512, stride=2)
        self.res_blk4 = res_block(512, 512, stride=2)
            
            
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.res_blk1(x)
        x = self.res_blk2(x)
        x = self.res_blk3(x)
        x = self.res_blk4(x)
        return x
    
    
class AVNet(nn.Module):
    def __init__(self, output_dim):
        super(AVNet, self).__init__()
        
        self.audio_net = ResNet_18(input_channels=1)
        self.visual_net = ResNet_18(input_channels=3)
        
        self.ada_pool_a = nn.AdaptiveAvgPool2d((1,1))
        self.ada_pool_v = nn.AdaptiveAvgPool2d((1,1))
        self.flatten_a = nn.Flatten()
        self.flatten_v = nn.Flatten()
        
        self.fc_a = nn.Linear(512, 128)
        self.fc_v = nn.Linear(512, 128)
        
        self.fc_ = nn.Linear(256, output_dim)
        
        
        
    def forward(self, a, v):
        a = self.audio_net(a)
        v = self.visual_net(v)
        
        a = self.flatten_a(self.ada_pool_a(a))
        v = self.flatten_v(self.ada_pool_v(v))
        
        embedding_a = self.fc_a(a); embedding_v = self.fc_v(v)
        
        concat_embedding = torch.concat([
            embedding_a, embedding_v
        ], dim=1)
        
        pred = self.fc_(concat_embedding)
        
        
        
        return pred, embedding_a, embedding_v
class result_level_fusion(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.audio_net = ResNet_18(input_channels=1)
        self.visual_net = ResNet_18(input_channels=3)
        
        self.ada_pool_a = nn.AdaptiveAvgPool2d((1,1))
        self.ada_pool_v = nn.AdaptiveAvgPool2d((1,1))
        self.flatten_a = nn.Flatten()
        self.flatten_v = nn.Flatten()
        
        self.hidden_a = nn.Linear(512, 128)
        self.hidden_v = nn.Linear(512, 128)
        
        self.out_a = nn.Linear(128, output_dim)
        self.out_v = nn.Linear(128, output_dim)
        
        
        
    def forward(self, a, v):
        a = self.audio_net(a)
        v = self.visual_net(v)
        
        a = self.flatten_a(self.ada_pool_a(a))
        v = self.flatten_v(self.ada_pool_v(v))
        
        logits_a = self.out_a(self.hidden_a(a))
        logits_v = self.out_v(self.hidden_v(v))
    
        
        pred = logits_a + logits_v
        
        return pred, logits_a, logits_v

class late_fusion(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.audio_net = ResNet_18(input_channels=1)
        self.visual_net = ResNet_18(input_channels=3)
        
        self.ada_pool_a = nn.AdaptiveAvgPool2d((1,1))
        self.ada_pool_v = nn.AdaptiveAvgPool2d((1,1))
        self.flatten_a = nn.Flatten()
        self.flatten_v = nn.Flatten()
        
        self.mlp= nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, output_dim)
        )
        
        
        
    def forward(self, a, v):
        a = self.audio_net(a)
        v = self.visual_net(v)
        
        a = self.flatten_a(self.ada_pool_a(a))
        v = self.flatten_v(self.ada_pool_v(v))
        
        pred = self.mlp(a + v)
        
        return pred, None, None

if __name__ == '__main__':
    a = torch.zeros((1, 1, 39, 196))
    v = torch.zeros((1, 3, 224, 224))

    net = AVNet(output_dim=8)
    for k, v in net.state_dict().items():
        if k.endswith('num_batches_tracked'):
            print(v.dtype)