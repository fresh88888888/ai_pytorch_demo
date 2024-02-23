import torch
import torch.nn as nn

# 定义变分自编码器模型类
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # 定义编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 2* latent_dim),
        )
        
        # 分离出均值和标准差
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        
        # 定义解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )
    
    def reparameterize(self, mu, log_var):
        '''
        实现重新参数化，从给定的均值和对数方差中采样
        '''
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)     # 从标准正态分布中采样噪声
        return mu * eps * std
    
    def forward(self, x):
        '''
        前向传播过程包括编码、采样和解码步骤
        '''
        # 编码阶段得到的均值和方差
        z_params = self.encoder(x)
        mu = self.mu(z_params)
        log_var = self.log_var(z_params)
        
        # 通过reparmeterize函数进行采样
        z = self.reparameterize(mu, log_var)
        
        # 解码阶段从采样的潜在变量生成重构数据
        reconstructed_x = self.decoder(z)
        
        return reconstructed_x, mu, log_var


# 实例化VAE模型
model = VAE(24, 32)
