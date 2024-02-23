import torch
import torch.nn as nn


class CTDataLoader:
    def __init__(self):
        pass
    def __iter__(self):
        while True:
            artifact_ct, clean_ct = load_pair_of_CT_images()
            
            # 加载CT图像
            yield torch.tensor(artifact_ct).unsqueeze(0), torch.tensor(clean_ct).unsqueeze(0)
            
class CTArtifactRemoveGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 更多层...
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid() # 输出判别真伪的概率
        )
    def forward(self, x):
        return self.main(x)

# 初始化生成器和判别器
G = CTArtifactRemoveGenerator()
D = CTArtifactRemoveDiscriminator()

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas= (0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 数据加载器
data_loader = CTDataLoader()

# 训练过程，简化...
for epoch in range(num_epochs):
    for artifact, clean_c in data_loader:
        # 训练判别器
        # 编写代码更新判别器参数以区分真实无伪影图像与生成器...
        
        # 训练生成器

