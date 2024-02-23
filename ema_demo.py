import torch

# 定义一个简单的类来管理指数移动平均
class ExponentMovingAverage:
    def __init__(self, model, decay_rate = 0.99):
        '''
        初始化指数移动平均类
        '''
        self.model = model 
        self.decay_rate = decay_rate
        # 创建一个字典来存储EMA版本的权重和偏置
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.require_grad and name in self.shadow:
                new_avrage = (1 - self.decay_rate) * param.data + self.decay_rate * self.show[name]
                self.shadow[name].copy_(new_avrage)
    
    def apply_shadow(self):
        '''
        将EMA计算出的权重应用原模型
        '''
        for name, param in self.model.name_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        
# 模型实例化
model = SomePyTorchModel()

# 创建EMA对象
ema = ExponentMovingAverage(model, decay_rate=0.9999)

# 在每次训练步骤后更新EMA
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 训练模型
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 更新EMA权重
        ema.update()
        
# 在训练结束后，可以将EMA权重应用回原始模型以进行推断或评估
ema.apply_shadow()
