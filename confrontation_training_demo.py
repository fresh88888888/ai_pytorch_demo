import torch

# 初始化模型和优化器
model = YourTextClassificationModel()
optimizer = torch.optim.Adam(model.parameters())

# 初始化对抗训练工具类
fgm = FGM(model)

# 开始循环训练
for epoch in range(num_epochs):
    for batch_idx, (batch_input, batch_labels) in enumerate(data_loader):
        model.zero_grad()
        output = model(batch_input)
        loss = criterion(output, batch_labels)
        
        # 使用损失函数计算损失
        loss.backward() # 反向传播计算梯度
        
        # 对抗训练
        fgm.attack()
        adv_output = model(fgm.adv_x)
        # 使用对抗样本重新计算模型输出
        adv_loss = criterion(adv_output, batch_labels) # 计算对抗样本的损失
        adv_loss.backward()                            # 对抗样本的反向传播计算梯度
        
        # 合并正则训练与对抗样本训练的梯度更新
        optimizer.step()  # 更新权重（同时考虑正常样本和对抗样本的影响）
        
        # 可选：恢复原始输入，准备下一轮迭代
        fgm.restore()
        
    # 每个epoch结束后的常规操作，如记录日志、评估模型等
        
        
        
        