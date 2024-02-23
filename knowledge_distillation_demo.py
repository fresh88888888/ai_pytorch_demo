import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50

# 定义教师模型（复杂模型）
teacher_model = resnet50(weights=True) # 使用预训练好的resnet50作为教师模型
teacher_model.eval()                   # 设置为评估模式，因为这里我们不需要进一步训练教师模型

# 定义学生模型（较小模型）
student_model = resnet18(num_classes = teacher_model.fc.out_features) # 学生模型的输出类别数与教师模型相同

# 定义损失函数（只是蒸馏通常用KL散度作为损失）
loss_fn = nn.KLDivLoss(reduction='batchmean')  # Kullback-Leibler Divergence用于比较两个概率分布

# 假设我们有一个数据加载器data_loader
data_loader = []
tempperature = 1.0

optimizer = optim.SGD(student_model.parameters(), lr=0.01)

for images, labels in data_loader:
    # 将图像输入到教师和学生模型中得到预测结果
    with torch.no_grad():
        # 教师模型不更新参数，所以关闭梯度计算
        teacher_outputs = teacher_model(images)
        # 软化教师模型的输出（tempperature时刻调节的温度参数）
        teacher_probs = torch.softmax(teacher_outputs / tempperature, dim=1)
    
    student_outputs = student_model(images)
    # 对学生模型的输出同样进行软化
    student_probs = torch.softmax(student_outputs / tempperature, dim=1)
    
    # 计算KL散度损失
    loss = loss_fn(student_probs, teacher_probs)

    # 反向传播并优化学生模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# 这里省略了实际训练过程中循环、学习了调整和验证等细节
