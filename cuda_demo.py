import torch


# 创建一个在CPU上的张量
cpu_tensor = torch.randn(100, 100)  # 随机生成一个100 x 100 de 浮点数矩阵

# 如果系统安装CUDA并且支持的话，我们可以将CPU上的张量移动到GPU上
if torch.cuda.is_available():
    # 检查是否有可用的GPU
    device = torch.device('cuda')  # 定义设备为CUDA设备
    # 将CPU张量转移到GPU上
    gpu_tensor = cpu_tensor.to(device)
    
    print(f'CPU 张量：{cpu_tensor}')
    print(f'GPU 张量：{gpu_tensor}')

# 在GPU上执行矩阵乘法操作
result = torch.mm(gpu_tensor, gpu_tensor) # 矩阵乘法

# 当需要查看结果或进一步在CPU上处理时，可以将GPU上的结果再撤回到CPU
result_cpu = result.to('cpu')

# 注意：实际使用中，我们会直接在GPU上创建张量，并在那里完成所有计算。