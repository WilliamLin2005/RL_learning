import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class my_nn(nn.Module):
    def __init__(self, num_input, num_output, dropout_prob=0.3):
        super(my_nn, self).__init__()
        self.function = nn.Sequential(
            nn.Linear(num_input, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(3, num_output)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.function(x)
    
if __name__ == '__main__':
    # ---------------------------------------------------------
    # 1. 超参数与环境初始化
    # ---------------------------------------------------------
    num_epochs = 200
    learning_rate = 0.01
    
    # 实例化模型
    net = my_nn(num_input=2, num_output=10)
    
    # 定义损失函数：多分类任务使用 CrossEntropyLoss（内部包含 Softmax）
    loss = nn.CrossEntropyLoss()
    
    # 定义优化器：使用 Adam 优化器（自适应学习率的随机梯度下降变体）
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # 实例化 TensorBoard 记录器
    writer = SummaryWriter("logs/my_nn_training")

    # ---------------------------------------------------------
    # 2. 构造伪造数据集 (Dummy Data)
    # ---------------------------------------------------------
    # 输入特征：Batch size = 2，特征维度 = 2
    inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    
    # 真实标签：对于 CrossEntropyLoss，标签必须是类别索引（Long 类型），范围在 [0, num_output-1]
    # 这里我们假设第一条数据属于类别 3，第二条数据属于类别 7
    targets = torch.tensor([3, 7], dtype=torch.long)

    print("网络结构:\n", net)
    print("-" * 50)
    print("开始训练...")

    # ---------------------------------------------------------
    # 3. 核心训练循环
    # ---------------------------------------------------------
    # 启用训练模式（此时 Dropout 会按照指定概率随机丢弃神经元）
    net.train() 
    
    for epoch in range(num_epochs):
        # Step 1: 梯度清零
        optimizer.zero_grad()
        
        # Step 2: 前向传播
        outputs = net(inputs)
        
        # Step 3: 计算损失
        loss_t = loss(outputs, targets)
        
        # Step 4: 反向传播
        loss_t.backward()
        
        # Step 5: 参数更新
        optimizer.step()
        
        # 将当前 Loss 写入 TensorBoard
        writer.add_scalar("Training/Loss", loss_t.item(), epoch)
        
        # 每隔 20 次迭代打印一次日志，观察收敛情况
        if (epoch + 1) % 20 == 0:
            print(f"Epoch[{epoch+1:03d}/{num_epochs}], Loss: {loss_t.item():.4f}")

    # ---------------------------------------------------------
    # 4. 训练结束与测试验证
    # ---------------------------------------------------------
    writer.close()
    print("-" * 50)
    print("训练结束。请在终端运行 'tensorboard --logdir=logs' 查看损失曲线。")
    
    # 切换到评估模式（此时 Dropout 停止工作，保证输出确定性）
    net.eval()
    with torch.no_grad(): # 停止梯度计算，节约内存
        final_output = net(inputs)
        # 若需获取最终的概率分布，可在此时手动增加 Softmax
        probabilities = torch.softmax(final_output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        print("\n最终预测概率分布:\n", probabilities.numpy())
        print("预测类别:", predicted_classes.numpy())
        print("真实类别:", targets.numpy())