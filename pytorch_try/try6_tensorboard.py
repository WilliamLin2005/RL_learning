import math
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # 实例化 TensorBoard 记录器
    writer = SummaryWriter("logs/sine_wave")
    funcs={"sin":math.sin,"cos":math.cos,"tan":math.tan}

    for angle in range(-360,360):
        angle_rad = math.radians(angle)
        for func_name, func in funcs.items():
            value = func(angle_rad)
            writer.add_scalar(f"{func_name}_wave", value, angle)
    writer.close()
    print("数据已写入 TensorBoard。请在终端运行 'tensorboard --logdir=logs' 查看波形曲线。")