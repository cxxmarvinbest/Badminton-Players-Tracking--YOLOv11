import os
from ultralytics import YOLO

def main():
    # 强制禁用多进程相关环境变量
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # 加载模型
    model = YOLO('yolo11n.pt')
    
    # 训练配置（关键修改：workers=0）
    results = model.train(
        data='datasets/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='badminton_tracker',
        workers=0,  # 完全禁用多进程
        device=0    # 明确指定使用GPU
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()