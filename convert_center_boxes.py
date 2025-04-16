#该文件主要将微小框转换为点坐标
import os

def convert_boxes_to_points(label_dir, img_size=(640, 368)):
    """ 
    将微小框转换为点坐标（YOLO格式）
    参数：
        label_dir: 包含txt标注文件的目录
        img_size: (width, height) 单位：像素
    """
    max_box_size = 8  # 你标注时允许的最大框尺寸（像素）
    
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
            
        path = os.path.join(label_dir, filename)
        with open(path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:  # 若你的格式不同需调整
                continue
                
            cls, x, y, w, h = map(float, parts)
            # 如果是center类且为微小框
            if cls == 1 and w*img_size[0] <= max_box_size and h*img_size[1] <= max_box_size:
                new_lines.append(f"{int(cls)} {x} {y} 0 0\n")  # 转为点坐标
            else:
                new_lines.append(line)
        
        with open(path, 'w') as f:
            f.writelines(new_lines)
    print(f"转换完成：{label_dir}")

def safe_run():
    # 添加环境变量限制
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # 执行转换
    convert_boxes_to_points("datasets/labels/train", img_size=(640, 368))
    convert_boxes_to_points("datasets/labels/val", img_size=(640, 368))
    print("转换完成！")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    safe_run()