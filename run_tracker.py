# 方法一：直接从utils模块导入（推荐）
from utils.tracker import PlayerTracker
import os

# 方法二：或通过相对导入（如果文件在同一目录下）
# from .utils.tracker import PlayerTracker  # 注意开头的点号
if __name__ == "__main__":
    # 解决Windows多进程问题
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # 初始化跟踪器
    tracker = PlayerTracker(
        model_path="models/best.pt",
        video_path="videos/test.mp4",
        output_video_path="output.mp4",
        output_csv_path="distances.csv"
    )
    
    # === 开始处理 ===
    tracker.process_video()
    print("处理完成！结果已保存到：")
    print(f"- 轨迹视频: {tracker.output_video_path}")
    print(f"- 距离数据: {tracker.output_csv_path}")