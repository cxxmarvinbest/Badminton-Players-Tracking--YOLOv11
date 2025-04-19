import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from scipy.signal import savgol_filter

class TrajectoryPlotter:
    def __init__(self, video_path, output_video_path, court_img_path=None):
        """
        初始化轨迹绘图器
        
        参数:
            video_path: 输入视频路径
            output_video_path: 输出视频路径
            court_img_path: 可选的标准场地图片路径，如果为None则生成标准场地
        """
        self.video_path = video_path
        self.output_video_path = output_video_path
        
        # 创建或加载标准羽毛球场地
        self.court_img = self.create_or_load_court_image(court_img_path)
        
        # 存储标定点
        self.video_points = []  # 视频中的四个角点
        self.court_points = []  # 标准场地中的四个对应角点
        
        # 轨迹数据
        self.trajectories = defaultdict(list)
        self.smoothed_trajectories = defaultdict(list)
        
        # 标定状态
        self.calibrated = False
        self.current_point_idx = 0
        
        # 运动员ID到颜色的映射
        self.player_colors = {
            1: (255, 0, 0),  # 红色 - Antonsen
            2: (0, 0, 255)   # 蓝色 - Chou
        }
        
        # 运动员ID到姓名的映射
        self.player_names = {
            1: "Antonsen",
            2: "Chou"
        }
        
        # 轨迹平滑参数
        self.window_size = 25  # 滑动窗口大小(必须为奇数)
        self.poly_order = 2    # 多项式阶数
        
        # 场地显示比例
        self.court_scale = 0.4  # 增大场地显示比例
    
    def create_or_load_court_image(self, court_img_path):
        """创建或加载标准羽毛球场地图像"""
        if court_img_path:
            court_img = cv2.imread(court_img_path)
            if court_img is None:
                raise ValueError(f"无法加载场地图片: {court_img_path}")
            return court_img
        else:
            return self.create_standard_court()
    
    def create_standard_court(self):
        """创建符合国际标准的羽毛球场地"""
        # 标准尺寸（单位：米）
        court_length = 13.40  # 总长度（纵向）
        court_width_double = 6.10  # 双打宽度（横向）
        court_width_single = 5.18  # 单打宽度
        front_service_line = 1.98  # 前发球线距离球网
        back_service_line_single = 3.88  # 单打后发球线距离底线
        back_service_line_double = 0.76  # 双打后发球线距离底线
        
        # 转换为像素比例（保持场地纵横比）
        target_height = 800  # 增大场地图像分辨率
        scale = target_height / court_length
        width_pixels = int(court_width_double * scale)
        length_pixels = target_height
        
        # 创建白色背景图像（竖版：高度>宽度）
        img = np.ones((length_pixels, width_pixels, 3), dtype=np.uint8) * 255
        
        # 计算所有关键线位置（竖版坐标系）
        net_line = length_pixels // 2
        front_service_lines = (
            net_line - int(front_service_line * scale),
            net_line + int(front_service_line * scale)
        )
        back_service_lines_single = (
            int(back_service_line_single * scale),
            length_pixels - int(back_service_line_single * scale)
        )
        back_service_lines_double = (
            int(back_service_line_double * scale),
            length_pixels - int(back_service_line_double * scale)
        )
        single_line_offset = int((court_width_double - court_width_single)/2 * scale)
        
        # 绘制所有标准线（线宽3像素）
        line_thick = 3
        # 1. 双打边线（最外侧）
        cv2.rectangle(img, (0, 0), (width_pixels-1, length_pixels-1), (0, 0, 0), line_thick)
        # 2. 单打边线（内侧）
        cv2.line(img, (single_line_offset, 0), (single_line_offset, length_pixels), 
                (0, 0, 0), line_thick)
        cv2.line(img, (width_pixels-single_line_offset, 0), 
                (width_pixels-single_line_offset, length_pixels), (0, 0, 0), line_thick)
        # 3. 中线（红色加粗）
        cv2.line(img, (width_pixels//2, 0), (width_pixels//2, length_pixels), 
                (0, 0, 255), line_thick+1)
        # 4. 前发球线（两条）
        cv2.line(img, (0, front_service_lines[0]), (width_pixels, front_service_lines[0]), 
                (0, 0, 0), line_thick)
        cv2.line(img, (0, front_service_lines[1]), (width_pixels, front_service_lines[1]), 
                (0, 0, 0), line_thick)
        # 5. 双打后发球线（两条）
        cv2.line(img, (0, back_service_lines_double[0]), (width_pixels, back_service_lines_double[0]), 
                (0, 0, 0), line_thick)
        cv2.line(img, (0, back_service_lines_double[1]), (width_pixels, back_service_lines_double[1]), 
                (0, 0, 0), line_thick)
        # 6. 球网线（黄色加粗）
        cv2.line(img, (0, net_line), (width_pixels, net_line), 
                (0, 255, 255), line_thick+2)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于标定点选择"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point_idx < 4:
                # 在视频帧上标定点
                self.video_points.append((x, y))
                print(f"已标定视频点 {self.current_point_idx + 1}: ({x}, {y})")
                
                # 在标准场地上标定对应点
                court_x, court_y = self.get_court_corner_point(self.current_point_idx)
                self.court_points.append((court_x, court_y))
                print(f"已标定场地点 {self.current_point_idx + 1}: ({court_x}, {court_y})")
                
                self.current_point_idx += 1
                
                # 绘制标定点
                cv2.circle(self.current_frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(self.current_frame, str(self.current_point_idx), 
                           (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # 更新显示
                cv2.imshow("Video Frame", self.current_frame)
                
                if self.current_point_idx == 4:
                    # 所有点已标定，计算透视变换矩阵
                    self.calculate_perspective_matrix()
                    self.calibrated = True
                    print("标定完成！透视变换矩阵已计算。")
    
    def get_court_corner_point(self, idx):
        """获取标准场地的四个角点坐标"""
        h, w = self.court_img.shape[:2]
        if idx == 0:  # 左上角
            return (0, 0)
        elif idx == 1:  # 右上角
            return (w - 1, 0)
        elif idx == 2:  # 右下角
            return (w - 1, h - 1)
        elif idx == 3:  # 左下角
            return (0, h - 1)
        else:
            raise ValueError("无效的角点索引")
    
    def calculate_perspective_matrix(self):
        """计算从视频坐标系到场地坐标系的透视变换矩阵"""
        if len(self.video_points) != 4 or len(self.court_points) != 4:
            raise ValueError("需要标定4个点才能计算透视变换矩阵")
        
        # 转换为numpy数组
        src_points = np.array(self.video_points, dtype=np.float32)
        dst_points = np.array(self.court_points, dtype=np.float32)
        
        # 计算透视变换矩阵
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        print("透视变换矩阵:")
        print(self.perspective_matrix)
    
    def manual_calibration(self, frame):
        """手动标定视频帧与标准场地的对应点"""
        self.current_frame = frame.copy()
        cv2.namedWindow("Video Frame")
        cv2.setMouseCallback("Video Frame", self.mouse_callback)
        
        print("请在视频帧上按顺序点击羽毛球场的四个角点:")
        print("1. 左上角")
        print("2. 右上角")
        print("3. 右下角")
        print("4. 左下角")
        
        while True:
            cv2.imshow("Video Frame", self.current_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.calibrated:
                break
        
        cv2.destroyWindow("Video Frame")
        return self.calibrated
    
    def smooth_trajectory(self, trajectory):
        """使用Savitzky-Golay滤波器平滑轨迹"""
        if len(trajectory) < self.window_size:
            return trajectory
        
        # 提取x和y坐标
        x = np.array([p[0] for p in trajectory])
        y = np.array([p[1] for p in trajectory])
        
        try:
            # 平滑x和y坐标
            x_smooth = savgol_filter(x, self.window_size, self.poly_order)
            y_smooth = savgol_filter(y, self.window_size, self.poly_order)
            
            # 组合成平滑后的轨迹
            smoothed = list(zip(x_smooth, y_smooth))
            return smoothed
        except:
            return trajectory
    
    def process_video(self):
        """处理视频并绘制轨迹"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        # 读取第一帧进行标定
        ret, frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频帧")
        
        # 手动标定
        if not self.manual_calibration(frame.copy()):
            raise ValueError("标定失败，无法继续处理视频")
        
        # 重置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 初始化YOLO模型
        model = YOLO('models/best.pt')  # 使用适合的模型
        
        # 处理视频帧
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测运动员
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            # 获取检测框和跟踪ID
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            
            # 处理每个检测到的运动员
            for box, track_id in zip(boxes, track_ids):
                if track_id not in self.player_names:
                    continue
                
                x, y, w, h = box
                center_point = (float(x), float(y + h/2))
                
                # 存储原始轨迹点
                self.trajectories[track_id].append(center_point)
                
                # 平滑轨迹
                if len(self.trajectories[track_id]) > self.window_size:
                    self.smoothed_trajectories[track_id] = self.smooth_trajectory(self.trajectories[track_id])
            
            # 只在标准场地上绘制轨迹
            if self.calibrated:
                # 创建标准场地副本用于绘制
                court_img = self.court_img.copy()
                
                # 绘制所有运动员的轨迹
                for player_id in self.trajectories:
                    # 使用平滑后的轨迹或原始轨迹
                    points = self.smoothed_trajectories.get(player_id, self.trajectories[player_id])
                    
                    if len(points) < 2:
                        continue
                    
                    color = self.player_colors[player_id]
                    name = self.player_names[player_id]
                    
                    # 转换所有点到场地坐标系
                    pts = np.array([[[p[0], p[1]]] for p in points], dtype=np.float32)
                    transformed_pts = cv2.perspectiveTransform(pts, self.perspective_matrix)
                    
                    # 绘制场地上的轨迹
                    for i in range(1, len(transformed_pts)):
                        pt1 = (int(transformed_pts[i-1][0][0]), 
                               int(transformed_pts[i-1][0][1]))
                        pt2 = (int(transformed_pts[i][0][0]), 
                               int(transformed_pts[i][0][1]))
                        cv2.line(court_img, pt1, pt2, color, 3)
                    
                    # 添加运动员姓名标签
                    if len(transformed_pts) > 0:
                        last_point = (int(transformed_pts[-1][0][0]), 
                                     int(transformed_pts[-1][0][1]))
                        cv2.putText(court_img, name, 
                                   (last_point[0] + 10, last_point[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 将标准场地叠加到视频右下角
                h, w = court_img.shape[:2]
                scaled_court = cv2.resize(court_img, 
                                        (int(w * self.court_scale), 
                                         int(h * self.court_scale)))
                
                # 叠加场地到视频帧右下角
                frame_h, frame_w = frame.shape[:2]
                roi = frame[frame_h-scaled_court.shape[0]-10:frame_h-10, 
                            frame_w-scaled_court.shape[1]-10:frame_w-10]
                
                # 透明叠加
                blended = cv2.addWeighted(scaled_court, 0.7, roi, 0.3, 0)
                frame[frame_h-scaled_court.shape[0]-10:frame_h-10, 
                      frame_w-scaled_court.shape[1]-10:frame_w-10] = blended
            
            # 写入输出视频
            out.write(frame)
            
            # 显示处理进度
            if frame_count % 10 == 0:
                print(f"已处理 {frame_count} 帧")
        
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"视频处理完成，结果已保存到 {self.output_video_path}")

# 使用示例
if __name__ == "__main__":
    # 输入视频路径和输出视频路径
    input_video = "output.mp4"
    output_video = "trajectory_mapped.mp4"
    
    # 创建轨迹绘图器并处理视频
    plotter = TrajectoryPlotter(input_video, output_video)
    plotter.process_video()