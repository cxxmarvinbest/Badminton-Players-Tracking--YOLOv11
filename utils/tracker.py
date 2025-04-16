import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from .geometry import get_perspective_transform_matrix

class PlayerTracker:
    def __init__(self, model_path, video_path, output_video_path, output_csv_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.output_csv_path = output_csv_path
        self.perspective_matrix = get_perspective_transform_matrix(video_path)
        
        # 轨迹历史记录（存储最近60帧的点）
        self.track_history = defaultdict(lambda: [])
        # 运动员ID到姓名的映射（只包含我们关心的运动员）
        self.player_names = {
            1: "Antonsen",
            2: "Chou"
        }
        # 距离记录（初始化时就为两位运动员创建记录）
        self.distance_history = defaultdict(lambda: [0])
        # 确保两位运动员都有初始距离记录
        for player_id in self.player_names.keys():
            self.distance_history[player_id] = [0]
            
        self.frame_count = 0
        
        # 创建场地背景图
        self.court_img = self.create_court_image()
        
        # 轨迹可视化参数
        self.max_trail_length = 60  # 最大轨迹长度（帧数）
        self.trail_colors = {
            1: (255, 0, 0),  # 玩家1颜色（蓝色）
            2: (0, 0, 255)   # 玩家2颜色（红色）
        }
        
        # 距离显示框参数
        self.distance_box_position = (20, 20)  # 左上角位置
        self.distance_box_size = (200, 100)    # 宽度, 高度
        self.distance_box_color = (255, 255, 255)  # 白色背景
        self.distance_text_color = (0, 0, 0)       # 黑色文字
        
    def create_court_image(self):
        """创建羽毛球场地背景图"""
        court_length_pixels = 1340
        court_width_pixels = 670
        img = np.ones((court_width_pixels, court_length_pixels, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (0, 0), (court_length_pixels, court_width_pixels), (0, 0, 0), 2)
        cv2.line(img, (court_length_pixels//2, 0), (court_length_pixels//2, court_width_pixels), (0, 0, 0), 1)
        return img
    
    def draw_distance_info_box(self, frame):
        """在视频帧上绘制距离信息框"""
        x, y = self.distance_box_position
        width, height = self.distance_box_size
        
        # 绘制半透明背景框
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     self.distance_box_color, -1)
        alpha = 0.7  # 透明度
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制边框
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 2)
        
        # 添加标题（字体缩小到0.5）
        cv2.putText(frame, "Running Distance(m):", (x + 10, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.distance_text_color, 1)
        
        # 添加每个运动员的距离信息
        y_offset = 50
        # 按照player_names的顺序显示，确保顺序一致
        for track_id in sorted(self.player_names.keys()):
            player_name = self.player_names[track_id]
            # 获取最新距离，如果没有记录则显示0.00
            distance = self.distance_history[track_id][-1] if track_id in self.distance_history else 0.00
            
            # 使用对应的轨迹颜色显示
            color = self.trail_colors.get(track_id, (0, 0, 0))
            cv2.putText(frame, f"{player_name}: {distance:.2f}", 
                       (x + 10, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 视频输出设置
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        # 读取第一帧
        success, frame = cap.read()
        if success:
            # 在第一帧就绘制距离信息框
            self.draw_distance_info_box(frame)
            out.write(frame)
            self.frame_count += 1
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            self.frame_count += 1
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            # 获取检测结果
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            
            for box, track_id in zip(boxes, track_ids):
                # 只跟踪我们关心的运动员（1和2）
                if track_id not in self.player_names:
                    continue
                    
                x, y, w, h = box
                # 计算脚部中心点（框底部中心）
                center_point = np.array([[float(x), float(y + h/2)]])
                
                # 透视变换到场地坐标系
                transformed_point = cv2.perspectiveTransform(
                    center_point.reshape(1, -1, 2), 
                    self.perspective_matrix
                )[0][0]
                
                # 更新轨迹历史
                self.track_history[track_id].append((float(x), float(y + h/2)))
                # 限制轨迹长度
                if len(self.track_history[track_id]) > self.max_trail_length:
                    self.track_history[track_id].pop(0)

                # 计算移动距离
                if len(self.track_history[track_id]) > 1:
                    # 获取历史的两个点（屏幕坐标）
                    prev_point = self.track_history[track_id][-2]
                    current_point = self.track_history[track_id][-1]
                    
                    # 将两个点转换到场地坐标系
                    prev_transformed = cv2.perspectiveTransform(
                        np.array([[[prev_point[0], prev_point[1]]]], dtype=np.float32),
                        self.perspective_matrix
                    )[0][0]
                    current_transformed = cv2.perspectiveTransform(
                        np.array([[[current_point[0], current_point[1]]]], dtype=np.float32),
                        self.perspective_matrix
                    )[0][0]
                    
                    # 计算在场地坐标系中的距离
                    distance = np.linalg.norm(np.array(current_transformed) - np.array(prev_transformed))
                    total_distance = self.distance_history[track_id][-1] + distance
                    self.distance_history[track_id].append(total_distance)
                
                # 绘制轨迹残影（渐变色）
                trail_points = self.track_history[track_id]
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)  # 透明度渐变
                    color = [int(c * alpha) for c in self.trail_colors.get(track_id, (0, 255, 255))]
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, 
                             (int(trail_points[i-1][0]), int(trail_points[i-1][1])),
                             (int(trail_points[i][0]), int(trail_points[i][1])),
                             color, thickness)
                
                # 绘制当前中心点（大圆点）
                cv2.circle(frame, (int(x), int(y + h/2)), 8, self.trail_colors.get(track_id, (0, 255, 255)), -1)
            
            # 绘制距离信息框
            self.draw_distance_info_box(frame)
            
            # 写入帧到输出视频
            out.write(frame)
            
            # 定期保存轨迹图
            if self.frame_count % 10 == 0:
                self.save_trajectory_plot()
                
        cap.release()
        out.release()
        self.save_distance_data()
    
    def save_trajectory_plot(self):
        """保存带运动员名称的轨迹图"""
        plot_img = self.court_img.copy()
        
        for track_id, points in self.track_history.items():
            # 只绘制我们关心的运动员
            if track_id not in self.player_names or len(points) < 2:
                continue
                
            # 转换点到场地坐标系
            court_points = []
            for pt in points:
                transformed = cv2.perspectiveTransform(
                    np.array([[[pt[0], pt[1]]]], dtype=np.float32),
                    self.perspective_matrix
                )[0][0]
                court_points.append(transformed)
            
            # 绘制轨迹
            pts = np.array(court_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(plot_img, [pts], False, self.trail_colors.get(track_id, (0, 255, 255)), 2)
            
            # 绘制当前点和名称
            last_point = court_points[-1]
            cv2.circle(plot_img, (int(last_point[0]), int(last_point[1])), 8, 
                    self.trail_colors.get(track_id, (0, 255, 255)), -1)
            player_name = self.player_names[track_id]
            cv2.putText(plot_img, player_name, 
                    (int(last_point[0]) + 15, int(last_point[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.imwrite(f"trajectory_plot_{self.frame_count}.png", plot_img)
    
    def save_distance_data(self):
        """保存带运动员名称的距离数据"""
        max_length = max(len(distances) for distances in self.distance_history.values() if len(distances) > 0)
        if max_length == 0:
            return
            
        data = {"frame": list(range(1, max_length + 1))}
        
        for track_id in self.player_names.keys():
            if track_id in self.distance_history:
                distances = self.distance_history[track_id]
                padded_distances = distances + [distances[-1]] * (max_length - len(distances))
                player_name = self.player_names[track_id]
                data[f"{player_name}_distance"] = padded_distances
            
        df = pd.DataFrame(data)
        df.to_csv(self.output_csv_path, index=False)