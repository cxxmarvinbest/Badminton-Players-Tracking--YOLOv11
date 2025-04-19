import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from .geometry import get_perspective_transform_matrix
from .trajectory import TrajectoryDrawer

class PlayerTracker:
    def __init__(self, model_path, video_path, output_video_path, output_csv_path):
        self.model = YOLO(model_path)
        self.video_path = video_path

        self.output_video_path = output_video_path
        self.output_csv_path = output_csv_path
        self.perspective_matrix = get_perspective_transform_matrix(video_path)
        
        # 运动员ID到姓名的映射
        self.player_names = {
            1: "Antonsen",
            2: "Chou"
        }
        
        # 距离记录
        self.distance_history = defaultdict(lambda: [0])
        for player_id in self.player_names.keys():
            self.distance_history[player_id] = [0]
            
        self.frame_count = 0
        
        # 创建场地背景图和轨迹绘制器
        self.court_img = self.create_court_image()
        self.trajectory_drawer = TrajectoryDrawer(self.perspective_matrix, self.player_names)
        
        # 距离显示框参数
        self.distance_box_position = (20, 20)
        self.distance_box_size = (200, 100)
        self.distance_box_color = (255, 255, 255)
        self.distance_text_color = (0, 0, 0)
        
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
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制边框
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 2)
        
        # 添加标题
        cv2.putText(frame, "Running Distance(m):", (x + 10, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.distance_text_color, 1)
        
        # 添加每个运动员的距离信息
        y_offset = 50
        for track_id in sorted(self.player_names.keys()):
            player_name = self.player_names[track_id]
            distance = self.distance_history[track_id][-1] if track_id in self.distance_history else 0.00
            color = self.trajectory_drawer.trail_colors.get(track_id, (0, 0, 0))
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        success, frame = cap.read()
        if success:
            self.draw_distance_info_box(frame)
            out.write(frame)
            self.frame_count += 1
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            self.frame_count += 1
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            
            for box, track_id in zip(boxes, track_ids):
                if track_id not in self.player_names:
                    continue
                    
                x, y, w, h = box
                center_point = (float(x), float(y + h/2))
                
                # 更新轨迹
                self.trajectory_drawer.update_trajectory(track_id, center_point)
                # 绘制轨迹
                self.trajectory_drawer.draw_trajectory(frame, track_id, center_point)
                
                # 计算移动距离
                if len(self.trajectory_drawer.track_history[track_id]) > 1:
                    prev_point = self.trajectory_drawer.track_history[track_id][-2]
                    current_point = self.trajectory_drawer.track_history[track_id][-1]
                    
                    prev_transformed = cv2.perspectiveTransform(
                        np.array([[[prev_point[0], prev_point[1]]]], dtype=np.float32),
                        self.perspective_matrix
                    )[0][0]
                    current_transformed = cv2.perspectiveTransform(
                        np.array([[[current_point[0], current_point[1]]]], dtype=np.float32),
                        self.perspective_matrix
                    )[0][0]
                    
                    distance = np.linalg.norm(np.array(current_transformed) - np.array(prev_transformed))
                    total_distance = self.distance_history[track_id][-1] + distance
                    self.distance_history[track_id].append(total_distance)
            
            self.draw_distance_info_box(frame)
            out.write(frame)
            
            if self.frame_count % 10 == 0:
                self.trajectory_drawer.save_trajectory_plot(self.court_img, self.frame_count)
                
        cap.release()
        out.release()
        self.save_distance_data()
    
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
