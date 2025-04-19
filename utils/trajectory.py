import cv2
import numpy as np
from collections import defaultdict

class TrajectoryDrawer:
    def __init__(self, perspective_matrix, player_names):
        self.perspective_matrix = perspective_matrix
        self.player_names = player_names
        
        # 轨迹可视化参数
        self.max_trail_length = 60  # 最大轨迹长度（帧数）
        self.trail_colors = {
            1: (255, 0, 0),  # 玩家1颜色（红色）
            2: (0, 0, 255)   # 玩家2颜色（蓝色）
        }
        
        # 轨迹历史记录
        self.track_history = defaultdict(lambda: [])
        
    def update_trajectory(self, track_id, point):
        """更新轨迹历史"""
        if track_id not in self.player_names:
            return
            
        self.track_history[track_id].append(point)
        if len(self.track_history[track_id]) > self.max_trail_length:
            self.track_history[track_id].pop(0)
    
    def draw_trajectory(self, frame, track_id, current_point):
        """在帧上绘制单个运动员的轨迹"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return
            
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
        cv2.circle(frame, (int(current_point[0]), int(current_point[1])), 8, 
                  self.trail_colors.get(track_id, (0, 255, 255)), -1)
    
    def save_trajectory_plot(self, court_img, frame_count):
        """保存带运动员名称的轨迹图"""
        plot_img = court_img.copy()
        
        for track_id, points in self.track_history.items():
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
        
        cv2.imwrite(f"trajectory_plot_{frame_count}.png", plot_img)