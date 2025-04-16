import cv2
import numpy as np

def select_points(event, x, y, flags, param):
    """ 鼠标回调函数，用于选择角点 """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"已选择点({x}, {y})")
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Court Corners", frame)

def get_perspective_transform_matrix(video_path):
    # 初始化
    global frame, points
    points = []
    
    # 读取视频帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频帧！请检查路径是否正确")
    
    # 交互式选择角点
    cv2.namedWindow("Select Court Corners")
    cv2.setMouseCallback("Select Court Corners", select_points)
    
    print("请按顺序点击场地的四个角点：左上→右上→右下→左下")
    while len(points) < 4:
        cv2.imshow("Select Court Corners", frame)
        if cv2.waitKey(20) & 0xFF == 27:  # ESC退出
            break
    
    if len(points) != 4:
        raise ValueError("必须选择4个点！")
    
    # 设置源点和目标点
    src_points = np.float32(points)
    court_length = 13.4  # 米
    court_width = 6.7    # 米
    dst_points = np.float32([
        [0, 0],
        [court_length, 0],
        [court_length, court_width],
        [0, court_width]
    ])
    
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    cv2.destroyAllWindows()
    return M

# 使用示例
if __name__ == "__main__":
    video_path = "test.mp4"  # 修改为你的视频路径
    try:
        M = get_perspective_transform_matrix(video_path)
        print("透视变换矩阵：\n", M)
    except Exception as e:
        print("错误：", e)