from ultralytics import YOLO

# Load a model
model = YOLO('Yolo11n.pt')

# Train the model
model.train(data='yolo-badminton.yaml', workers=0, epochs=30, batch=8)