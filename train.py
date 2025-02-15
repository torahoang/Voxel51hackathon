from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="data    .yaml", epochs=10, imgsz=640, device=0, lr0= 0.0001)