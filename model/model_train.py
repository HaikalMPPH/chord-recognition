from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(data="../dataset/decorative-plants", epochs=5, imgsz=320)

model.save("./model-trained.pt")
