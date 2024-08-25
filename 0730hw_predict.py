from ultralytics import YOLO

model = YOLO("/home/tzul/candy/runs/detect/e150_b64/weights/best.pt")


result = model.predict(source="/home/tzul/candy/dataset_yolov5/test/images", mode="predict", save=True, device="cuda")