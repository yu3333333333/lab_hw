
from ultralytics.models.yolo.detect import DetectionValidator

args = dict(model="/home/tzul/candy/runs/detect/e150_b64/weights/best.pt", data='data.yaml')
validator = DetectionValidator(args=args)
validator()