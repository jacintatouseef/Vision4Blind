import kagglehub
from ultralytics import YOLO
import os



model = YOLO("yolov8n-cls.pt")

model.train(
    data="C:/Users/muhammad sheeraz/.cache/kagglehub/datasets/muhammadammarabid/pakistanicurrencydataset/versions/1/data-rescaled",
    epochs=50,
    imgsz=224,
    batch=16,
    name="pak_currency_model"
)