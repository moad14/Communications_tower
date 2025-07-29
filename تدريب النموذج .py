from ultralytics import YOLO

# تحميل نموذج YOLOv8 خفيف (يمكن تغييره إلى yolov8s.pt أو yolov8m.pt حسب قوة جهازك)
model = YOLO("yolov8s.pt")

# تدريب النموذج على بيانات الأبراج
model.train(
    data=r"C:\Users\PC\Desktop\Communication towers\tower_data\data.yaml",  # ← غيّر المسار إذا لزم الأمر
    epochs=50,
    imgsz=640,
    batch=8,
    name="tower_detector",
)
