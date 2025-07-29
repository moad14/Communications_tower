from ultralytics import YOLO
import os

# 🔧 المسار إلى النموذج المدرب (عدله لو تغيّر)
model_path = "C:/Users/PC/Desktop/Communication towers/runs/detect/tower_detector/weights/best.pt"

# 🔧 مسار الصورة التي تريد تحليلها
image_path = "C:/Users/PC/Desktop/Communication towers/image2.jpg"  # ← غيّر هذا المسار حسب الصورة

# ✅ التأكد من وجود الصورة
if not os.path.exists(image_path):
    print(f"❌ الصورة غير موجودة: {image_path}")
    exit()

# ✅ تحميل النموذج
model = YOLO(model_path)

# ✅ تحليل الصورة وحفظ النتيجة
print("🚀 جاري تحليل الصورة...")
model.predict(source=image_path, save=True, conf=0.5)
print("✅ تم التحليل! النتيجة محفوظة في مجلد runs/detect/predict/")
