from ultralytics import YOLO
import os
import cv2

# 🔧 المسار إلى النموذج المدرب
model_path = "C:/Users/PC/Desktop/Communication towers/runs/detect/tower_detector/weights/best.pt"

# 🔧 مسار الصورة الأصلية
image_path = "C:/Users/PC/Desktop/Communication towers/image2.jpg"

# ✅ التأكد من وجود الصورة
if not os.path.exists(image_path):
    print(f"❌ الصورة غير موجودة: {image_path}")
    exit()

# ✅ تحميل النموذج
model = YOLO(model_path)

# ✅ تحليل الصورة
print("🚀 جاري تحليل الصورة...")
results = model.predict(source=image_path, conf=0.5, save=False)

# ✅ استخراج النتائج
result = results[0]
annotated_img = result.plot()

# ✅ استخراج أسماء الفئات المكتشفة
names = model.names  # أسماء الفئات من النموذج
detected_classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []

# ✅ تجهيز النص الذي سيُكتب في أعلى الصورة
if len(detected_classes) == 0:
    label_text = "❌ لم يتم الكشف عن أي نوع برج"
else:
    unique_classes = list(set(detected_classes))
    label_names = [names[i] for i in unique_classes]
    label_text = " Type of discovered tower: " + ", ".join(label_names)

# ✅ كتابة النص على الصورة
cv2.putText(annotated_img, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

# ✅ حفظ الصورة الناتجة
save_path = "runs/detect/predict/image_with_label.jpg"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, annotated_img)

print(f"✅ تم حفظ الصورة مع أسماء الأبراج في: {save_path}")

