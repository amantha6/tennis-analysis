from ultralytics import YOLO
model=YOLO('yolov8x')

result = model.predict('/mnt/c/Users/SUNYLoaner/Downloads/tennis-analysis/input_videos/input_video.mp4', save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)