from ultralytics import YOLO

# Load a model
model = YOLO('checkpoint/best.pt')  # load an official model
x = model.predict('z5222775896173_7260eef54c872a798da9cb3b5d399798.jpg', device='cpu', save=True, imgsz=960, conf=0.35, show_labels=False, line_width=1, agnostic_nms=True, retina_masks=True)
print(type(x))