from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
import cv2


model = YOLO('checkpoint/best.pt')  # load an official model
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    bytes_file = await file.read()
    input_image = Image.open(io.BytesIO(bytes_file)).convert("RGB")
    result = model.predict(input_image, device='cpu', save=True, imgsz=960, conf=0.5, show_labels=False, line_width=1, agnostic_nms=True, retina_masks=True, iou=0.4)

    num_bags = file.filename.split('_')[0]
    print('num_bags: ', num_bags)
    predict_num_bags = len(result[0].boxes)
    print('predict_num_bags: ', predict_num_bags)

    if int(predict_num_bags) == int(num_bags):
        color = (0,255,0)
        print('green')
    else: color = (0,0, 255)

    text = f'predict/label: {predict_num_bags}/{num_bags}'
    result_path = f'runs/segment/predict/image0.jpg'
    img_result = cv2.imread(result_path)
    img_result = cv2.putText(img = img_result, text = text, org = (20, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = color, thickness = 2)
    cv2.imwrite(result_path, img_result)

    return FileResponse(result_path)




