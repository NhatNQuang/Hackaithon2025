import os
# Vô hiệu hóa GPU và giảm log
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Chạy trên CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Giảm log lỗi/warning
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # Tắt JIT của XLA

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import pickle
import time

# Khởi tạo FastAPI
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Cấu hình static files và templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model (trong try-except để bắt lỗi)
try:
    model = tf.keras.models.load_model("MobileNetV2_model_alpha035.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    raise SystemExit(1)

# Route cho trang chính
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route xử lý dự đoán
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        
        # Đọc và xử lý ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))  # Kích thước phù hợp với MobileNetV2
        image_array = np.array(image) / 255.0  # Chuẩn hóa
        image_array = np.expand_dims(image_array, axis=0)  # Thêm batch dimension

        # Dự đoán
        prediction = model.predict(image_array)
        result = "real" if prediction[0][0] > 0.5 else "fake"  # Ví dụ ngưỡng
        confidence = float(prediction[0][0])

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        return {
            "result": result,
            "confidence": confidence,
            "processing_time": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, workers=1)  # 1 worker cho t2.micro
