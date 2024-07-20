import os
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

model = load_model('model.h5')

def get_class_name(class_no):
    return "Pneumonia" if class_no == 0 else "Normal"

def preprocess_image(img):
    img = img.resize((150, 150))  
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array.astype('float32') / 255.0  
    return img_array

def get_result(img_path):
    image = Image.open(img_path)  
    processed_img = preprocess_image(image)  
    result = model.predict(processed_img)  
    class_label = (result > 0.5).astype(int)[0][0]  
    return class_label

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    prediction = get_result(file_path)
    category = get_class_name(prediction)
    return JSONResponse(content={"prediction": category})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
