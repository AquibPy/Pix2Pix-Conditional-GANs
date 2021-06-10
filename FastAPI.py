import shutil
from io import BytesIO
import os
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import FileResponse
from helper import generate_image

app = FastAPI()

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.get('/')
def index():
    return {"GAN": 'This Model Generate satellite images into google map'}


@app.post('/uplaodfile/')
async def create_upload_file(file:UploadFile = File(...)):
    with open(os.path.join(f"images/input/",file.filename),'wb+') as buffer:
        shutil.copyfileobj(file.file, buffer)
    input_file_path = f"images/input/{file.filename}"
    output_file_name = generate_image(input_file_path)
    output_file_path = f"images/output/{output_file_name}"
    return FileResponse(output_file_path)

if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)