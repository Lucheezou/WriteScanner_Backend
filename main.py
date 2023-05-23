from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

import base64
import cv2
import typing
import numpy as np

from imageutils.inferenceModel import OnnxInferenceModel
from imageutils.utils.text_utils import ctc_decoder, get_cer, get_wer
from imageutils.transformers import ImageResizer

import pandas as pd
from tqdm import tqdm
from imageutils.configs import BaseModelConfigs

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


configs = BaseModelConfigs.load("Models/202304010616/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
df = pd.read_csv("Models/202304010616/val.csv").values.tolist()
accum_cer, accum_wer = [], []



@app.post("/upload")
def upload(filename: str = Form(...), filedata: str = Form(...)):
    image_as_bytes = str.encode(filedata)  # convert string to bytes
    img_recovered = base64.b64decode(image_as_bytes)  # decode base64string
    try:
        with open("image.jpg", "wb") as f:
            f.write(img_recovered)
    except Exception:
        return {"message": "There was an error uploading the file"}
        
    image = cv2.imread("image.jpg")
    prediction_text = model.predict(image)
    print(prediction_text)
    return {"prediction" : prediction_text}








