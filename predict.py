import cv2
import typing
import numpy as np

from imageutils.inferenceModel import OnnxInferenceModel
from imageutils.utils.text_utils import ctc_decoder, get_cer, get_wer
from imageutils.transformers import ImageResizer





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

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from imageutils.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/202304010616/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    df = pd.read_csv("Models/202304010616/val.csv").values.tolist()
    accum_cer, accum_wer = [], []
    image = cv2.imread("Datasets/1.png")
    prediction_text = model.predict(image)

 
    print("Image: ", image)
    print("Prediction: ", prediction_text)
 



        #cv2.imshow(prediction_text, image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

