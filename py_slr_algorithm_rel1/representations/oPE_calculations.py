from numpy import asarray, random, count_nonzero
from PIL import Image, ImageDraw, ImageFont
from typing import Literal, Optional


class prediction:
    type: Literal["Pixel"]
    words: []
    prediction: []
    source: Optional[str]
    font: Optional[str]
    height_image: Optional[int]
    width_image: Optional[int]
    letters: Optional[str]
    sequences: Optional[str]
    threshold_for_neuronal_threshold: Optional[float]
    mode: Literal["binary", "mean"]
    version: Literal["gagl_2020", "fu_2024"]

    def __init__(self, type="Pixel"):
        self.type = type
        self.words = []
        self.prediction = []


class pe_class:
    pe: []
    pe_N: Optional[int]

    def __init__(self, type="Pixel"):
        self.pe = []


def get_prediction_img(knowledge_list: object,
                       font_path: object = "fonts/Courier_Prime/CourierPrime-Bold.ttf",
                       height_image: object = 40,
                       width_image: object = 200,
                       noise_amount: object = 0,
                       threshold: object = 0.5,
                       mode: object = "mean",
                       version: object = "fu_2024") -> object:
    pred = prediction()
    pred.words = knowledge_list
    pred.font = font_path
    pred.height_image = height_image
    pred.width_image = width_image
    pred.threshold_for_neuronal_threshold = threshold
    pred.mode = mode
    pred.version = version

    if noise_amount == 0:
        knowledge_list_array = [
            get_word_image_as_array(word, font_path=font_path, height_image=height_image, width_image=width_image) for
            word
            in knowledge_list]
        if len(knowledge_list) == 0:
            pred.prediction_wo_threshold=0
        else:
            pred.prediction_wo_threshold = sum(knowledge_list_array) / len(knowledge_list)
        if mode == "binary":
            pred.prediction = pred.prediction_wo_threshold
            pred.prediction[pred.prediction_wo_threshold < (255 * (1 - threshold))] = 0
            pred.prediction[pred.prediction_wo_threshold >= (255 * (1 - threshold))] = 255
        else:
            pred.prediction = pred.prediction_wo_threshold

    elif 0 < noise_amount <= 1:
        knowledge_list_array = [
            get_word_image_as_array_add_noise(word, font_path=font_path, height_image=height_image,
                                              width_image=width_image, noise_amount=noise_amount) for
            word
            in knowledge_list]
        pred.prediction = sum(knowledge_list_array) / len(knowledge_list)
        if mode == "binary":
            pred.prediction[pred.prediction_wo_threshold < (255 * (1 - threshold))] = 0
            pred.prediction[pred.prediction_wo_threshold >= (255 * (1 - threshold))] = 255

    else:
        print
        'Noise amount value must be between 0 and +1.'

    return pred


def get_word_image_as_array(word, font_path, height_image, width_image):
    word = str(word)
    if len(word) < 21:  # come back here to automatize
        fnt = ImageFont.truetype(font_path, 25)
        img = Image.new('L', (width_image, height_image), color=255)
        d = ImageDraw.Draw(img)
        d.text((0, 5), word, fill=0, font=fnt)
        arr_num = asarray(img).astype("uint")
        arr_num[arr_num < 127] = 0
        arr_num[arr_num >= 127] = 255
        return arr_num

    else:
        print("Word has more than 20 letters. Only words with fewer letters can be processed.")


def get_word_image_as_array_add_noise(word: object, font_path: object, height_image: object, width_image: object,
                                      noise_amount: object) -> object:
    word = str(word)
    if len(word) < 21:
        fnt = ImageFont.truetype(font_path, 40)
        img = Image.new('L', (width_image, height_image), color=255)
        d = ImageDraw.Draw(img)
        d.text((20, 2), word, fill=0, font=fnt)
        arr_w_pl_noise = asarray(img).astype("uint") + get_noise_image_as_array(height_image,
                                                                                width_image) * noise_amount
        arr_w_pl_noise[arr_w_pl_noise > 255] = 255
        return arr_w_pl_noise

    else:
        print("Word has more than 20 letters. Only words with fewer letters can be processed.")


def get_noise_image_as_array(height_image, width_image):
    return random.random((height_image, width_image)) * 255


def get_prediction_error_array(word: object, prediction: object) -> object:
    if prediction.type == "Pixel":
        word_array = get_word_image_as_array(word, font_path=prediction.font, height_image=prediction.height_image,
                                             width_image=prediction.width_image)
        if (prediction.version == "gagl_2020"):
            pe = abs(word_array - prediction.prediction_wo_threshold)
        elif (prediction.version == "fu_2024"):
            if (prediction.mode == "binary"):
                pe = abs(word_array - prediction.prediction)
            elif (prediction.mode == "mean"):
                pe = abs(word_array - prediction.prediction_wo_threshold)

            pe[pe < (255 * prediction.threshold_for_neuronal_threshold)] = 0
            pe[pe >= (255 * prediction.threshold_for_neuronal_threshold)] = 255

        pe_elements = count_nonzero(pe != 0)
        pe_info = pe_class()
        pe_info.pe = pe
        pe_info.pe_N = pe_elements

        return pe_info

    else:
        print("No prediction specified, please specify a prediction first")


def get_prediction_error_sum(word: object, prediction: object) -> object:
    pe_dict = get_prediction_error_array(word, prediction)
    if prediction.version == "gagl_2020":
        if pe_dict.pe_N == 0:
            pe = 0
        else:
            pe = sum(sum(pe_dict.pe)) / 255 #/ pe_dict.pe_N
    elif prediction.version == "fu_2024":
        pe = pe_dict.pe_N

    return pe


def show_image(word_image_array):
    return Image.fromarray(word_image_array).show()


def wrapper_multi_parameter_modelling_oPE(string, prediction_img):
    pe = get_prediction_error_sum(word=string, prediction=prediction_img)
    return round(pe, ndigits=5)
