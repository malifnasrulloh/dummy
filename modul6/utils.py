import base64
from io import BytesIO
import re
import nltk
import tensorflow as tf
import gdown
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import cv2
import numpy as np
import streamlit as st
from zipfile import ZipFile
from pytorch_tabnet.tab_model import TabNetClassifier
from transformers import BertTokenizer

nltk.download('punkt')
nltk.download('wordnet') 
nltk.download('stopwords')

stop_words = stopwords.words('english')
wnl = WordNetLemmatizer()

im_model = "model/image.h5"  # rock paper scissor (224, 224, 3)
text_model = "model/text"  # sentiment (english)
tabular_model = "model/tabular.zip"  # income (14-dim input)


@st.cache_resource
def download_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.isdir(text_model):
        gdown.download(
            "https://drive.google.com/file/d/1-0l8aK1fLZQb1VRaOMuZrPvfHDL2uNhy/view?usp=sharing",
            fuzzy=True,
            output="model/",
        )
        with ZipFile(text_model + ".zip", "r") as f:
            f.extractall(path="model")
        f.close()

    if not os.path.isfile(tabular_model):
        gdown.download(
            "https://drive.google.com/file/d/1m2fKNGVKb5FqA5cv2NhUWM6brCjJvOJ1/view?usp=sharing",
            fuzzy=True,
            output="model/",
        )

    if not os.path.isfile(im_model):
        gdown.download(
            "https://drive.google.com/file/d/1-1m6RTjU6Fkz9tfx9UW5QVH7VJXcbcl-/view?usp=sharing",
            fuzzy=True,
            output="model/",
        )


def text_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


@st.cache_resource
def image_load_model():
    return tf.keras.models.load_model(im_model)


@st.cache_resource
def text_load_model():
    model = tf.saved_model.load(text_model)
    return model.signatures["serving_default"]


@st.cache_resource
def tabular_load_model():
    model = TabNetClassifier()
    model.load_model(tabular_model)
    return model


def compress_image(image, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode(".jpg", image, encode_param)
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return decoded_img


def byte_to_im(byte):
    image_stream = BytesIO(byte.getvalue())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    im = cv2.imdecode(file_bytes, 1)[:, :, ::-1]

    _, buffer = cv2.imencode(".png", im[:, :, ::-1])
    image_bytes = buffer.tobytes()

    return im, base64.b64encode(image_bytes).decode("utf-8")


def preprocessing_text(text: str):
    url_pattern = r"http[s]?://\S+|www\.\S+"
    mention_pattern = r"@\w+"
    hashtag_pattern = r"#\w+"
    symbol_pattern = r"[^A-Za-z0-9\s]+"

    combined_pattern = (
        f"({url_pattern})|({mention_pattern})|({hashtag_pattern})|({symbol_pattern})"
    )
    text = re.sub(combined_pattern, "", str(text)).split(" ")

    text = list(filter(lambda x: x.isalpha(), text))
    text = list(filter(lambda x: x.lower() not in stop_words, text))
    text = list(map(lambda x: wnl.lemmatize(x), text))

    return " ".join(text)


def image_inference(model, image):
    label = ["paper", "rock", "scissor"]
    image = cv2.resize(image, (224, 224))
    image = compress_image(image, 10)
    pred = np.argmax(model.predict(np.array([image])), axis=1)[0]
    return label[pred]


def text_inference(model, tokenizer, text):
    text = text.split("\n")
    prep_text = list(map(lambda x: preprocessing_text(x), text))
    tokenize_text = tokenizer(prep_text, max_length=128, padding=True)

    result = model(**tokenize_text)["logits"]
    result = np.argmax(result, axis=1)
    return text, result
