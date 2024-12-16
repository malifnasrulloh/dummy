import streamlit as st
import utils
from streamlit_card import card
import numpy as np

utils.download_model()
model = utils.image_load_model()

if "files" not in st.session_state:
    st.session_state["files"] = []
if "picture" not in st.session_state:
    st.session_state["picture"] = None
if "show_camera" not in st.session_state:
    st.session_state["show_camera"] = False


@st.dialog("Ambilll POTO")
def show_camera():
    picture = st.camera_input("Ambil Gambar")
    if picture:
        st.session_state["files"].append(picture)
        st.session_state["show_camera"] = False
        st.rerun()
        del picture

st.title("Image Classifier (RPS)")

col1, col2 = st.columns(2, vertical_alignment="center")
with col1:
    uploaded_files = st.file_uploader(
        "Masukkan Gambar",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.session_state["files"].extend(
            uploaded_files
        )  # Tambahkan file ke session_state

with col2:
    if st.button("Ambil Gambar dari Kamera"):
        st.session_state["show_camera"] = True  # Tampilkan kamera

# Jika tombol untuk kamera ditekan, jalankan fungsi kamera
if st.session_state["show_camera"]:
    show_camera()

# Menampilkan gambar dan hasil prediksi
if st.session_state["files"]:
    st.subheader("Hasil Prediksi")
    col1, col2 = st.columns(2)

    # Loop melalui file yang diunggah, menggunakan enumerasi untuk pengurutan
    for i, im in enumerate(st.session_state["files"]):
        filename = im.name

        # Proses gambar dan konversi ke base64
        processed_image, im_byte = utils.byte_to_im(im)

        # Dapatkan prediksi model
        prediction = utils.image_inference(model, processed_image)

        # Pilih kolom untuk menampilkan kartu secara bergantian
        column = col1 if i % 2 == 0 else col2

        # Tampilkan kartu pada kolom yang sesuai
        with column:
            card(
                image="data:image/png;base64," + im_byte,
                title=prediction,
                text="",
                styles={
                    "title": {
                        "position": "absolute",
                        "bottom": "10px",
                        "left": "50%",
                        "transform": "translateX(-50%)",
                        "text-align": "center",
                        "background": "rgba(0, 0, 0, 0.8)",
                        "border-radius": "10px",
                        "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.1)",
                        "padding": "10px",
                        "margin": "10px",
                        "font-size": "24px",
                        "font-weight": "bold",
                        "color": "#fff",
                    },
                    "card": {
                        "bacdkground": "rgba(255, 255, 255, 0.8)",
                        "border-radius": "10px",
                        "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.1)",
                        "padding": "0px",
                        "margin": "0px",
                    },
                    "filter": {
                        "background-color": "rgba(0, 0, 0, 0)"  # <- make the image not dimmed anymore
                    },
                },
            )
else:
    st.warning("Silakan unggah atau ambil gambar untuk diproses.")
