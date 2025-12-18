import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Config
st.set_page_config(
    page_title="Banana Ripeness Detector",
    page_icon="🍌",
    layout="centered"
)

# CSS
st.markdown("""
    <style>
    /* Mengubah font global ke Sans Serif */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Tombol Prediksi Custom */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00B4DB, #0083B0);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        color: white;
    }

    /* Card container untuk hasil */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="banana_model_final.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

interpreter = load_tflite_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Preproses dan Logic
class_names = ['Overripe', 'Ripe', 'Rotten', 'Unripe']

def preprocess_image(image):
    IMG_SIZE = 160
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("🍌 Predicting Banana Ripeness")
st.markdown("""
    Selamat datang di aplikasi deteksi kematangan pisang. 
""")
st.write("")

# File Uplosd dan Display
with st.container():
    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG, PNG, JPEG)", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
    
    with col2:
        st.info("Gambar berhasil terbaca! Klik tombol di bawah untuk mulai menganalisis.")
        predict_btn = st.button("Mulai Prediksi")

    if predict_btn:
        with st.spinner("🤖 AI sedang berpikir..."):
            img_array = preprocess_image(image)
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            label_index = np.argmax(prediction)
            confidence = np.max(prediction)
            label = class_names[label_index]

            status_icons = {
                'Unripe': '🟢',
                'Ripe': '🟡',
                'Overripe': '🟠',
                'Rotten': '🟤'
            }
            icon = status_icons.get(label, "🍌")

            st.markdown("---")
            st.markdown(f"### {icon} Hasil Analisis")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="Status Kematangan", value=label)
            
            with res_col2:
                st.metric(label="Tingkat Keyakinan", value=f"{confidence * 100:.1f}%")

            if label == 'Ripe':
                st.success("Saran: Pisang dalam kondisi prima untuk dikonsumsi!")
            elif label == 'Unripe':
                st.info("Saran: Tunggu beberapa hari lagi agar pisang manis sempurna.")
            else:
                st.warning("Saran: Periksa kembali tekstur pisang sebelum digunakan.")

else:
    # Tampilan Dashboard saat kosong
    st.write("---")
    st.markdown("""
        <div style="text-align: center; color: #6c757d; padding: 50px;">
            <p style="font-size: 20px;">Silakan upload gambar untuk melihat hasil</p>
        </div>
    """, unsafe_allow_html=True)