import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import google.generativeai as genai
import time

# --- 1. SETUP & LLM CONFIG ---
API_KEY = st.secrets["GEMINI_API_KEY"]  # keep secret in Streamlit Cloud
genai.configure(api_key=API_KEY)
llm = genai.GenerativeModel('gemini-3.1-flash-lite')

# --- 2. LOAD TRAINED VISION MODEL ---
@st.cache_resource
def load_trained_model():
    model = models.mobilenet_v2(weights=None)
    num_classes = 15
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('crop_disease_model.pth', map_location='cpu'))
    model.eval()
    return model

# --- 3. UI CONFIG ---
st.set_page_config(page_title="🌾 Tamil Nadu Smart Farmer AI", page_icon="🌱", layout="wide")
st.title("🌾 Tamil Nadu Smart Farmer AI")
st.markdown("### Diagnosis • Advisory • Chatbot")

# --- Sidebar Language Toggle ---
st.sidebar.title("⚙️ Settings")
lang_choice = st.sidebar.radio(
    "Choose advisory/chat language:",
    ["English", "Tamil", "Bilingual"],
    index=2  # default to Bilingual
)

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# --- 4. FILE UPLOAD ---
uploaded_file = st.file_uploader("📂 Upload a diseased leaf photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Leaf", width=300)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0)

    model = load_trained_model()
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
        diagnosis = CLASS_NAMES[pred[0].item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][pred[0]].item()

    # Diagnosis card
    st.markdown("## 🩺 Diagnosis")
    st.success(f"**Disease Identified:** {diagnosis}")
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.2f}")

    # --- 5. ADVISORY GENERATION BASED ON LANGUAGE CHOICE ---
    if "chat" not in st.session_state or st.session_state.get("last_diagnosis") != diagnosis:
        st.session_state.chat = llm.start_chat(history=[])
        st.session_state.last_diagnosis = diagnosis

        if lang_choice == "English":
            initial_prompt = (
                f"The identified crop disease is {diagnosis}. "
                f"Provide a concise treatment advisory in English only."
            )
        elif lang_choice == "Tamil":
            initial_prompt = (
                f"The identified crop disease is {diagnosis}. "
                f"Provide a concise treatment advisory in Tamil script only."
            )
        else:  # Bilingual
            initial_prompt = (
                f"The identified crop disease is {diagnosis}. "
                f"Provide a concise treatment advisory in English. "
                f"Then also provide the same advisory translated into Tamil script."
            )

        with st.spinner("Generating Advisory..."):
            start_time = time.time()
            try:
                response = st.session_state.chat.send_message(
                    initial_prompt,
                    generation_config={"max_output_tokens": 512}
                )
                elapsed = time.time() - start_time
                st.session_state.initial_advice = response.text
                st.caption(f"⏱️ Advisory generated in {elapsed:.2f} seconds")
            except Exception:
                elapsed = time.time() - start_time
                st.error(f"⚠️ Advisory service timed out after {elapsed:.2f} seconds.")
                st.session_state.initial_advice = ""

    st.markdown("## 📋 Advisory")
    st.info(st.session_state.initial_advice)

    # --- 6. CHATBOT LOOP WITH LANGUAGE CHOICE + TIMING ---
    st.markdown("## 💬 Farmer Support Chat")

    for msg in st.session_state.chat.history:
        if msg.role == "model":
            st.chat_message("assistant").write(msg.parts[0].text)
        else:
            if "The identified crop disease is" not in msg.parts[0].text:
                st.chat_message("user").write(msg.parts[0].text)

    if user_msg := st.chat_input("Ask a follow-up question..."):
        st.chat_message("user").write(user_msg)

        # Apply sidebar language choice
        if lang_choice == "English":
            lang_instruction = "Respond in English."
        elif lang_choice == "Tamil":
            lang_instruction = "Respond clearly in Tamil script."
        else:
            lang_instruction = "Respond in English and also provide Tamil translation."

        start_time = time.time()
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat.send_message(
                    user_msg + " " + lang_instruction,
                    generation_config={"max_output_tokens": 256}
                )
                elapsed = time.time() - start_time
                st.chat_message("assistant").write(response.text)
                st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")
            except Exception:
                elapsed = time.time() - start_time
                st.warning(f"⚠️ Response timed out after {elapsed:.2f} seconds. Retrying...")
                try:
                    response = st.session_state.chat.send_message(
                        user_msg + " " + lang_instruction,
                        generation_config={"max_output_tokens": 256}
                    )
                    elapsed = time.time() - start_time
                    st.chat_message("assistant").write(response.text)
                    st.caption(f"⏱️ Response time (retry): {elapsed:.2f} seconds")
                except Exception:
                    st.error("⚠️ Response timed out again. Please try later.")

