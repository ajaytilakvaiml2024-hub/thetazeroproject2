# 🌾 Tamil Nadu Smart Farmer AI

An AI-powered Streamlit application that helps farmers diagnose crop diseases from leaf images and receive treatment advisories in **English and Tamil**. The app also includes a chatbot for farmer support.

---

## ✨ Features
- 📸 **Leaf Image Upload**: Farmers upload a photo of a diseased leaf.
- 🧠 **CNN Disease Prediction**: MobileNetV2 model predicts the crop disease.
- 📊 **Confidence Score**: Shows how confident the model is about its prediction.
- 📋 **Advisory Generation**: Gemini AI provides treatment guidance in English, Tamil, or both.
- 💬 **Farmer Chatbot**: Farmers can ask follow-up questions in English or Tamil.
- ⚙️ **Language Toggle**: Sidebar option to choose English, Tamil, or bilingual output.

---

## 🛠️ Tech Stack
- [Streamlit](ca://s?q=Learn_about_Streamlit) – UI framework
- [PyTorch](ca://s?q=Learn_about_PyTorch) – Deep learning model
- [Torchvision](ca://s?q=Learn_about_Torchvision) – Pretrained MobileNetV2
- [Google Generative AI](ca://s?q=Learn_about_Google_Generative_AI) – Gemini model for advisory and chatbot
- [PIL](ca://s?q=Learn_about_PIL) – Image processing

---

## 📂 Project Structure
Crop_AI_Project/
│── app.py              # Main Streamlit app
│── crop_disease_model.pth  # Trained CNN weights
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation

Code

---

## 🚀 Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Crop_AI_Project.git
   cd Crop_AI_Project
Install dependencies:

bash
pip install -r requirements.txt
Add your Gemini API key:

In app.py, replace:

python
API_KEY = ""  # replace with st.secrets["GEMINI_API_KEY"]
Or set it in Streamlit Cloud secrets.

Run the app:

bash
streamlit run app.py
🌱 Usage
Upload a diseased leaf photo (.jpg, .png, .jpeg).

The CNN model predicts the disease and shows confidence.

Gemini generates treatment advisory in your chosen language.

Use the chatbot to ask follow-up questions in English or Tamil.

📌 Notes
Default language is Bilingual (English + Tamil).

Advisory and chatbot respect the sidebar language toggle.

If Gemini times out, the app shows a friendly error message and retries once.
