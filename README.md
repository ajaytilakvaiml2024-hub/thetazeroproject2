# 🌾 Tamil Nadu Smart Farmer AI
**Advanced Crop Disease Diagnosis & Multilingual Advisory System**

## 📖 Overview
This project is an AI-powered solution designed to assist farmers in Tamil Nadu with instant crop disease identification and treatment. By combining **Computer Vision** (Deep Learning) with **Generative AI** (LLMs), the system provides a seamless experience: from uploading a photo of an infected leaf to receiving a comprehensive treatment plan in **Tamil**.

---

## 🚀 Features
* **Precision Diagnosis:** Uses a custom-trained **MobileNetV2** model to identify 15 different crop disease classes (Tomato, Potato, Pepper) with **94.6% accuracy**.
* **Localized Advisory:** Automatically generates structured treatment plans (Symptoms, Organic and Chemical methods) in **Tamil script**.
* **Stateful Support Chat:** Features a conversational interface that remembers the current diagnosis, allowing farmers to ask follow-up questions without re-uploading images.
* **Cloud Ready:** Fully deployed and accessible via web browser through Streamlit.

---

## 🛠️ Technical Stack
* **Deep Learning Framework:** PyTorch & Torchvision
* **LLM Integration:** Google Gemini 2.5 Flash (via Generative AI SDK)
* **Frontend UI:** Streamlit
* **Programming Language:** Python 3.12+
* **Architecture:** Transfer Learning on MobileNetV2

---

## 📊 Model Performance
The vision model was trained on the PlantVillage dataset using transfer learning. 
* **Validation Accuracy:** 94.6%
* **Optimization:** Adam Optimizer with CrossEntropyLoss.
* **Input Size:** 224x224 pixels.
* **Normalization:** ImageNet standards ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).

---

## 📦 Local Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **API Configuration:**
    Set your Gemini API Key in `app.py` or as a Streamlit Secret:
    ```python
    API_KEY = "YOUR_API_KEY_HERE"
    ```

4.  **Run the App:**
    ```bash
    python -m streamlit run app.py
    ```

---

## 📂 Project Structure
* `app.py` - The core Streamlit application script.
* `crop_disease_model.pth` - Trained PyTorch model weights (State Dict).
* `requirements.txt` - List of required Python libraries for deployment.
* `README.md` - Project documentation and setup guide.

---

## 👨‍💻 Author
**Ajay Tilak V**
*Full-Stack Software Developer | AI Specialist*

---

## 🛡️ License & Disclaimer
This project is for educational and internship purposes. Always consult with a local agricultural officer before applying chemical treatments.

## 📷 Images 

<img width="1920" height="1080" alt="🌾 Tamil Nadu Smart Farmer AI - Google Chrome 5_12_2026 11_39_51 PM" src="https://github.com/user-attachments/assets/773a41d8-3bfb-42af-8ffb-b080cff13c7c" />

<img width="1920" height="1080" alt="🌾 Tamil Nadu Smart Farmer AI - Google Chrome 5_12_2026 11_40_19 PM" src="https://github.com/user-attachments/assets/00c873a4-6971-4809-b3f3-c83c3be91e02" />

<img width="1920" height="1080" alt="🌾 Tamil Nadu Smart Farmer AI - Google Chrome 5_12_2026 11_40_24 PM" src="https://github.com/user-attachments/assets/b776245c-d988-4d57-b709-04e339bedb85" />



