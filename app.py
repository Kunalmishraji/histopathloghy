
pip install -r requirements.txt

import streamlit as st
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gdown
from torchvision import transforms
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt

# ---------- LABELS AND UTILITIES ----------
labels = ["MC", "EC", "HGSC", "LGSC", "CC"]

def classify_ca125(ca125_pred):
    if ca125_pred < 35:
        return "<35 U/mL", "Normal or Mild"
    elif 35 <= ca125_pred <= 150:
        return "35–150 U/mL", "Mild to Moderate"
    elif 150 < ca125_pred <= 300:
        return "150–300 U/mL", "Moderate"
    elif ca125_pred > 300:
        return "100–5000+ U/mL", "High to Extremely High"
    return "Ambiguous", "Unclassified"

# ---------- MODEL DEFINITION ----------
class MultiTaskViT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.vit = base_model
        self.class_head = nn.Linear(768, 5)
        self.reg_head = nn.Linear(768, 1)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        cls_token = outputs.hidden_states[-1][:, 0, :]
        class_logits = self.class_head(cls_token)
        reg_output = self.reg_head(cls_token)
        return class_logits, reg_output

# ---------- IMAGE PREPROCESSING ----------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- MODEL LOADING FROM DRIVE ----------
file_id = "1Pk69Hl_m7es0VjMIOGLq_E5nVb9ll8IF"  # 🔁 <- Replace with your actual model file ID
gdrive_url = f"https://drive.google.com/drive/folders/1Pk69Hl_m7es0VjMIOGLq_E5nVb9ll8IF?usp=drive_link"
model_path = "histo_ViT.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(gdrive_url, model_path, quiet=False)

    base_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    model = MultiTaskViT(base_model)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ---------- STREAMLIT UI ----------
st.title("🧬 Histopathology Cancer Type & CA-125 Predictor")

# Patient name input
patient_name = st.text_input("👤 Enter Patient Name")

# Image uploader
uploaded_file = st.file_uploader("📷 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

# If both inputs provided
if patient_name and uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = preprocess(image).unsqueeze(0)

        # Display input image
        st.image(image, caption=f"Uploaded by {patient_name}", use_column_width=True)

        # Load model and make prediction
        model = load_model()
        with torch.no_grad():
            class_logits, reg_output = model(img_tensor)
            predicted_idx = torch.argmax(class_logits, dim=1).item()
            predicted_label = labels[predicted_idx]
            ca125_pred = np.exp(reg_output.squeeze().cpu().numpy())
            range_str, severity = classify_ca125(ca125_pred)

        # Prediction summary
        st.markdown("### 🧪 Prediction Result")
        st.success(f"""
        **Patient Name**: {patient_name}  
        **Cancer Type**: {predicted_label} (Class ID: {predicted_idx})  
        **Predicted CA-125**: {ca125_pred:.2f} U/mL  
        **Range**: {range_str}  
        **Severity Class**: {severity}
        """)
    except Exception as e:
        st.error(f"❌ Error processing image or model: {e}")
else:
    st.info("👈 Please enter patient name and upload an image to continue.")
