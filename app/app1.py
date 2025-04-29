import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import UNetAutoencoder

# Load model
model = UNetAutoencoder(num_classes=36)
state_dict = torch.load("UNet_weights.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("ASL Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        _, logits = model(img_tensor)
        predicted_class = logits.argmax(dim=1).item()

    st.success(f'Predicted Class: {predicted_class}')
