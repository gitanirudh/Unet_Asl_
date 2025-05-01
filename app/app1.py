import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import UNetAutoencoder

# Load model
model = UNetAutoencoder(num_classes=24)  # Set to 24 if your dataset has 24 labels (A-Y excluding J & Z)
state_dict = torch.load("app/UNet_weights.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Label mapping (adjust this if your training used a different order)
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # Include normalization if you used it in training:
    # transforms.Normalize((0.5,), (0.5,))
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

    predicted_char = label_map.get(predicted_class, "Unknown")

    st.success(f'Predicted ASL Letter: {predicted_char}')
