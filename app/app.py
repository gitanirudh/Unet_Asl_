from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from .model import UNetAutoencoder
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io

app = FastAPI()

# Load model
model = UNetAutoencoder(num_classes=36)
state_dict = torch.load("app/UNet_weights.pth", map_location=torch.device("cpu"))
unet_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("unet.enc1")}
missing, unexpected = model.load_state_dict(unet_state_dict, strict=False)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')  # Convert to grayscale
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_tensor = transform(image).unsqueeze(0)

    if image_tensor.shape != (1, 1, 28, 28):
        raise HTTPException(status_code=400, detail=f"Unexpected image tensor shape: {image_tensor.shape}")

    try:
        with torch.no_grad():
            _, logits = model(image_tensor)
            predicted_class = logits.argmax(dim=1).item()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return JSONResponse(content={"predicted_class": predicted_class})
