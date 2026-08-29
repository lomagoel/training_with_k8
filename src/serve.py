import io
import os
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from src.dataset import get_transforms
from src.model import get_model

app = FastAPI(title="CIFAR-10 Model Serving API")

# Global variables for model state
MODEL = None
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_checkpoint():
    global MODEL
    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"
    )
    path = Path(checkpoint_path)

    if not path.exists():
        print(f"Warning: Checkpoint path {checkpoint_path} not found.")
        return False

    try:
        model = get_model(architecture="resnet18", num_classes=10)
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        MODEL = model
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return False


@app.on_event("startup")
def startup_event():
    load_checkpoint()


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    if MODEL is None:
        # Re-attempt loading in case training finished recently
        success = load_checkpoint()
        if not success:
            raise HTTPException(
                status_code=503, detail="Model checkpoint not loaded"
            )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized",
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image"
        )

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        transform = get_transforms(train=False)
        tensor_image = transform(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = MODEL(tensor_image)
            probabilities = F.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probabilities, dim=0)

        probs_dict = {
            CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))
        }

        return JSONResponse(
            content={
                "prediction": CLASSES[pred_idx.item()],
                "confidence": float(conf),
                "probabilities": probs_dict,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")