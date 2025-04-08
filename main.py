from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import base64
import os

app = FastAPI()

# Load TFLite classification model
try:
    # Update the model path if necessary
    interpreter = tf.lite.Interpreter(model_path="ingredient_classifier.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Classification model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

# Ingredient class labels for your classifier model
class_labels = [
    'Bitter_m', 'Calamansi', 'Eggplant', 'Garlic', 'Ginger',
    'Okra', 'Onion', 'Pork', 'Potato', 'Squash', 'Tomato'
]

def preprocess_image(image: Image.Image):
    """
    Preprocess the image: resize to (224, 224) as expected by the model,
    convert to float32, expand dims, and normalize.
    """
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization
    return img_array

def predict_classification(uploaded_file: UploadFile):
    try:
        # Read the image from the uploaded file
        image_bytes = uploaded_file.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_image(image)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # Get 1D array of class probabilities

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[::-1][:3]
        result = []
        for idx in top_3_indices:
            label = class_labels[idx]
            confidence = predictions[idx] * 100
            result.append({
                "ingredient": label,
                "confidence": round(confidence, 2)
            })

        # Annotate the image with the top prediction
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception as e:
            # Fallback to default PIL font if TrueType font cannot be loaded
            font = ImageFont.load_default()

        top_label = class_labels[top_3_indices[0]]
        top_conf = predictions[top_3_indices[0]] * 100
        text = f"{top_label} {top_conf:.1f}%"
        draw.text((10, 10), text, fill="red", font=font)

        # Encode annotated image to Base64 string
        img_io = BytesIO()
        image.save(img_io, format="JPEG")
        img_io.seek(0)
        encoded_img = base64.b64encode(img_io.getvalue()).decode("utf-8")

        return {
            "predictions": result,
            "image_base64": encoded_img
        }

    except Exception as e:
        return {"error": str(e)}

# FastAPI POST endpoint for classification
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    return JSONResponse(content=predict_classification(file))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
