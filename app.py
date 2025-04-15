import os
from flask import Flask, request, render_template, send_from_directory
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def preprocess_image(image_path, target_size=(320, 320)):
    image = Image.open(image_path).convert("RGBA")
    image = image.resize(target_size, Image.BILINEAR)
    image_np = np.array(image)
    image_np = image_np / 255.0
    image_np = np.transpose(image_np, (2, 0, 1)).astype(np.float32)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def postprocess_output(output, original_image, target_size=(320, 320)):
    output_mask = output[0][0]
    mask_resized = cv2.resize(output_mask, (original_image.width, original_image.height))
    mask_resized = (mask_resized > 0.5).astype(np.uint8)
    original_np = np.array(original_image)
    result = original_np.copy()
    result[:, :, 3] = mask_resized * 255
    return Image.fromarray(result)

def remove_background_with_u2net(image_path, model_path="u2net.onnx"):
    ort_session = ort.InferenceSession(model_path)
    input_tensor = preprocess_image(image_path)
    outputs = ort_session.run(None, {"input": input_tensor})
    original_image = Image.open(image_path)
    result_image = postprocess_output(outputs, original_image)
    result_image_path = os.path.join(RESULT_FOLDER, "output_image.png")
    result_image.save(result_image_path)
    return result_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the image with the U2Net model
    result_image_path = remove_background_with_u2net(file_path)

    # Send the processed image back to the user
    return send_from_directory(RESULT_FOLDER, 'output_image.png')

if __name__ == "__main__":
    app.run(debug=True)
