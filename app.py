import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import easyocr
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load YOLO models
vehicle_model = YOLO("models/vehicle.pt")
plate_model = YOLO("models/license_plate.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Set the upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload and detection
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect vehicles
        vehicle_results = vehicle_model(filepath)[0]
        vehicle_boxes = vehicle_results.boxes.xyxy.cpu().numpy().astype(int)

        # Read image for further processing
        img = cv2.imread(filepath)

        # List to store OCR results
        ocr_results = []

        for idx, box in enumerate(vehicle_boxes):
            x1, y1, x2, y2 = box
            vehicle_crop = img[y1:y2, x1:x2]
            vehicle_crop_path = f"vehicle_crop_{idx}.jpg"
            cv2.imwrite(vehicle_crop_path, vehicle_crop)

            # Detect license plates within the vehicle crop
            plate_results = plate_model(vehicle_crop_path)[0]
            plate_boxes = plate_results.boxes.xyxy.cpu().numpy().astype(int)

            for p_idx, p_box in enumerate(plate_boxes):
                px1, py1, px2, py2 = p_box
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                plate_crop_path = f"plate_crop_{idx}_{p_idx}.jpg"
                cv2.imwrite(plate_crop_path, plate_crop)

                # Perform OCR on the license plate
                ocr_result = reader.readtext(plate_crop_path)
                ocr_results.append(ocr_result)

        return render_template('index.html', filename=filename, ocr_results=ocr_results)
    
    return redirect(url_for('index'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
