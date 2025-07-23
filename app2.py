


import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import write_csv
import uuid
import os

# Mobile optimization settings
st.set_page_config(layout="wide")

# Mobile-friendly CSS
st.markdown("""
<style>
    .stButton>button {
        min-height: 3rem;
        min-width: 100%;
        padding: 12px !important;
    }
    .stImage>img {
        max-width: 100% !important;
        height: auto !important;
    }
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    .stSpinner>div {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize paths and models
folder_path = "./detected_license_imgs/"
os.makedirs(folder_path, exist_ok=True)
os.makedirs("./csv_detections", exist_ok=True)

@st.cache_resource
def load_models():
    return {
        "coco": YOLO("./models/yolov8n.pt"),
        "license": YOLO("./models/license_plate_detector.pt"),
        "reader": easyocr.Reader(['en'], gpu=False)
    }

models = load_models()
vehicles = [2]  # COCO class IDs for cars
threshold = 0.15

def correct_orientation(image):
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass
    return image

def read_license_plate(license_plate_crop, img):
    detections = models["reader"].readtext(license_plate_crop)
    
    if not detections:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate = []
    scores = 0
    
    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length * height / rectangle_size > 0.17:
            text = result[1].upper()
            scores += result[2]
            plate.append(text)
    
    if plate:
        return " ".join(plate), scores/len(plate)
    return None, 0

def model_prediction(img):
    results = {}
    licenses_texts = []
    license_plate_crops = []
    
    # Convert and correct image orientation
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Vehicle detection
    vehicle_detections = models["coco"](img)[0]
    license_detections = models["license"](img)[0]
    
    # Process vehicle detections
    if vehicle_detections.boxes.cls.tolist():
        for detection in vehicle_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # Process license plates
    if license_detections.boxes.cls.tolist():
        for i, license_plate in enumerate(license_detections.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = license_plate
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Crop license plate
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Save cropped image
            img_name = f'{uuid.uuid1()}.jpg'
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
            
            # Read license plate text
            license_plate_text, text_score = read_license_plate(license_plate_crop_gray, img)
            
            if license_plate_text:
                licenses_texts.append(license_plate_text)
                license_plate_crops.append(license_plate_crop)
                
                results[i] = {
                    'car': {'bbox': [x1, y1, x2, y2], 'score': score},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': text_score
                    }
                }
                
                # Draw text on image
                cv2.putText(img, license_plate_text, 
                           (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    write_csv(results, "./csv_detections/detection_results.csv")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if licenses_texts:
        return img, licenses_texts, license_plate_crops
    return img,

# Streamlit UI
st.title("📱 Mobile License Plate Recognition")
st.markdown("Upload an image of a vehicle to detect its license plate")

# Image uploader
img = st.file_uploader(
    "Upload a car image (JPG, PNG)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Take a photo with your camera and upload it here"
)

if img:
    img = Image.open(img)
    img = correct_orientation(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect License Plate", type="primary"):
        with st.spinner("Analyzing image..."):
            result = model_prediction(img)
            
            if len(result) == 3:
                processed_img, texts, crops = result
                
                st.success("Detection Complete!")
                st.image(processed_img, caption="Detection Results", use_column_width=True)
                
                st.subheader("License Plate Information")
                for i, (text, crop) in enumerate(zip(texts, crops)):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(crop, caption=f"Plate {i+1}")
                    with col2:
                        print("This is text", text)

                        reader = easyocr.Reader(['en'])
                        results = reader.readtext(crop)  # or image array

                        for (bbox, text, confidence) in results:
                            st.markdown(f"""
                                **Detected Text:** `{text}`  
                                **Confidence:** {confidence:.2f}%
                            """)

                        # st.markdown(f"""
                        #     **Detected Text:**  
                        #     `{text}`  
                        #     **Confidence:**  
                            
                        # """)
                
                try:
                    df = pd.read_csv("./csv_detections/detection_results.csv")
                    st.subheader("Detection Data")
                    st.dataframe(df)
                except Exception as e:
                    st.warning(f"Couldn't load detection data: {str(e)}")
            else:
                st.warning("No license plates were detected in the image")
                st.image(result[0], caption="Processed Image", use_column_width=True)

st.markdown("---")
st.markdown("""
### Usage Instructions:
1. Take a clear photo of a vehicle (parked cars work best)
2. Upload the image using the button above
3. Click "Detect License Plate" to analyze
4. View the detected plates and information

Tips for better results:
- Capture the vehicle straight-on if possible
- Ensure good lighting conditions
- Avoid blurry or angled photos
""")
 