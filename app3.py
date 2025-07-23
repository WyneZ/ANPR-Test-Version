import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import write_csv
import uuid
import os
from streamlit_webrtc import webrtc_streamer
import av

folder_path = "./detected_license_imgs/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=False)

vehicles = [2]

header = st.container()
body = st.container()

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_to_an = img.copy()
        img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
        license_detections = license_plate_detector(img_to_an)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
            
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)
    
    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]
    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            text = result[1]
            text = text.upper()
            scores += result[2]
            plate.append(text)
    
    if len(plate) != 0: 
        return " ".join(plate), scores/len(plate)
    else:
        return " ".join(plate), 0

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else:
            xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
            car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None:
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}
                
                results[license_numbers][license_numbers] = {
                    'car': {
                        'bbox': [xcar1, ycar1, xcar2, ycar2], 
                        'car_score': car_score
                    },
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                } 
                license_numbers += 1
          
        write_csv(results, f"./csv_detections/detection_results.csv")
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    else: 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]

with header:
    st.title("Car License Plate Detection")

with body:
    tab1, tab2, tab3 = st.tabs(["Upload an Image", "Take a Photo", "Live Detection"])
    
    with tab1:
        uploaded_img = st.file_uploader("Upload a Car Image:", type=["png", "jpg", "jpeg"], key="uploader")
        
        if uploaded_img is not None:
            image = np.array(Image.open(uploaded_img))
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Apply Detection", key="detect_upload"):
                results = model_prediction(image)
                
                if len(results) == 3:
                    prediction, texts, license_plate_crop = results[0], results[1], results[2]
                    texts = [i for i in texts if i is not None]
                    
                    st.header("Detection Result")
                    st.image(prediction)
                    
                    st.header("License Cropped")
                    st.image(license_plate_crop[0], width=350)
                    
                    st.success(f"License Number: {texts[0]}")
                    
                    df = pd.read_csv(f"./csv_detections/detection_results.csv")
                    st.dataframe(df)
                else:
                    st.header("Detection Result")
                    st.image(results[0])
    
    with tab2:
        camera_img = st.camera_input("Take a Photo for License Plate Detection:", key="camera")

        if camera_img is not None:
            image = np.array(Image.open(camera_img))
            st.image(image, caption="Captured Image", width=400)

            if st.button("Apply Detection", key="detect_camera"):
                with st.spinner("Processing image..."):
                    results = model_prediction(image)

                st.header("Detection Result")

                if len(results) == 3:  # License plate detected
                    prediction, texts, license_plate_crop = results[0], results[1], results[2]
                    texts = [i for i in texts if i is not None]

                    st.image(prediction)

                    if len(license_plate_crop) > 0 and len(texts) > 0:  # Additional check
                        st.header("License Plate Cropped")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(license_plate_crop[0], width=300)
                            st.success(f"License Number: {texts[0]}")

                        try:
                            df = pd.read_csv(f"./csv_detections/detection_results.csv")
                            st.dataframe(df)
                        except FileNotFoundError:
                            st.warning("Could not load detection results file")
                    else:
                        st.warning("License plate was detected but couldn't extract text")
                else:  # No license plate detected
                    st.image(results[0])
                    st.warning("No license plate detected in the image")
    
    with tab3:
        st.write("Live License Plate Detection")
        webrtc_streamer(key="live", video_processor_factory=VideoProcessor)