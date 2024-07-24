import streamlit as st
import cv2 as cv
import numpy as np



# Load pre-trained models with appropriate paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)#this values are for RGB channels and it is taken from data set (caffe) which is here used
ageList = ['(0-5)', '(6-10)', '(11-15)', '(16-22)', '(23-29)', '(30-40)', '(40-55)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
ageNet = cv.dnn.readNetFromCaffe(ageProto, ageModel)
genderNet = cv.dnn.readNetFromCaffe(genderProto, genderModel)
faceNet = cv.dnn.readNet(faceModel, faceProto)

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    #for face frame
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    #here is 4*4 matrix array in this matrix the 3rd column is perdition of the face and generate the output in mtrix value form
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes

def detect_age_gender(image):
    frame = cv.imdecode(np.frombuffer(image.read(), np.uint8), cv.IMREAD_COLOR)
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        st.warning("No face detected")
        return frame
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - 20):min(bbox[3] + 20, frame.shape[0] - 1),
                     max(0, bbox[0] - 20):min(bbox[2] + 20, frame.shape[1] - 1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        label = f"{gender}, {age}"
        cv.putText(frameFace, label, (bbox[0] - 5, bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    return frameFace

# Streamlit application
st.set_page_config(page_title="Age and Gender Detection", page_icon=":guardsman:", layout="wide")

st.title("Age and Gender Detection :guardsman:")
st.write("""
    Upload an image to detect age and gender using deep learning models.
    This app uses pre-trained models to accurately predict the age and gender of individuals in an image.
""")

st.sidebar.title("Instructions")
st.sidebar.info("""
    1. Upload an image using the file uploader below.
    2. Wait for the image to be processed.
    3. View the detected age and gender directly on the image.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        processed_image = detect_age_gender(uploaded_file)
        st.image(processed_image, channels="BGR", caption="Processed Image")

st.sidebar.title("About")
st.sidebar.info("""
    This application uses OpenCV's deep learning module to detect faces, and predict age and gender.
    The models used are:
    - Face Detection: OpenCV DNN model
    - Age Prediction: Caffe model
    - Gender Prediction: Caffe model
""")
