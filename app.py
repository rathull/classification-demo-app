import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import requests
from io import BytesIO

label_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]

# Load the pretrained model
@st.cache_resource()
def load_model():
    model = resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 43)
    model.load_state_dict(torch.load('resnet50_gtsrb.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to predict the class of an image
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

st.title("GTSRB Traffic Sign Classifier")

# Option to upload an image
uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    model = load_model()
    label = predict(image, model)
    st.write(f"Predicted Label: {label}")

# Option to capture an image using the device camera
st.write("Or capture an image using your device camera:")
camera_photo = st.camera_input("Take a picture")

if camera_photo is not None:
    image = Image.open(camera_photo)
    st.image(image, caption='Captured Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    model = load_model()
    label = predict(image, model)
    st.write(f"Predicted Label: {label_names[label]} [{label}]")
