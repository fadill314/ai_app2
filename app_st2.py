import streamlit as st
from streamlit.elements.image import image_to_url
import io

st.title("EMC2 / Cloud AI Module")
st.header("App for testing Azure AI Service")

st.sidebar.image("https://seeklogo.com/images/E/ecole-hassania-des-travaux-publics-ehtp-logo-3D5770F217-seeklogo.com.png")
st.sidebar.header("Executive Master Cloud Computing")

app=st.sidebar.selectbox('Select type of Application', ["--- Choose application ---","Image Analysis", "Thumbnail Image", "Face Analysis", "OCR"])

import os
from PIL import Image, ImageDraw
import sys
import matplotlib.pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

ai_endpoint = st.secrets['AI_SERVICE_ENDPOINT']
ai_key = st.secrets['AI_SERVICE_KEY']

# Import namespaces
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Authenticate Azure AI Vision client
credential = CognitiveServicesCredentials(ai_key) 
cv_client = ComputerVisionClient(ai_endpoint, credential)

def AnalyzeImage(image_file):
    st.write('Analyzing', image_file.name)

    # Specify features to be retrieved
    features = [VisualFeatureTypes.description,
            VisualFeatureTypes.tags,
            VisualFeatureTypes.categories,
            VisualFeatureTypes.brands,
            VisualFeatureTypes.objects,
            VisualFeatureTypes.adult]
    
    # Get image analysis
    image_data = io.BytesIO(image_file.read())
    analysis = cv_client.analyze_image_in_stream(image_data , features)

    # Get image description
    for caption in analysis.description.captions:
        st.write("\n**Description:**")
        st.write("'{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

    # Get image tags
    if (len(analysis.tags) > 0):
        st.write("**Tags:**")
    for tag in analysis.tags:
        st.write(" -'{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))

    # Get image categories 
    # Get image categories
    if (len(analysis.categories) > 0):
        st.write("**Categories:**")
        landmarks = []
        for category in analysis.categories:
            # st.write the category
            st.write(" -'{}' (confidence: {:.2f}%)".format(category.name, category.score * 100))
            if category.detail:
                # Get landmarks in this category
                if category.detail.landmarks:
                    for landmark in category.detail.landmarks:
                        if landmark not in landmarks:
                            landmarks.append(landmark)

        # If there were landmarks, list them
        if len(landmarks) > 0:
            st.write("**Landmarks:**")
            for landmark in landmarks:
                st.write(" -'{}' (confidence: {:.2f}%)".format(landmark.name, landmark.confidence * 100))

    # Get brands in the image
    # Get brands in the image
    if (len(analysis.brands) > 0):
        st.write("**Brands:**")
        for brand in analysis.brands:
            st.write(" -'{}' (confidence: {:.2f}%)".format(brand.name, brand.confidence * 100))

    # Get objects in the image
    # Get objects in the image
    if len(analysis.objects) > 0:
        st.write("**Objects in image:**")

        # Prepare image for drawing
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)
        color = 'cyan'
        for detected_object in analysis.objects:
            # st.write object name
            st.write(" -{} (confidence: {:.2f}%)".format(detected_object.object_property, detected_object.confidence * 100))
            
            # Draw object bounding box
            r = detected_object.rectangle
            bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
            draw.rectangle(bounding_box, outline=color, width=3)
            plt.annotate(detected_object.object_property,(r.x, r.y), backgroundcolor=color)
        # Save annotated image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        st.write('  Results in image', image)

    # Get moderation ratings
    st.write("**moderation ratings:**")
    ratings = "\n -Adult: {} \n -Racy: {} \n -Gore: {}".format(analysis.adult.is_adult_content, analysis.adult.is_racy_content, analysis.adult.is_gory_content)
    st.write(ratings)
    
def GetThumbnail(image_file):
    st.write('Generating thumbnail')
    # Generate a thumbnail
    image_data = io.BytesIO(image_file.read())
    # Get thumbnail data
    thumbnail_stream = cv_client.generate_thumbnail_in_stream(100, 100, image_data, True)

    thumbnail_data = b"".join(thumbnail_stream)
    thumbnail_image = Image.open(io.BytesIO(thumbnail_data))
    st.write("Generated Thumbnail:")
    st.image(thumbnail_image, caption="Thumbnail")

   
    
if app == "Image Analysis" :
    st.subheader("Application : Image Analysis")
    # Get image
    image_file = st.file_uploader('Load image ',type=['png', 'jpg'])
    if image_file is not None:
        st.image(image_file)
        AnalyzeImage(image_file)
        
if app == "Thumbnail Image" :
    st.subheader("Application : Thumbnail Image")
    # Get image
    image_file = st.file_uploader('Load image ',type=['png', 'jpg'])
    if image_file is not None:
        st.image(image_file)
        GetThumbnail(image_file)
        
elif app == "OCR" :
    st.subheader("Application : OCR")
    st.write('ok')
    
elif app == "Face Analysis" :
    st.subheader("Application : Face Analysis")
    st.write('Do it yourself !')
