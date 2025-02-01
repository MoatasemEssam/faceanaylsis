import streamlit as st
import boto3
import os
import botocore.exceptions
import io
from PIL import Image, ImageDraw, ImageFont

def get_multiline_text_size(text, font, draw):
    try:
        # Use draw.multiline_textbbox if available (Pillow >= 8.0.0)
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older versions of Pillow
        lines = text.split('\n')
        widths = [font.getsize(line)[0] for line in lines]
        heights = [font.getsize(line)[1] for line in lines]
        width = max(widths)
        height = sum(heights)
    return width, height

def main():
    aws_access_key_id = st.secrets["aws"]["access_key_id"]
    aws_secret_access_key = st.secrets["aws"]["secret_access_key"]
    aws_region = st.secrets.get("aws", {}).get("region", "us-east-1")

    # Initialize Rekognition client
    rekognition = boto3.client(
        'rekognition',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read image content
        image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Analyze Image'):
            with st.spinner('Analyzing faces...'):
                # Convert image to bytes
                image_bytes = io.BytesIO()

                # Save image as PNG to bytes buffer
                image.save(image_bytes, format='PNG')
                image_bytes = image_bytes.getvalue()

                try:
                    # Call AWS Rekognition
                    response = rekognition.detect_faces(
                        Image={'Bytes': image_bytes},
                        Attributes=['ALL']
                    )
                except Exception as e:
                    st.error(f"Error calling Rekognition: {str(e)}")
                    return

                # Process response
                faces = response.get('FaceDetails', [])

                if not faces:
                    st.warning('No faces detected in the image.')
                else:
                    # Create a copy of the image to draw on
                    img_draw = image.copy()
                    draw = ImageDraw.Draw(img_draw)
                    # Use a TrueType font or default font
                    try:
                        font = ImageFont.truetype("arial.ttf", size=14)
                    except IOError:
                        font = ImageFont.load_default()

                    # Image dimensions
                    img_width, img_height = image.size

                    # Display details for each face
                    st.success(f'Detected {len(faces)} face(s) in the image.')

                    for i, face in enumerate(faces):
                        # Get bounding box
                        box = face['BoundingBox']
                        left = img_width * box['Left']
                        top = img_height * box['Top']
                        width = img_width * box['Width']
                        height = img_height * box['Height']

                        # Draw rectangle
                        points = [
                            (left, top),
                            (left + width, top),
                            (left + width, top + height),
                            (left, top + height),
                            (left, top)
                        ]
                        draw.line(points, fill='#00d400', width=2)

                        # Extract face attributes
                        age_range = face['AgeRange']
                        gender = face['Gender']['Value']
                        emotion = max(face['Emotions'], key=lambda x: x['Confidence'])['Type']

                        # Prepare label text
                        label = f"Age: {age_range['Low']}-{age_range['High']}\nGender: {gender}\nEmotion: {emotion}"

                        # Calculate text size
                        try:
                            text_width, text_height = get_multiline_text_size(label, font, draw)
                        except Exception as e:
                            st.error(f"Error calculating text size: {str(e)}")
                            return

                        # Adjust label position to prevent overflow
                        label_x = left
                        label_y = top - text_height - 10
                        if label_y < 0:
                            label_y = top + height + 10  # Place label below the bounding box if it doesn't fit above

                        # Define text background rectangle
                        text_background = [
                            label_x,
                            label_y,
                            label_x + text_width + 10,
                            label_y + text_height + 10
                        ]
                        # Draw background rectangle
                        draw.rectangle(text_background, fill='#000000')
                        # Draw text
                        draw.multiline_text(
                            (label_x + 5, label_y + 5),
                            label,
                            fill='#FFFFFF',
                            font=font
                        )

                        # Display face details in the app
                        st.write(f"**Face {i+1}:**")
                        st.write(f"- **Age Range:** {age_range['Low']} - {age_range['High']} years")
                        st.write(f"- **Gender:** {gender}")
                        st.write(f"- **Primary Emotion:** {emotion}")

                    # Display the image with detections
                    st.image(img_draw, caption='Analyzed Image.', use_column_width=True)

if __name__ == '__main__':
    main()