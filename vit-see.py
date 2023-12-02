# Object detection with ViT (Vision Transformer) using PyTorch and OpenCV
#
# run: python vit-see.py
# model ref: https://huggingface.co/google/vit-base-patch16-224

import cv2
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

def initialize_vit_model():
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return feature_extractor, model

def classify_image(image, feature_extractor, model):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

def main():
    feature_extractor, model = initialize_vit_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR (OpenCV format) to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Classify image
        label = classify_image(rgb_image, feature_extractor, model)
        print("Predicted class:", label)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
