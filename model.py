import numpy as np
import torch
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO


path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "")

class acqalens: 
    def __init__(self, model) -> None:
        if torch.backends.mps.is_available():
            self.mps_device = torch.device("mps")
            print("MPS working...")
        else:
            quit("MPS device not found.")
        self.model = YOLO(model)
        self.image_name = 'output_image'
        self.image_type = 'jpg'
        self.image_path = ''
    
    def predict(self, image):
        return self.visualize_predictions(self.image_prediction(image), image)
    
    def image_prediction(self, image):
        return self.model.predict(source=image, device=self.mps_device, save=True)
    
    def visualize_predictions(self, prediction, image):
        img = plt.imread(image)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        confidence_stack = []
        for box in prediction[0].boxes:
            # Extract the bounding box coordinates, confidence, and label as scalar values
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            confidence = (box.conf.cpu().numpy().item())*100  # Use .item() to extract the scalar
            label = int(box.cls.cpu().numpy().item())  # Use .item() and int() to extract the scalar
            confidence_stack.append(confidence)
            # Create a rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min), 
                x_max - x_min, 
                y_max - y_min, 
                linewidth=1, 
                edgecolor='r', 
                facecolor='none'
            )

            # Add the patch to the Axes
            plt.gca().add_patch(rect)
            plt.text(x_min, y_min, f'{confidence:.2f}%', color='white', fontsize=12, backgroundcolor='red')

        
        plt.axis('off')

        self.image_path = path + self.image_name + '.' + self.image_type
        plt.savefig(self.image_path, bbox_inches='tight', pad_inches=0)

        plt.show()


        self.object_count = len(prediction[0].boxes)
        self.max_confidence, self.min_confidence = max(confidence_stack), min(confidence_stack)
        self.average_confidence = sum(confidence_stack) / self.object_count

    def properties(self):
        return {'image_path': self.image_path, 'image_type': self.image_type, 'image_name': self.image_name, 'min':round(self.min_confidence, 2) , 'max': round(self.max_confidence, 2), 'average': round(self.average_confidence, 2), 'count': self.object_count}    
if __name__ == "__main__":
    pretrained_model_location = path + 'best.pt'
    model = acqalens(pretrained_model_location)
    model.predict(path + '/data/images/train/1_jpg.rf.3bd160570ca04ce20d506e7d30bba9db.jpg')
    print(
        model.image_path,
        model.min_confidence,
        model.properties()
    )