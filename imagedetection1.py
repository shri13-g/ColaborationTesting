import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes, num_boxes=5):
        super(SimpleObjectDetector, self).__init__()
        # Use a pre-trained backbone (ResNet50)
        self.backbone = models.resnet50(pretrained=True)
        # Remove the last classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # output: feature map

        # Additional layers for detection
        self.conv = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # For each grid cell, predict bounding boxes and class probabilities
        grid_size = 7
        self.grid_size = grid_size

        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.prediction = nn.Conv2d(1024, 
                                    (num_boxes * 5 + num_classes) * grid_size * grid_size, 
                                    kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)  # shape: (batch_size, 2048, H, W)
        features = self.relu(self.conv(features))
        predictions = self.prediction(features)
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, 
                                       self.grid_size, 
                                       self.grid_size, 
                                       self.num_boxes * 5 + self.num_classes)
        return predictions

# Load the model
model = SimpleObjectDetector(num_classes=20, num_boxes=5)

# Load a pre-trained classifier for image classification
classifier = models.resnet50(pretrained=True)
classifier.eval()

# Transform for input image
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225])   # ImageNet stds
])

# Load your input image
image_path = r"C:\Users\Krishna\Pictures\Screenshots\Screenshot (312).png"  # replace with your image path
img = Image.open(image_path).convert('RGB')

# Draw a rectangle around the entire image as a placeholder
draw_img = img.copy()
draw = ImageDraw.Draw(draw_img)

# Prepare the image for classification
input_tensor = transform(img).unsqueeze(0)  # add batch dimension

# Classify the image
with torch.no_grad():
    output = classifier(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)

# Map class ID to label
labels = {idx: label for idx, label in enumerate(open(r"C:\Users\Krishna\Desktop\imagenet_classes.txt"))}
# Or use a predefined list of ImageNet labels
imagenet_labels = [line.strip() for line in open(r"C:\Users\Krishna\Desktop\imagenet_classes.txt")]
predicted_label = imagenet_labels[top_catid.item()]

# Draw a rectangle around the whole image (or detected object if you have detection logic)
width, height = img.size
# Draw a rectangle around the entire image
draw.rectangle([(0, 0), (width, height)], outline='red', width=4)

# Save or show the image
draw_img.show()  # or draw_img.save('output.jpg')

# Print the classification result
print(f'Predicted: {predicted_label} ({top_prob.item()*100:.2f}%)')