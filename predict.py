import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 1. REDEFINE THE MODEL ARCHITECTURE ===
# PyTorch needs to know the model structure before it can load the weights.
# Copy-paste the AgeGenderModel class exactly as it was during training.
class AgeGenderModel(nn.Module):
    def __init__(self):  # 'num_features' removed, since it's determined by ResNet50
        super(AgeGenderModel, self).__init__()
        self.base_model = models.resnet50(pretrained=False)

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # gender branch
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        # age branch
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.base_model(x)
        features = features.view(features.size(0), -1)

        gender_output = self.gender_head(features)
        age_output = self.age_head(features)

        return gender_output, age_output


# === 2. LOAD THE MODEL AND THE SAVED WEIGHTS ===
# Initialize the model with the same architecture
model = AgeGenderModel().to(device)

# .pth
model_path = 'age_gender_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
model.eval()
print(f"The model was successfully loaded from {model_path}")


# === 3. PREPARE THE IMAGE AND THE PREDICTION FUNCTION ===
# Transform for the input image (must match the validation transform used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_path, model):
    try:
        # open and transform the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Disable gradient computation for inference
        with torch.no_grad():
            gender_output, age_output = model(image_tensor)

            # Process gender output
            gender_prob = torch.sigmoid(gender_output).item()
            gender = "Male" if gender_prob > 0.5 else "Female"

            # Process age output
            age = age_output.item()

        print("\n--- Prediction Result ---")
        print(f"Image: {image_path}")
        print(f"Gender: {gender} (Probability of Male: {gender_prob:.2f})")
        print(f"Estimated Age: {age:.1f} years")
        print("-------------------------")

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")


# === 4. RUN THE PREDICTION ===
# Replace 'path/to/your/image.jpg' with the path of the image you want to test.
# You can download a face image from the internet to try it out.
test_image_path = '27.jpg'
predict_image(test_image_path, model)

# Another example
# other_image_path = 'actor_face.jpg'
# predict_image(other_image_path, model)