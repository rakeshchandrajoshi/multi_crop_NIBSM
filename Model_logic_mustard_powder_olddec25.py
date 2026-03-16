import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import contextlib

# ====== Configuration ======
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 5
MODEL_PATH = 'best_mobilenetv2_finegrained_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
amp_autocast = torch.cuda.amp.autocast if USE_AMP else contextlib.nullcontext

# ====== CBAM Module ======
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(torch.cat(
            [ca.mean(1, keepdim=True), ca.max(1, keepdim=True)[0]], dim=1))
        return sa * ca

# ====== Model Definition ======
class FineGrainedNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        backbone = mobilenet_v2(weights=weights)
        self.feature_extractor = backbone.features 

        self.cbam = CBAM(1280)  

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.head(x)
        return x

# ====== Load Model ======
@torch.no_grad()
def load_model():
    model = FineGrainedNet(NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ====== Image Preprocessing ======
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# ====== Labels ======
class_names = [
    "Mild Powdery Mildew infection affecting mustard plants",
    "Moderate Powdery Mildew infection affecting mustard plants",
    "Severe Powdery Mildew infection affecting mustard plants",
    "Very Severe Powdery Mildew infection affecting mustard plants",
    "Healthy Mustard Plant"
]

original_labels = [
    'SCORE_1_3',
    'SCORE_5',
    'SCORE_7',
    'SCORE_9',
    'mustard_healthy'
]

remedies = {
    'SCORE_1_3': "🟡 **Mild Powdery Mildew Infection**: Few spots observed. Monitor crop regularly and ensure proper air circulation.",
    'SCORE_5': "🟠 **Moderate Infection**: Apply sulfur-based fungicides or organic treatments like baking soda spray. Remove affected leaves if possible.",
    'SCORE_7': "🔴 **Severe Infection**: Large patches observed. Use recommended fungicides and avoid overhead watering. Improve plant spacing.",
    'SCORE_9': "🔴❗ **Very Severe Infection**: Extensive mildew present. Immediate chemical fungicide intervention is required. Consider crop rotation in future.",
    'mustard_healthy': "✅ Your mustard plant appears healthy. Maintain proper spacing and monitor for early signs of powdery mildew."
}

