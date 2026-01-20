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
    "Powdery Mildew infection affecting mustard plants: SCORE 1 to 3",
    "Powdery Mildew infection affecting mustard plants: SCORE 5",
    "Powdery Mildew infection affecting mustard plants: SCORE 7",
    "Powdery Mildew infection affecting mustard plants: SCORE 9",
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
    'mustard_healthy': (
        "✅ **Healthy Mustard Plant**:\n"
        "• No visible signs of powdery mildew.\n"
        "• Maintain proper plant spacing and balanced fertilization.\n"
        "• Continue regular field monitoring.\n"
        "\n• **Powdery mildew yield weight :** {34.30 – 51.18 g}"
    ),

    'SCORE_1_3': (
        "🟡 **Mild Powdery Mildew Infection**:\n"
        "• Few white powdery spots observed on leaves.\n"
        "• Monitor crop regularly.\n"
        "• Ensure proper air circulation and sunlight.\n"
        "\n• **Powdery mildew yield weight :** {18.51 – 34.29 g}"
    ),

    'SCORE_5': (
        "🟠 **Moderate Powdery Mildew Infection**:\n"
        "• Infection spreading on multiple leaves.\n"
        "• Apply sulfur-based fungicides or baking soda spray.\n"
        "• Remove infected leaves if possible.\n"
        "\n• **Powdery mildew yield weight :** {13.75 – 18.50 g}"
    ),

    'SCORE_7': (
        "🔴 **Severe Powdery Mildew Infection**:\n"
        "• Large infected patches observed on plants.\n"
        "• Use recommended systemic fungicides.\n"
        "• Avoid overhead irrigation and improve plant spacing.\n"
        "\n• **Powdery mildew yield weight :** {7.08 – 13.74 g}"
    ),

    'SCORE_9': (
        "🔴❗ **Very Severe Powdery Mildew Infection**:\n"
        "• Extensive powdery mildew covering major plant area.\n"
        "• Immediate chemical fungicide intervention required.\n"
        "• Follow proper crop rotation in next season.\n"
        "\n• **Powdery mildew yield weight :** {0.03 – 7.07 g}"
    )
}

