import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import contextlib

# ====== Configuration ======
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 5
MODEL_PATH = 'best_wheat_resnet50_finegrained_model.pth'
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
        backbone = resnet50(pretrained=True)
        
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2]) 
        self.cbam = CBAM(2048)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, num_classes)
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
    "Healthy wheat plant with no white ear symptoms",
    "Wheat spike turns white early due to disease, reducing yield: Single White Ear",
    "Wheat spike turns white early due to disease, reducing yield: Double White Ear",
    "Wheat spike turns white early due to disease, reducing yield: Triple White Ear",
    "Wheat spike turns white early due to disease, reducing yield: Multiple White Ear" 
]

original_labels = [
        '0_HEALTHY_PLANT',
    '1_Single_White_Ear',
    '2_Double_White_Ear',
    '3_Trple_White_Ear',
    '5_6_7_white_Ear',

]

remedies = {
    '0_HEALTHY_PLANT': (
        "✅ **Healthy Plant**:\n"
        "• Maintain good irrigation and nutrient balance.\n"
        "• Watch for early signs of ear whitening.\n"
        "\n• **Weight of husk less grain:** {10.0 – 17.0 g}\n"
        "\n• **Number of affected tillers / Total tillers:** {0 -- 0 %}"
    ),

    '1_Single_White_Ear': (
        "🟡 **Very Mild White Ear**:\n"
        "One ear affected — monitor plant closely.\n"
        "• Ensure proper irrigation and avoid heat stress.\n"
        "\n• **Weight of husk less grain:** {7.0 – 9.9 g}\n"
        "\n• **Number of affected tillers / Total tillers:** {17 -- 33.32 %}"
    ),

    '2_Double_White_Ear': (
        "🟡 **Mild White Ear**:\n"
        "• Two ears affected.\n"
        "• Check for early signs of pest or fungal infection.\n"
        "• Maintain optimal nutrition.\n"
        "\n• **Weight of husk less grain:** {4.0 – 6.9 g}\n"
        "\n• **Number of affected tillers / Total tillers:** {33.33 -- 50 %}"
    ),

    '3_Triple_White_Ear': (
        "🟠 **Moderate Infection**:\n"
        "• Three ears affected.\n"
        "• Apply appropriate fungicides or pest control.\n"
        "• Monitor surrounding crops.\n"
        "\n• **Weight of husk less grain:** {2.0 – 3.9 g}\n"
        "\n• **Number of affected tillers / Total tillers:** {51 -- 75 %}"
    ),

    '5_6_7_White_Ear': (
        "🔴❗ **Severe White Ear**:\n"
        "• Multiple ears affected — high risk of yield loss.\n"
        "• Immediate intervention with systemic fungicide or agronomic correction required.\n"
        "\n• **Weight of husk less grain:** {0 – 1.9 g}\n"
        "\n• **Number of affected tillers / Total tillers:** {76 -- 96 %}"
    ),
}

