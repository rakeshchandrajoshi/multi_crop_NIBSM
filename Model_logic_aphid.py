import torch
import torch.nn as nn
from torchvision import transforms, models
import contextlib

# ====== Configuration ======
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 6
MODEL_PATH = 'best_finegrained_model2.pth'
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
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.cbam = CBAM(2048)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
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
    "Aphid Infestation: 101–250 Aphids",
    "Aphid Infestation: 20–30 Aphids",
    "Aphid Infestation: 251–500 Aphids",
    "Aphid Infestation: 31–100 Aphids",
    "Aphid Infestation: More Than 500 Aphids",
    "Healthy Mustard Plant"
]

original_labels = [
    'aphid_101_250',
    'aphid_20_30',
    'aphid_251_500',
    'aphid_31_100',
    'aphid_more_than_500',
    'mustard_healthy'
]

remedies = {
    'mustard_healthy': (
        "✅ **Healthy Plant**:\n"
        "• No aphid infestation detected.\n"
        "• Maintain proper irrigation and nutrient balance.\n"
        "• Continue regular field monitoring.\n"
        "\n• **Aphid Yield weight:** {20.01 – 51.22 g}"
    ),

    'aphid_20_30': (
        "🟡 **Mild Aphid Infestation**:\n"
        "• Light aphid presence on plant.\n"
        "• Monitor daily for population increase.\n"
        "• Encourage natural predators like ladybirds.\n"
        "\n• **Aphid Yield weight:** {1.81 – 8.00 g}"
    ),

    'aphid_31_100': (
        "🟠 **Moderate Aphid Infestation**:\n"
        "• Aphid population increasing.\n"
        "• Use yellow sticky traps.\n"
        "• Apply neem oil or botanical insecticide.\n"
        "\n• **Aphid Yield weight:** {1.11 – 1.80 g}"
    ),

    'aphid_101_250': (
        "🔴 **High Aphid Infestation**:\n"
        "• Heavy aphid presence observed.\n"
        "• Apply neem oil or insecticidal soap.\n"
        "• Monitor neighboring plants.\n"
        "\n• **Aphid Yield weight:** {8.01 – 20.00 g}"
    ),

    'aphid_251_500': (
        "🔴 **Severe Aphid Infestation**:\n"
        "• Large aphid colonies on plant.\n"
        "• Use strong bio-pesticides or selective chemical spray.\n"
        "• Isolate affected plants if possible.\n"
        "\n• **Aphid Yield weight:** {0.71 – 1.10 g}"
    ),

    'aphid_more_than_500': (
        "🔴❗ **Very Severe Aphid Infestation**:\n"
        "• Extremely high aphid population.\n"
        "• Immediate chemical treatment is mandatory.\n"
        "• High risk of yield failure.\n"
        "\n• **Aphid Yield weight:** {0.11 – 0.70 g}"
    )
}

