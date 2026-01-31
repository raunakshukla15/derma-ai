import torch, json
from torchvision import models, transforms
from PIL import Image
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("model/classes.json") as f:
    classes = json.load(f)

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.eval().to(device)

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = softmax(model(img), dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return classes[idx.item()], float(conf.item())
