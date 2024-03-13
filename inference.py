import torch
import torch.nn as nn
import torch.nn.functional as F
import json

num_features = 1878
num_classes = 500

# Define the same neural network model as used during training
class ThreeLayerNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Function to load the model
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeLayerNet(num_features, num_classes)  # specify num_features and num_classes
    with open(f"{model_dir}/model.pth", "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

# Function for processing input data
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        data = torch.tensor(data, dtype=torch.float32).reshape(1, -1)
        return data
    else:
        # Handle other content-types here or raise an exception
        pass

# Function for prediction
def predict_fn(input_data, model):
    model.eval()
    with torch.no_grad():
        return model(input_data)

# Function for processing output data
def output_fn(prediction_output, accept):
    if accept == "application/json":
        return json.dumps(prediction_output.numpy().tolist())
    else:
        # Handle other content-types here or raise an exception
        pass
