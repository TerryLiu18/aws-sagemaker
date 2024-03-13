import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Define the neural network model
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

def train(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 1:
                end_time = time.time()
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    model_dir = os.environ['SM_MODEL_DIR']
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=500) 
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()
    train_data_path = os.path.join(args.train, 'train_data.npy')
    train_labels_path = os.path.join(args.train, 'train_label.npy')
    all_features = np.load(train_data_path)
    all_labels_original = np.load(train_labels_path)
    
    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels_original)
    
    all_labels = torch.from_numpy(all_labels)
    all_features = torch.from_numpy(all_features).float()
    
    train_dataset = TensorDataset(all_features, all_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    num_features = all_features.shape[1]
    model = ThreeLayerNet(num_features, args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train(model, train_loader, optimizer, args.epochs)
   