# train_event_model.py
import torch
from torch.utils.data import Dataset, DataLoader
from models.lgmd_net import LGMDNet
from event_data_loader import load_event_data
import os
import h5py
import hdf5plugin

class EventDataset(Dataset):          
    def __init__(self, dataset_path):

        self.samples = []
        print("Looking for data inside:", dataset_path)
        for seq in os.listdir(dataset_path):
            evt_path = os.path.join(dataset_path, seq, 'left', 'events.h5')
            print("evt_path",evt_path)
            if os.path.exists(evt_path):
                self.samples.append(evt_path)
                print("this is self.samples: ",self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        events = load_event_data(self.samples[idx])
        # Convert to a 2D event image or voxel grid
        event_tensor = torch.zeros(2, 64, 64)  # Simplified
        # Fill event_tensor with logic...
        label = torch.tensor([1.0])  # Dummy label
        return event_tensor, label

def train():
    dataset = EventDataset(r"train_dataset\zurich_city_11_a")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(loader)
    model = LGMDNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(5):
        for batch_x, batch_y in loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
    torch.save(model.state_dict(), r"F:\cRAIS\new_insect_inspired_navigation\models\lgmd_net.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
