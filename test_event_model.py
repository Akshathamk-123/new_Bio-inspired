import torch
from torch.utils.data import Dataset, DataLoader
from models.lgmd_net import LGMDNet
from event_data_loader import load_event_data
import os

class EventTestDataset(Dataset):
    def __init__(self, dataset_path):
        self.samples = []
        for seq in os.listdir(dataset_path):
            evt_path = os.path.join(dataset_path, seq, 'events', 'left', 'events.h5')
            if os.path.exists(evt_path):
                self.samples.append(evt_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        events = load_event_data(self.samples[idx])
        event_tensor = torch.zeros(2, 64, 64)
        return event_tensor

def test():
    dataset = EventTestDataset(r"test_dataset\zurich_city_04_e")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = LGMDNet()
    model.load_state_dict(torch.load(r"F:\cRAIS\new_insect_inspired_navigation\models\lgmd_net.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        for idx, batch_x in enumerate(loader):
            output = model(batch_x)
            print(f"Sample {idx}: Prediction={output.item():.4f}")

if __name__ == "__main__":
    test()