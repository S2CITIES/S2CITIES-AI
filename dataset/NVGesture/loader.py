import torch
from torch.utils.data import Dataset
from .readdata import load_split_nvgesture, load_data_from_file
from torch.utils.data import DataLoader
import os.path

class NVGestureColorDataset(Dataset):
    def __init__(self, annotations_file, path_prefix):
        self.data = list()
        # Fill self.data with annotations from annotations_file
        load_split_nvgesture(file_with_split = annotations_file, list_split = self.data)
        self.path_prefix = path_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = load_data_from_file(example_config = self.data[idx], 
                                          sensor = 'color', 
                                          image_width = 320, 
                                          image_height = 240, 
                                          starting_path=self.path_prefix)
        return data, label
    
if __name__ == '__main__':
    train_dataset = NVGestureColorDataset('./nvgesture_train_correct_cvpr2016.lst', path_prefix='.')
    test_dataset = NVGestureColorDataset('./nvgesture_test_correct_cvpr2016.lst', path_prefix='.')

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Testing if it works correctly
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")