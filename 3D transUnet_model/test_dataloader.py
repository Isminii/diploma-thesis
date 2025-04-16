from torch.utils.data import DataLoader, Subset
from brats_dataloader import MedicalImageDataset, load_splits

def get_test_dataloader(path, batch_size=1, num_workers=2, pin_memory=True, split_dir='splits2'):
    # Create dataset with resize=False to keep original resolution
    dataset = MedicalImageDataset(path, resize=False)
    # Load the saved splits
    splits = load_splits(split_dir=split_dir)
    test_indices = splits['test']
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return test_loader
