from torch.utils.data import DataLoader
from dataset import SyntheticDataset
import pdb

train_data = SyntheticDataset(path_to_dataset='/Users/atef/VQA-Memnet/synthetic/synthetic_data_5_species_6_attributes_5_clues_or_train.pckl')
train_loader = DataLoader(train_data, batch_size=5, num_workers=0, shuffle=True)
for batched_data in train_loader:
	pdb.set_trace()
	x=1
