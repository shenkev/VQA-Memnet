from torch.utils.data import DataLoader
from dataset import SyntheticDataset
import data_util as util
import pdb

train_data = SyntheticDataset(path_to_dataset='/Users/atef/VQA-Memnet/birds/synthetic_data_100_species_100_attributes_100_clues.pckl', dataset_type="train")
train_loader = DataLoader(train_data, batch_size=5, num_workers=0, shuffle=True)
for batched_data in train_loader:
	pdb.set_trace()
	x=1
