from torch.utils.data import DataLoader
from dataset import DatasetType, birdCaptionSimpleYesNoDataset
import pdb


train_data = birdCaptionSimpleYesNoDataset(dataset_dir='/Users/atef/VQA-Memnet/birds', limit_to_species=False, dataset_type=DatasetType.TRAIN)
train_loader = DataLoader(train_data, batch_size=5, num_workers=0, shuffle=False)
for batched_data in train_loader:
	pdb.set_trace()
	x=1
