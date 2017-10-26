from torch.utils.data import DataLoader
from dataset import DatasetType, birdCaptionSimpleYesNoDataset
import pdb


train_data = birdCaptionSimpleYesNoDataset(dataset_dir='/Users/atef/VQA-Memnet/birds', limit_to_species=False, dataset_type=DatasetType.TRAIN)
train_loader = DataLoader(train_data, batch_size=6, num_workers=0, shuffle=False)
pdb.set_trace()
for batched_data in train_loader:
	pdb.set_trace()
	x=1
# train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)

# test_data = bAbIDataset(dataset_dir, task, train=False)
# test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=False)

# captions, questions, answers, vocab = util.load_data('/Users/atef/VQA-Memnet/birds', 10.0)
# sentence_size, vocab_size, word_idx = util.calculate_parameter_values(captions=captions, questions=questions, vocab=vocab)
# captions_vec, questions_vec, answers_vec = util.vectorize_data(captions=captions, questions=questions, answers=answers,
#                                                               sentence_size=sentence_size, word_idx=word_idx)

# S, Q, A = util.generate_s_q_a(questions=questions_vec, answers=answers_vec, limit_to_species=False)

# data = (S, Q, A)
# train_set, train_batches, test_set, test_batches = util.batch_data(data=data, batch_size=16, test_size=0.2)


# pdb.set_trace()
# x = 1