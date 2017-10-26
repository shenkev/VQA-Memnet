import torch.utils.data as data
import data_util as util
from enum import Enum, auto

import pdb


class DatasetType(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

class birdCaptionSimpleYesNoDataset(data.Dataset):
    def __init__(self, dataset_dir, limit_to_species, dataset_type):
        '''
    
        Initializes the following values:
    
        From Args:

            dataset_dir: the directory in which caption, attribute and question-answer pair information is stored
            limit_to_species: if true, relevant captions for a question are limited to those about a species, otherwise all captions are used
            dataset_type: an enum indicating if the data set is training, validation or test

        Via Calculation:

            captions: a dictionary of the form {<species_id>: <captions>}
                      <species_id> - integer representing the species, 0 indicates all species
                      <captions> - tuples of the form (<species_id>, <caption_words>)
                                   <species_id> - integer representing the species
                                   <vectorized_caption> - list of integers corresponding to words in word_idx
            qa_pairs: a list tuples of the form (<species_id>, <question>, <answer>)
                      <species_id> - integer representing the species
                      <vectorized_question> - list of integers corresponding to words in word_idx
                      <answer> - [0, 1] = Yes or [1, 0] = No
            word_idx: a dictionary of the form {<word>: <idx>}
            sentence_size: max sentence size between all questions and captions
            memory_size: size of required memory for MemN2N

        '''

        captions = util.load_captions(dataset_dir)

        qa_train, qa_val, qa_test = util.load_simple_yes_no_qa_pairs(dataset_dir)

        word_idx, sentence_size = util.get_common_parameters(captions, qa_train + qa_val + qa_test)

        captions = util.vectorize_captions(captions, word_idx, sentence_size)

        if dataset_type == DatasetType.TRAIN:
            qa_pairs = util.vectorize_qa_pairs(qa_train, word_idx, sentence_size)
        if dataset_type == DatasetType.VAL:
            qa_pairs = util.vectorize_qa_pairs(qa_val, word_idx, sentence_size)
        if dataset_type == DatasetType.TEST:
            qa_pairs = util.vectorize_qa_pairs(qa_test, word_idx, sentence_size)

        memory_size = len(captions[1]) if limit_to_species else len(captions[0])

        self.dataset_dir = dataset_dir
        self.limit_to_species = limit_to_species
        self.dataset_type = dataset_type
        self.captions = captions
        self.qa_pairs = qa_pairs
        self.word_idx = word_idx
        self.sentence_size = sentence_size
        self.memory_size = memory_size

    def __getitem__(self, idx):
        '''
        Returns:
        relevant_captions: list of tuples of the form (<species_id, vectorized_caption>)
        question: tuple of the form (<species_id, vectorized_question>)
        answer:  [0, 1] = Yes or [1, 0] = No
        '''
        relevant_captions = self.captions[self.qa_pairs[idx][0] if self.limit_to_species else 0]
        question = self.qa_pairs[idx][1]
        answer = self.qa_pairs[idx][2]
        return relevant_captions, question, answer

    def __len__(self):
        return len(self.qa_pairs)