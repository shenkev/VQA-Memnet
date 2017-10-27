import torch
import torch.utils.data as data
import data_util as util

import pdb


class birdCaptionSimpleYesNoDataset(data.Dataset):
    def __init__(self, dataset_dir, limit_to_species, dataset_type):
        '''
    
        Initializes the following values:
    
        From Args:

            dataset_dir: the directory in which caption, attribute and question-answer pair information is stored
            limit_to_species: if true, relevant captions for a question are limited to those about a species, otherwise all captions are used
            dataset_type: an enum indicating if the data set is training, validation or test

        Via Calculation:

            word_idx: a dictionary of the form {<word>: <idx>}
            sentence_size: max sentence size between all questions and captions
            memory_size: size of required memory for MemN2N

            caption_species: dictionary of the form {<species_id>: list of <species_id>} (yes this is dumb and only mildly useful for the all species case)
            captions: dictionary of the form {<species_id>: list of <vectorized_caption>}
            question_species: list of <species_id> corresponding to questions
            question: list of <vectorized_question>
            answer: list of [0, 1] = Yes or [1, 0] = No
        '''

        captions = util.load_captions(dataset_dir)

        qa_train, qa_val, qa_test = util.load_simple_yes_no_qa_pairs(dataset_dir)

        word_idx, sentence_size = util.get_common_parameters(captions, qa_train + qa_val + qa_test)

        captions = util.vectorize_captions(captions, word_idx, sentence_size)

        if dataset_type == "train":
            qa_pairs = util.vectorize_qa_pairs(qa_train, word_idx, sentence_size)
        if dataset_type == "val":
            qa_pairs = util.vectorize_qa_pairs(qa_val, word_idx, sentence_size)
        if dataset_type == "test":
            qa_pairs = util.vectorize_qa_pairs(qa_test, word_idx, sentence_size)

        memory_size = len(captions[1]) if limit_to_species else len(captions[0])

        self.dataset_dir = dataset_dir
        self.limit_to_species = limit_to_species
        self.dataset_type = dataset_type
        self.word_idx = word_idx
        self.sentence_size = sentence_size
        self.memory_size = memory_size

        self.captions_species = {species_id: torch.LongTensor([caption[0] for caption in species_captions]) for (species_id, species_captions) in captions.items()}   
        self.captions = {species_id: torch.LongTensor([caption[1] for caption in species_captions]) for (species_id, species_captions) in captions.items()}   
        self.question_species = torch.LongTensor([qa_pair[0] for qa_pair in qa_pairs])
        self.question = torch.LongTensor([qa_pair[1] for qa_pair in qa_pairs])
        self.answer = torch.LongTensor([qa_pair[2] for qa_pair in qa_pairs])

    def __getitem__(self, idx):
        '''
        Returns:
        captions_species: species_id corresponding to each caption
        captions: vectorized captions relevant to the question
        quesion_species: species_id corresponding to the question
        question: vectorized question
        answer:  [0, 1] = Yes or [1, 0] = No
        '''
        captions_idx = self.question_species[idx] if self.limit_to_species else 0
        return self.captions_species[captions_idx], self.captions[captions_idx], self.question_species[idx], self.question[idx], self.answer[idx]

    def __len__(self):
        return len(self.question)