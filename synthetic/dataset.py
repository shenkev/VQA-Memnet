import torch
import torch.utils.data as data
import pickle

import pdb

class SyntheticDataset(data.Dataset):
    def __init__(self, path_to_dataset):
        '''
        Initializes the following values:

            sentence_size: max sentence size between all questions and captions
            memory_size: size of required memory for MemN2N

            species_clues: list of <clues>, index corresponds to <species_id>
            question_species: list of <species_id> corresponding to question
            question: list of <vectorized_question>
            answer: list of [0, 1] = Yes or [1, 0] = No
        '''

        species_clues, question_species, questions, answers, num_species, num_attributes, num_clues_per_species = pickle.load(
        open(path_to_dataset, "rb"))

        self.sentence_size = num_attributes
        self.memory_size = num_clues_per_species

        self.species_clues = torch.LongTensor(species_clues)
        self.question_species = torch.LongTensor(question_species)
        self.questions = torch.LongTensor(questions)
        self.answer = torch.LongTensor(answers)

    def __getitem__(self, idx):
        '''
        Returns:
        species_clues: clues relevant to the question
        quesion_species: species_id corresponding to the question
        question: vectorized question
        answer:  [0, 1] = Yes or [1, 0] = No
        '''
        return self.species_clues[self.question_species[idx]], self.question_species[idx], self.questions[idx], \
               self.answer[idx]

    def __len__(self):
        return len(self.questions)
