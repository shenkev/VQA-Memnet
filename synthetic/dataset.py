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

        species_clues, question_tuples, num_species, num_attributes, num_clues_per_species = pickle.load(
        open(path_to_dataset, "rb"))
        question_species, questions, answers = map(list, zip(*question_tuples))

        # # pad clues with 0s
        # max_att_per_clue = 0
        # min_att_per_clue = 10000
        # for i in range(len(species_clues)):
        #     for j in range(len(species_clues[0])):
        #         max_att_per_clue = max(max_att_per_clue, len(species_clues[i][j]))
        #         min_att_per_clue = min(min_att_per_clue, len(species_clues[i][j]))
        #
        # for i in range(len(species_clues)):
        #     for j in range(len(species_clues[0])):
        #         species_clues[i][j].extend([0] * (max_att_per_clue-len(species_clues[i][j])))
        #
        # for i in range(len(questions)):
        #     questions[i].extend([0] * (max_att_per_clue-len(questions[i])))
        max_clue_per_specie = 0
        min_clue_per_specie = 10000
        for i in range(len(species_clues)):
            max_clue_per_specie = max(max_clue_per_specie, len(species_clues[i]))
            min_clue_per_specie = min(min_clue_per_specie, len(species_clues[i]))

        for i in range(len(species_clues)):
            for _ in range(max_clue_per_specie-len(species_clues[i])):
                species_clues[i].append([0])

        self.sentence_size = num_attributes
        self.memory_size = num_clues_per_species
        self.vocab_size = num_attributes
        # self.max_att_per_clue = max_att_per_clue
        self.max_clue_per_specie = max_clue_per_specie

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
