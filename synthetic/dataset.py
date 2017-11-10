import torch
import torch.utils.data as data
import pickle

import pdb

class SyntheticDataset(data.Dataset):
    def __init__(self, path_to_dataset, dataset_type):
        '''
        Initializes the following values:

        From Args:
            dataset_type: an enum indicating if the data set is training, validation or test

        Via Calculation:

            sentence_size: max sentence size between all questions and captions
            memory_size: size of required memory for MemN2N

            species_clues: list of <clues>, index corresponds to <species_id>
            question_species: list of <species_id> corresponding to question
            question: list of <vectorized_question>
            answer: list of [0, 1] = Yes or [1, 0] = No
        '''

        # helper function
        def vectorize_synthetic_questions(questions_to_vectorize, num_attributes):

            # helpers inside helpers :O ~ nasty
            def one_hot(length, index):
                a = [0] * length
                a[index] = 1
                return a

            question_species = []
            questions = []
            answers = []

            num_answers = 2

            for q in questions_to_vectorize:
                species_id, attribute_id, answer_id = q
                question_species.append(species_id)
                questions.append(one_hot(num_attributes, attribute_id))
                answers.append(one_hot(num_answers, answer_id))

            return question_species, questions, answers

        species_clues, questions_train, questions_test, num_species, num_attributes, num_clues_per_species = pickle.load(
            open(path_to_dataset, "rb"))

        if dataset_type == "train":
            question_species, questions, answers = vectorize_synthetic_questions(questions_train, num_attributes)
        if dataset_type == "test":
            question_species, questions, answers = vectorize_synthetic_questions(questions_test, num_attributes)

        self.sentence_size = num_attributes
        self.memory_size = num_clues_per_species

        pdb.set_trace()
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
