import os
import argparse
import random
import pickle
from sklearn.model_selection import train_test_split

import pdb

def one_hot(length, index):
    a = [0] * length
    a[index] = 1
    return a

def vectorize_questions(questions_to_vectorize, num_attributes):

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

def generate_synthetic_data(num_species, num_attributes, num_clues_per_species, test_percentage):
    '''
    Make synthetic data (species and attibutes are zero indexed)

    Returns two lists:
    
    species_clues: each element of list corresponds to a species and contains a list of clues for that species
    questions: tuples of the form (species_id, attribute_id, answer) 

    '''

    # Constants that might become variables
    proportion_of_species_with_attribute = 0.5

    # Initialize clues to all 0s
    species_clues = []
    for species_id in range(num_species):
        clues = []
        for x in range(num_clues_per_species):
            clues.append([0]*num_attributes)
        species_clues.append(clues)

    # Set questions and clues
    questions = []
    for attribute_id in range(num_attributes):
        
        selected_species = set(random.sample(list(range(num_species)), int(proportion_of_species_with_attribute*num_species)))
        
        # Set questions
        for species_id in range(num_species):
            if species_id in selected_species:
                questions.append((species_id, attribute_id, 1))
            else:
                questions.append((species_id, attribute_id, 0))

        # Set clues
        for species_id in selected_species:

            proportion_of_clues_with_attribute = random.uniform(0.3, 0.7)
            selected_clues = set(random.sample(list(range(num_clues_per_species)), int(proportion_of_clues_with_attribute*num_clues_per_species)))

            for clue_id in selected_clues:
                species_clues[species_id][clue_id][attribute_id] = 1

    # Get train vs test data
    questions_train, questions_test = train_test_split(questions, test_size = test_percentage)

    return species_clues, questions_train, questions_test


# def generate_and_or_questions(base_questions, num_questions):

#     # for each species generate a list of the attributes they have ground truth

#     # And Questions
#     # for each species generate a fixed number of true and false questions
#     # true questions by selecting a random number of attributes (in some fixed interval)
#     # false questions by selecting a random number of true + false attributes

#     # Or Questions
#     # for each species generate a fixed number of true and false
#     # true questions by selecting a random number of true + false attributes
#     # false questions by selecting a random number of false attributes



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Script to generate synthetic data for VQA-Memnet")

    arg_parser.add_argument("--num_species", type=int, default=100,
                            help="number of species")
    arg_parser.add_argument("--num_attributes", type=int, default=100,
                            help="number of attributes")
    arg_parser.add_argument("--num_clues_per_species", type=float, default=100,
                            help="number of clues per species")
    arg_parser.add_argument("--test_percentage", type=float, default=0.1,
                            help="percentage of quetsions to use for testing")
    args = arg_parser.parse_args()

    species_clues, questions_train, questions_test = generate_synthetic_data(num_species=args.num_species,
                                                       num_attributes=args.num_attributes,
                                                       num_clues_per_species=args.num_clues_per_species,
                                                       test_percentage=args.test_percentage)

    question_species_train, questions_train, answers_train = vectorize_questions(questions_train, args.num_attributes)
    question_species_test, questions_test, answers_test = vectorize_questions(questions_test, args.num_attributes)

    pickle.dump([species_clues, question_species_train, questions_train, answers_train, args.num_species, args.num_attributes, args.num_clues_per_species],
                 open( "synthetic_data_{}_species_{}_attributes_{}_clues_train.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))
    pickle.dump([species_clues, question_species_test, questions_test, answers_test, args.num_species, args.num_attributes, args.num_clues_per_species],
                 open( "synthetic_data_{}_species_{}_attributes_{}_clues_test.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))

