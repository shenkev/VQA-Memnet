import os
import argparse
import random
import pickle
from sklearn.model_selection import train_test_split

import pdb

def binary_vector(length, indicies):

    a = [0] * length
    for index in indicies:
        a[index] = 1
    return a

def generate_synthetic_data(num_species, num_attributes, num_clues_per_species):
    '''
    Make synthetic data (species and attibutes are zero indexed)

    Returns two lists:
    
    species_clues: each element of list corresponds to a species and contains a list of clues for that species
    questions: tuples of the form (species_id, attribute_id, answer) 

    '''

    # Constants that might become variables
    proportion_of_species_with_attribute = 0.5
    proportion_of_clues_with_attribute_low = 0.3
    proportion_of_clues_with_attribute_high = 0.7

    # Initialize clues to all 0s
    species_clues = []
    species_index_clues = []
    for species_id in range(num_species):
        clues = []
        species_index_clues.append([])
        for x in range(num_clues_per_species):
            clues.append([0]*num_attributes)
            species_index_clues[species_id].append([])
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

            proportion_of_clues_with_attribute = random.uniform(proportion_of_clues_with_attribute_low, proportion_of_clues_with_attribute_high)
            selected_clues = set(random.sample(list(range(num_clues_per_species)), max(1, int(round(proportion_of_clues_with_attribute*num_clues_per_species)))))

            for clue_id in selected_clues:
                species_clues[species_id][clue_id][attribute_id] = 1
                species_index_clues[species_id][clue_id].append(attribute_id+1)

    # return the index clues instead of the binary clues
    return species_index_clues, questions

def generate_simple_questions(base_questions, num_attributes):

    simple_questions = []
    for q in base_questions:
        species_id, attribute_id, answer_id = q
        simple_questions.append((species_id, binary_vector(num_attributes, [attribute_id]), binary_vector(2, [answer_id])))

    return simple_questions

def generate_and_or_questions(base_questions, num_questions_to_generate, num_species, num_attributes):

    # Constants that might become variables
    question_num_attribute_max = 10

    # Get all true attributes for each species
    species_attributes = {}
    for question in base_questions:
        species_id, attribute_id, answer_id = question
        if species_id not in species_attributes:
            species_attributes[species_id] = set()
        if answer_id == 1:
            species_attributes[species_id].add(attribute_id)

    all_attributes = set(range(num_attributes))

    # Generate And questions
    and_questions_true = []
    and_questions_false = []

    # Generate half the questions as true
    # Done by selecting a random number of attributes
    for _ in xrange(num_questions_to_generate / 2):
        species_id = random.randint(0, num_species-1)
        true_attributes = species_attributes[species_id]
        
        num_true_attributes_for_question = random.randint(1, min(len(true_attributes), question_num_attribute_max))
        target_attributes = random.sample(list(true_attributes), num_true_attributes_for_question)
        
        and_questions_true.append((species_id, binary_vector(num_attributes, target_attributes), binary_vector(2, [1])))

    # Generate half the questions as false
    # Done by selecting a random number of true and false attributes (num true can equal 0)
    for _ in xrange(num_questions_to_generate / 2):
        species_id = random.randint(0, num_species-1)
        true_attributes = species_attributes[species_id]
        false_attributes = all_attributes - true_attributes
        
        num_true_attributes_for_question = random.randint(0, min(len(true_attributes), question_num_attribute_max / 2))
        true_target_attributes = random.sample(list(true_attributes), num_true_attributes_for_question)

        num_false_attributes_for_question = random.randint(1, min(len(false_attributes), question_num_attribute_max / 2))
        false_target_attributes = random.sample(list(false_attributes), num_false_attributes_for_question)
       
        and_questions_false.append((species_id, binary_vector(num_attributes, true_target_attributes + false_target_attributes), binary_vector(2, [0])))

    and_questions = and_questions_true + and_questions_false

    # Generate Or questions
    or_questions_true = []
    or_questions_false = []

    # Generate half the questions as true
    # Done by selecting a random number of true and false attributes (num false can equal 0)
    for _ in xrange(num_questions_to_generate / 2):
        species_id = random.randint(0, num_species-1)
        true_attributes = species_attributes[species_id]
        false_attributes = all_attributes - true_attributes
        
        num_true_attributes_for_question = random.randint(1, min(len(true_attributes), question_num_attribute_max / 2))
        true_target_attributes = random.sample(list(true_attributes), num_true_attributes_for_question)

        num_false_attributes_for_question = random.randint(0, min(len(false_attributes), question_num_attribute_max / 2))
        false_target_attributes = random.sample(list(false_attributes), num_false_attributes_for_question)
       
        or_questions_true.append((species_id, binary_vector(num_attributes, true_target_attributes + false_target_attributes), binary_vector(2, [1])))

    # Generate half the questions as false
    # Done by selecting a random number false attributes
    for _ in xrange(num_questions_to_generate / 2):
        species_id = random.randint(0, num_species-1)
        false_attributes = all_attributes - species_attributes[species_id]
        
        num_false_attributes_for_question = random.randint(1, min(len(false_attributes), question_num_attribute_max))
        target_attributes = random.sample(list(false_attributes), num_false_attributes_for_question)
        
        or_questions_false.append((species_id, binary_vector(num_attributes, target_attributes), binary_vector(2, [0])))

    or_questions = or_questions_true + or_questions_false

    return and_questions, or_questions, species_attributes


def get_index_questions(questions):
    for k in range(len(questions)):
        questions[k] = (questions[k][0],
                        [i+1 for i, j in enumerate(questions[k][1]) if j == 1],
                        questions[k][2])


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Script to generate synthetic data for VQA-Memnet")

    arg_parser.add_argument("--num_species", type=int, default=100,
                            help="number of species")
    arg_parser.add_argument("--num_attributes", type=int, default=100,
                            help="number of attributes")
    arg_parser.add_argument("--num_clues_per_species", type=int, default=100,
                            help="number of clues per species")
    arg_parser.add_argument("--num_and_or_questions", type=int, default=0,
                            help="number of clues per species")
    arg_parser.add_argument("--test_percentage", type=float, default=0.1,
                            help="percentage of questions to use for testing")
    args = arg_parser.parse_args()

    species_clues, base_questions = generate_synthetic_data(num_species=args.num_species,
                                                            num_attributes=args.num_attributes,
                                                            num_clues_per_species=args.num_clues_per_species)
    
    # Generate Simple Questions
    simple_questions = generate_simple_questions(base_questions, args.num_attributes)
    # get indices instead of binary questions
    get_index_questions(simple_questions)
    simple_questions_train, simple_questions_test = train_test_split(simple_questions, test_size = args.test_percentage)

    pickle.dump([species_clues, simple_questions_train, args.num_species, args.num_attributes, args.num_clues_per_species],
                 open( "synthetic_index_data_{}_species_{}_attributes_{}_clues_train.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))
    pickle.dump([species_clues, simple_questions_test, args.num_species, args.num_attributes, args.num_clues_per_species],
                 open( "synthetic_index_data_{}_species_{}_attributes_{}_clues_test.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))

    # # Generate And/Or Questions
    # if args.num_and_or_questions > 0:
    #     and_questions, or_questions, species_attributes = generate_and_or_questions(base_questions, args.num_and_or_questions, args.num_species, args.num_attributes)
    #     and_questions_train, and_questions_test = train_test_split(and_questions, test_size = args.test_percentage)
    #     or_questions_train, or_questions_test = train_test_split(or_questions, test_size = args.test_percentage)
    #
    #     #pdb.set_trace()
    #
    #     pickle.dump([species_clues, and_questions_train, args.num_species, args.num_attributes, args.num_clues_per_species],
    #                  open( "synthetic_data_{}_species_{}_attributes_{}_clues_and_train.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))
    #     pickle.dump([species_clues, and_questions_test, args.num_species, args.num_attributes, args.num_clues_per_species],
    #                  open( "synthetic_data_{}_species_{}_attributes_{}_clues_and_test.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))
    #
    #     pickle.dump([species_clues, or_questions_train, args.num_species, args.num_attributes, args.num_clues_per_species],
    #                  open( "synthetic_data_{}_species_{}_attributes_{}_clues_or_train.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))
    #     pickle.dump([species_clues, or_questions_test, args.num_species, args.num_attributes, args.num_clues_per_species],
    #                  open( "synthetic_data_{}_species_{}_attributes_{}_clues_or_test.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))
    #





