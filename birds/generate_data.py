import os
import argparse
import random
import pickle

import pdb


def generate_synthetic_data(num_species, num_attributes, num_clues_per_species):
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

    return species_clues, questions


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Script to generate synthetic data for VQA-Memnet")

    arg_parser.add_argument("--num_species", type=int, default=100,
                            help="number of species")
    arg_parser.add_argument("--num_attributes", type=int, default=100,
                            help="number of attributes")
    arg_parser.add_argument("--num_clues_per_species", type=float, default=100,
                            help="number of clues per species")

    args = arg_parser.parse_args()

    species_clues, questions = generate_synthetic_data(num_species=args.num_species,
                                                       num_attributes=args.num_attributes,
                                                       num_clues_per_species=args.num_clues_per_species)


    pickle.dump([species_clues, questions],
                 open( "synthetic_data_{}_species_{}_attributes_{}_clues.pckl".format(args.num_species, args.num_attributes, args.num_clues_per_species), "wb" ))

