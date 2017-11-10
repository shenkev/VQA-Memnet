import os
import argparse

from sklearn.model_selection import train_test_split

import pdb


def write_questions_to_file(questions, filepath):

    '''
    write questions to given file

    questions are expected to be of the form of a list/tuple and are written to a file one question per line
    with whitespace delimiting the items in the list/tuple

    Args:
        questions: list of questions
        filepath: path of file to write to
    '''
    
    with open(filepath, 'w') as f:
        for question in questions:
            question_str = ' '.join([str(x) for x in question])
            f.write("{}\n".format(question_str))


def generate_simple_yes_no_questions(data_dir, is_present_threshold, use_two_thresholds, val_percentage, test_percentage):

    '''
    generate simple yes-no questions of the form class and attribute 
    (i.e. is this attribute present for the class, yes or no)

    yes-or-no value is determined by whether the attribute is present in a higher percentage of images than a given threshold

    questions are written to files in the follow format:
    <species_id> <attribute_id> <is_present> - where <is_present> is either 0 or 1

    the questions are written to three files: simple_yes_no_train.txt, simple_yes_no_val.txt and simple_yes_no_test.txt

    Args:
        class_attributes_dir: path to where class_attribute_labels_continuous.txt is contained
        is_present_threshold: threshold for deciding whether the attribute is present for the class
        val_percentage: percentage of questions to be used for validation set
        test_percentage: percentage of questions to be used for test set
    '''

    class_attributes_path = os.path.join(data_dir, "class_attribute_labels_continuous.txt")

    lines = []
    
    with open(class_attributes_path) as f:
        lines = f.readlines()
    
    questions = []
    
    for species_id, line in enumerate(lines, 1):  
        attribute_present_percentages = line.split()
        for attribute_id, present_percentage in enumerate(attribute_present_percentages, 1):
            if use_two_thresholds:
                if 0.01 > float(present_percentage):
                    questions.append((species_id, attribute_id, 0))
                if (is_present_threshold)*100 < float(present_percentage):
                    questions.append((species_id, attribute_id, 1))
            else:
                is_present = int(is_present_threshold*100 <= float(present_percentage)) # multiply by 100 since file stores percentage values
                questions.append((species_id, attribute_id, is_present))


    questions_train, questions_holdout = train_test_split(questions, test_size = val_percentage + test_percentage)
    questions_val, questions_test = train_test_split(questions_holdout, test_size = test_percentage / (val_percentage+test_percentage))

    train_path = os.path.join(data_dir, "simple_yes_no_train_05_ceil_0_floor.txt")
    val_path = os.path.join(data_dir, "simple_yes_no_val_05_ceil_0_floor.txt")
    test_path = os.path.join(data_dir, "simple_yes_no_test_05_ceil_0_floor.txt")

    write_questions_to_file(questions_train, train_path)
    write_questions_to_file(questions_val, val_path)
    write_questions_to_file(questions_test, test_path)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Script to generate questions for VQA-Memnet")

    arg_parser.add_argument("--data_dir", type=str, default="./../birds/",
                            help="path to folder from where data is loaded")
    arg_parser.add_argument("--is_present_threshold", type=float, default=0.05,
                            help="threshold to determine if an attribute is present")
    arg_parser.add_argument("--use_two_thresholds", type=bool, default=True,
                            help="threshold to determine if an attribute is present")
    arg_parser.add_argument("--val_percentage", type=float, default=0.1,
                            help="percentage of data used for validation set")
    arg_parser.add_argument("--test_percentage", type=float, default=0.1,
                            help="percentage of data used for test set")
    args = arg_parser.parse_args()

    generate_simple_yes_no_questions(data_dir=args.data_dir,
                                     is_present_threshold=args.is_present_threshold,
                                     use_two_thresholds=args.use_two_thresholds,
                                     val_percentage=args.val_percentage,
                                     test_percentage=args.test_percentage)





