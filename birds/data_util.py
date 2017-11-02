import os
import re
import string

import pdb

# Python 2 version
# class punctuation_stripper(object):

#     def __init__(self):
#         self.to_space = re.compile('(-|_|::|/)')
#         # self.to_remove = string.maketrans('', '')

#     def strip(self, sentence):
#         return (self.to_space.sub(' ', sentence).translate(None, string.punctuation))

# Python 3 Version
class punctuation_stripper(object):

    def __init__(self):
        self.to_space = re.compile('(-|_|::|/)')
        self.to_remove = str.maketrans('', '', string.punctuation)

    def strip(self, sentence):
        return (self.to_space.sub(' ', sentence).translate(self.to_remove))


def load_captions(dataset_dir):
    '''
    grabs all captions in .txt files in data_dir/captions/
    returns a dictionary of the form: {<species_id>: <captions>}
       	<species_id> - integer representing the species, 0 indicates all species
           <captions> - tuples of the form (<species_id>, <caption_words>)
               <species_id> - integer representing the species
               <caption_words> - list of strings corresponding to values in words in the caption with punctuation removed    
    '''

    ps = punctuation_stripper()    
    captions_dir = os.path.join(dataset_dir, 'captions')    
    
    captions = {}
    subdirectories = [(int(subdirectory.split('.')[0]), os.path.join(captions_dir, subdirectory)) 
                       for subdirectory in os.listdir(captions_dir)
                       if os.path.isdir(os.path.join(captions_dir, subdirectory))]
    
    for (species_id, subdirectory) in subdirectories:
        species_captions = []
        caption_files = [(o, os.path.join(subdirectory, o)) for o in os.listdir(subdirectory) if '.txt' in o]

        for (file_name, path_to_file) in caption_files:
            lines = []
            with open(path_to_file) as f:
                lines = f.readlines()
            
            for line in lines:
                species_captions.append((species_id, ps.strip(line).split()))
    
        captions[species_id] = species_captions

    return captions

def load_attribute_based_captions(dataset_dir):

    image_species_map_path = os.path.join(dataset_dir, "image_class_labels.txt")
    image_attributes_map_path = os.path.join(dataset_dir, "image_attribute_labels.txt")

    image_species_map = {}
    with open(image_species_map_path) as f:
        lines = f.readlines()
        for line in lines:
            image_id, species_id = int(line.split()[0]), int(line.split()[1])
            image_species_map[image_id] = species_id

    image_attributes_map = {}
    with open(image_attributes_map_path) as f:
        lines = f.readlines()
        for line in lines:
            image_id, attribute_id, is_present = int(line.split()[0]), int(line.split()[1]), int(line.split()[2]) 
            if image_id not in image_attributes_map:
                image_attributes_map[image_id] = []
            if is_present == 1:
                image_attributes_map[image_id].append(attribute_id)

    attribute_id_text_map = load_attributes(dataset_dir)

    captions = {}
    for image_id, species_id in image_species_map.items():
        if species_id not in captions:
            captions[species_id] = []
    
    for image_id, attributes_list in image_attributes_map.items():
        caption = []
        for attribute_id in attributes_list:
            caption = caption + attribute_id_text_map[attribute_id][1:]
        
        species_id = image_species_map[image_id]
        captions[species_id].append((species_id,caption))

    return captions


def load_attributes(dataset_dir):
    '''
    returns dictionary of the form: {<attribute_id>, <attribute_text>}
        <attribute_id> - integer representing the species
        <attribute_words> - list of strings corresponding to values in words in the caption with punctuation removed
    '''

    ps = punctuation_stripper()

    attributes_path = os.path.join(dataset_dir, "attributes.txt")

    lines = []
    with open(attributes_path) as f:
        lines = f.readlines()

    attributes = {}
    for line in lines:
        attribute = ps.strip(line).split()
        attributes[int(attribute[0])] = attribute[1:]

    return attributes


def load_simple_yes_no_qa_pairs_helper(qa_path, attributes):
    '''
    returns a list tuples of the form (<species_id>, <question>, <answer>)
        <species_id> - integer representing the species
        <question_words> - list of strings corresponding to values in words in the question with punctuation removed
        <answer> - 0 or 1 corresponding to True or False
    '''

    with open(qa_path) as f:
        qa_pairs = [list(map(int,line.split())) for line in f.readlines()]

    for qa_pair in qa_pairs:
        qa_pair[1] = attributes[qa_pair[1]]

    max_questions = 200
    qa_pairs.sort(key= lambda qa_pair: qa_pair[2])
    end_of_ones_idx = next(i for i, v in enumerate(qa_pairs) if v[2]==1)
    qa_pairs = qa_pairs[0:min(end_of_ones_idx, int(max_questions/2))] \
               + qa_pairs[end_of_ones_idx: min(2*end_of_ones_idx, end_of_ones_idx+int(max_questions/2))]
 

    # Python 2 Version
    # def answer_compare(a, b):
    #     if a[2] > b[2]:
    #         return -1
    #     else:
    #         return 1

    # # TODO fix this up, this is hacky
    # qa_pairs.sort(cmp=answer_compare)
    # end_of_ones_idx = (i for i, v in enumerate(qa_pairs) if v[2]==0).next()
    # qa_pairs = qa_pairs[0:min(end_of_ones_idx, max_questions/2)] \
    #            + qa_pairs[end_of_ones_idx: min(2*end_of_ones_idx, end_of_ones_idx+max_questions/2)]


    return [tuple(qa_pair) for qa_pair in qa_pairs]


def load_simple_yes_no_qa_pairs(dataset_dir):
    '''
    returns three lists of tuples: qa_train, qa_val, and qa_test
    each tuple takes the form: (<species_id>, <question>, <answer>)
       	<species_id> - integer representing the species
        <question_words> - list of strings corresponding to values in words in the question with punctuation removed
        <answer> - 0 or 1 corresponding to True or False
    '''

    train_path = os.path.join(dataset_dir, "simple_yes_no_train.txt")
    val_path = os.path.join(dataset_dir, "simple_yes_no_val.txt")
    test_path = os.path.join(dataset_dir, "simple_yes_no_test.txt")
    
    attributes = load_attributes(dataset_dir)  

    qa_train = load_simple_yes_no_qa_pairs_helper(train_path, attributes)
    qa_val = load_simple_yes_no_qa_pairs_helper(val_path, attributes)
    qa_test = load_simple_yes_no_qa_pairs_helper(test_path, attributes)    
    return qa_train, qa_val, qa_test    

def get_common_parameters(captions, qa_pairs):

    words = set()

    for species, species_captions in captions.items():
        for caption in species_captions:
            words.update(caption[1])

    for qa_pair in qa_pairs:
        words.update(qa_pair[1])

    vocab = sorted(list(words))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    sentence_size = max([max([len(caption[1]) for caption in species_captions]) for species_id, species_captions in captions.items()])
    question_size = max([len(qa_pair[1]) for qa_pair in qa_pairs])
    sentence_size = max(question_size, sentence_size)

    # buckets = [0]*66
    # for species_id, species_captions in captions.items():
    #     for caption in species_captions:
    #         idx = len(caption[1])-1
    #         buckets[idx] = buckets[idx] + 1
    #
    # print(buckets)

    return word_idx, sentence_size

def vectorize_captions(captions, word_idx, sentence_size):

    captions_per_species = [len(species_captions) for species_id, species_captions in captions.items()]
    max_captions_per_species = max(captions_per_species)
    total_captions = sum(captions_per_species)

    captions_vec = {}
    all_captions = []
    for species_id, species_captions in captions.items():
        species_captions_vec = []
        for caption in species_captions:
            sentence_pad_length = max(0, sentence_size - len(caption[1]))
            species_captions_vec.append((caption[0], [word_idx[w] for w in caption[1]] + [0] * sentence_pad_length))

        all_captions.extend(species_captions_vec)

        memory_pad_length = max(0, max_captions_per_species - len(species_captions_vec))
        for _ in range(memory_pad_length):
            species_captions_vec.append((0, [0] * sentence_size))

        captions_vec[species_id] = species_captions_vec

    captions_vec[0] = all_captions

    return captions_vec


def vectorize_qa_pairs(qa_pairs, word_idx, sentence_size):

    qa_pairs_vec = []
    for qa_pair in qa_pairs:
        species_id, question_words, answer = qa_pair

        sentence_pad_length = max(0, sentence_size - len(question_words))
        question_vec = [word_idx[w] for w in question_words] + [0] * sentence_pad_length

        if answer:
            answer_vec = [0,1]
        else:
            answer_vec = [1,0]

        qa_pairs_vec.append((species_id, question_vec, answer_vec))

    return qa_pairs_vec






