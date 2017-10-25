import functools
import os
import re
from functools import reduce
from itertools import chain

import numpy as np
import torch
import string
from sklearn import model_selection
from sklearn.metrics import jaccard_similarity_score
from torch.autograd import Variable

import pdb

long_tensor_type = torch.LongTensor
float_tensor_type = torch.FloatTensor

if (torch.cuda.is_available()):
    long_tensor_type = torch.cuda.LongTensor
    float_tensor_type = torch.cuda.FloatTensor


def process_data(args):
    # test_size = .1
    # random_state = None
    captions, questions, answers, vocab = load_data(args.data_dir, args.is_present_threshold)

    sentence_size, vocab_size, word_idx = calculate_parameter_values(captions, questions, vocab)

    captions_vec, questions_vec, answers_vec = vectorize_data(caption=captions, questions=questions, answers=answers,
                                                              sentence_size=sentence_size, word_idx=word_idx)

    S, Q, A = generate_s_q_a(questions=questions_vec, answers=answers_vec, limit_to_species=args.limit_to_species)


    #return sentence_size, vocab_size, word_idx

    # train_set, train_batches, val_set, val_batches, test_set, test_batches = \
    #     vectorize_task_data(args.batch_size, data, args.debug, memory_size,
    #                         random_state, sentence_size, test_data, test_size, word_idx)

    # return train_batches, val_batches, test_batches, train_set, val_set, test_set, \
    #        sentence_size, vocab_size, memory_size, word_idx


class punctuation_stripper(object):

    def __init__(self):
        self.to_space = re.compile('(-|_|::|/)')
        self.to_remove = str.maketrans('', '', string.punctuation)

    def strip(self, sentence):
        return (self.to_space.sub(' ', sentence).translate(self.to_remove))

def load_data(data_dir, is_present_threshold):

    ps = punctuation_stripper()
    captions = load_captions(data_dir, ps)
    attributes = load_attributes(data_dir, ps)

    questions, answers = generate_questions(data_dir, attributes, is_present_threshold)

    vocab = get_vocab(captions, questions)

    return captions, questions, answers, vocab


def load_captions(captions_dir, ps):

    captions_dir = os.path.join(captions_dir, 'captions')

    all_captions = {}
    subdirectories = [(int(o.split('.')[0]), os.path.join(captions_dir, o)) for o in os.listdir(captions_dir) 
                    if os.path.isdir(os.path.join(captions_dir,o))]
    
    for (species, subdirectory) in subdirectories:
        species_captions = []
        caption_files = [(o, os.path.join(subdirectory, o)) for o in os.listdir(subdirectory) if '.txt' in o]

        for (file_name, path_to_file) in caption_files:
            lines = []
            with open(path_to_file) as f:
                lines = f.readlines()
            
            for line in lines:
                species_captions.append(ps.strip(line).split())
    
        all_captions[species] = species_captions

    return all_captions 

def load_attributes(attributes_dir, ps):

    attributes_path = os.path.join(attributes_dir, "attributes.txt")

    lines = []
    with open(attributes_path) as f:
        lines = f.readlines()

    attributes = {}
    for line in lines:
        attribute = ps.strip(line).split()
        attributes[int(attribute[0])] = attribute[1:]

    return attributes

def generate_questions(class_attributes_dir, attributes, is_present_threshold):

    class_attributes_path = os.path.join(class_attributes_dir, "class_attribute_labels_continuous.txt")

    lines = []
    with open(class_attributes_path) as f:
        lines = f.readlines()

    questions = {}
    answers = {}
    for species, line in enumerate(lines, 1):
        attribute_percentages = line.split()

        species_questions = []
        species_answers = []

        for attribute_id, present_percentage in enumerate(attribute_percentages, 1):
            species_questions.append(attributes[attribute_id])
            species_answers.append(is_present_threshold <= float(present_percentage))

        questions[species] = species_questions
        answers[species] = species_answers

    return questions, answers


def get_vocab(captions, questions):
    words = set()

    for species, species_captions in captions.items():
        for caption in species_captions:
            words.update(caption)

    for species, species_questions in questions.items():
        for question in species_questions:
            words.update(question)

    return sorted(list(words))


def calculate_parameter_values(captions, questions, vocab):
    
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    vocab_size = len(word_idx) + 1
    
    sentence_size = max([max([len(caption) for caption in species_captions]) for species_id, species_captions in captions.items()])
    question_size = max([max([len(question) for question in species_questions]) for species_id, species_questions in questions.items()])
    sentence_size = max(question_size, sentence_size)

    return sentence_size, vocab_size, word_idx


# def vectorize_task_data(batch_size, data, debug, memory_size, random_state, sentence_size, test,
#                         test_size, word_idx):
#     S, Q, Y = vectorize_data(data, word_idx, sentence_size, memory_size)

#     if debug is True:
#         print("S : ", S)
#         print("Q : ", Q)
#         print("Y : ", Y)
#     trainS, valS, trainQ, valQ, trainY, valY = model_selection.train_test_split(S, Q, Y, test_size=test_size,
#                                                                                 random_state=random_state)
#     testS, testQ, testY = vectorize_data(test, word_idx, sentence_size, memory_size)

#     if debug is True:
#         print(S[0].shape, Q[0].shape, Y[0].shape)
#         print("Training set shape", trainS.shape)

#     # params
#     n_train = trainS.shape[0]
#     n_val = valS.shape[0]
#     n_test = testS.shape[0]
#     if debug is True:
#         print("Training Size: ", n_train)
#         print("Validation Size: ", n_val)
#         print("Testing Size: ", n_test)
#     train_labels = np.argmax(trainY, axis=1)
#     test_labels = np.argmax(testY, axis=1)
#     val_labels = np.argmax(valY, axis=1)
#     n_train_labels = train_labels.shape[0]
#     n_val_labels = val_labels.shape[0]
#     n_test_labels = test_labels.shape[0]

#     if debug is True:
#         print("Training Labels Size: ", n_train_labels)
#         print("Validation Labels Size: ", n_val_labels)
#         print("Testing Labels Size: ", n_test_labels)

#     train_batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
#     val_batches = zip(range(0, n_val - batch_size, batch_size), range(batch_size, n_val, batch_size))
#     test_batches = zip(range(0, n_test - batch_size, batch_size), range(batch_size, n_test, batch_size))

#     return [trainS, trainQ, trainY], list(train_batches), [valS, valQ, valY], list(val_batches), [testS, testQ, testY], \
#            list(test_batches)


def vectorize_data(captions, questions, answers, sentence_size, word_idx):

    # Captions
    captions_per_species = [len(species_captions) for species_id, species_captions in captions.items()]
    max_captions_per_species = max(captions_per_species)
    total_captions = sum(captions_per_species)

    captions_vec = {}
    all_captions = []
    for species_id, species_captions in captions.items():
        species_captions_vec = []
        for caption in species_captions:
            sentence_pad_length = max(0, sentence_size - len(caption))
            species_captions_vec.append((species_id, [word_idx[w] for w in caption] + [0] * sentence_pad_length))

        all_captions.extend(species_captions_vec)

        memory_pad_length = max(0, max_captions_per_species - len(species_captions_vec))
        for _ in range(memory_pad_length):
            species_captions_vec.append((0, [0] * sentence_size))

        captions_vec[species_id] = species_captions_vec

    captions_vec[0] = all_captions

    # Questions
    questions_vec = {}
    for species_id, species_questions in questions.items():
        species_questions_vec = []
        for question in species_questions:
            sentence_pad_length = max(0, sentence_size - len(question))
            species_questions_vec.append((species_id, [word_idx[w] for w in question] + [0] * sentence_pad_length))

        questions_vec[species_id] =  species_questions_vec

    # Answers
    answers_vec = {}
    for species_id, species_answers in answers.items():
        species_answers_vec = []
        for answer in species_answers:
            if answer:
                species_answers_vec.append([0,1])
            else:
                species_answers_vec.append([1,0])

        answers_vec[species_id] =  species_answers_vec

    return captions_vec, questions_vec, answers_vec


def generate_s_q_a(questions, answers, limit_to_species):

    S = []
    Q = []
    A = []
    for species_id, species_questions in questions.items():
        for question in species_questions:
            S.append(species_id)
            Q.append(question)
    for species_id, species_answers in answers.items():
        for answer in species_answers:
            A.append(answer)

    if not limit_to_species:
        S = [0] * len(S)

    return S, Q, A

#return np.array(S), np.array(Q), np.array(A)

# Keeping this around since this could be a good way of filtering down our data
# Would result quite a bit of pre-processing, but still overall not a bad way to go
# if len(ss) > memory_s
#     # Use Jaccard similarity to determine the most relevant sentences
#     q_words = (q)
#     least_like_q = sorted(ss, key=functools.cmp_to_key(
#         lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
#                    :len(ss) - memory_size]
#     for sent in least_like_q:
#         # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
#         # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
#         ss.remove(sent)


# def generate_batches(batches_tr, batches_v, batches_te, train, val, test):
#     train_batches = get_batch_from_batch_list(batches_tr, train)
#     val_batches = get_batch_from_batch_list(batches_v, val)
#     test_batches = get_batch_from_batch_list(batches_te, test)

#     return train_batches, val_batches, test_batches


# def get_batch_from_batch_list(batches_tr, train):
#     trainS, trainQ, trainA = train
#     trainA, trainQ, trainS = extract_tensors(trainA, trainQ, trainS)
#     train_batches = []
#     train_batches = construct_s_q_a_batch(batches_tr, train_batches, trainS, trainQ, trainA)
#     return train_batches


# def extract_tensors(A, Q, S):
#     A = torch.from_numpy(A).type(long_tensor_type)
#     S = torch.from_numpy(S).type(float_tensor_type)
#     Q = np.expand_dims(Q, 1)
#     Q = torch.from_numpy(Q).type(long_tensor_type)
#     return A, Q, S


# def construct_s_q_a_batch(batches, batched_objects, S, Q, A):
#     for batch in batches:
#         # batches are of form : [(0,2), (2,4),...]
#         answer_batch = []
#         story_batch = []
#         query_batch = []
#         for j in range(batch[0], batch[1]):
#             answer_batch.append(A[j])
#             story_batch.append(S[j])
#             query_batch.append(Q[j])
#         batched_objects.append([story_batch, query_batch, answer_batch])

#     return batched_objects


# def process_eval_data(data_dir, task_num, word_idx, sentence_size, vocab_size, memory_size=50, batch_size=2,
#                       test_size=.1, debug=True, joint_training=0):
#     random_state = None
#     data, test, vocab = load_data(data_dir, joint_training, task_num)

#     if (joint_training == 0):
#         memory_size, sentence_size, vocab_size, word_idx = calculate_parameter_values(data=data, debug=debug,
#                                                                                       memory_size=memory_size,
#                                                                                       vocab=vocab)
#     train_set, train_batches, val_set, val_batches, test_set, test_batches = \
#         vectorize_task_data(batch_size, data, debug, memory_size, random_state,
#                             sentence_size, test, test_size, word_idx)

#     return train_batches, val_batches, test_batches, train_set, val_set, test_set, \
#            sentence_size, vocab_size, memory_size, word_idx


# def get_position_encoding(batch_size, sentence_size, embedding_size):
#     '''
#     Position Encoding 
#     '''
#     encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
#     ls = sentence_size + 1
#     le = embedding_size + 1
#     for i in range(1, le):
#         for j in range(1, ls):
#             encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
#     encoding = 1 + 4 * encoding / embedding_size / sentence_size
#     enc_vec = torch.from_numpy(np.transpose(encoding)).type(float_tensor_type)
#     lis = []
#     for _ in range(batch_size):
#         lis.append(enc_vec)
#     enc_vec = Variable(torch.stack(lis))
#     return enc_vec


# def weight_update(name, param):
#     update = param.grad
#     weight = param.data
#     print(name, (torch.norm(update) / torch.norm(weight)).data[0])