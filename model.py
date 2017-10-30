import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable


def embed_specie(specie, embed_layer):
    question_emb = embed_layer(specie)

    return question_emb


def embed_question(question, embed_layer):
    question_emb = embed_layer(question)

    # question_emb = question_emb * position_encoding

    return question_emb.sum(1)


def embed_evidence(evidence, question_embed_layer, evidence_embed_layer, position_encoding):
    expanded_evidence = evidence.view(-1, evidence.size(2))  # combine batch and sentence dimensions

    evidence_for_computation = question_embed_layer(expanded_evidence)
    # evidence_for_computation = evidence_for_computation * position_encoding
    evidence_for_computation = evidence_for_computation.sum(1)
    evidence_for_computation = evidence_for_computation.view(-1, evidence.size(1), position_encoding.size(2))

    evidence_features = evidence_embed_layer(expanded_evidence)
    # evidence_features = evidence_features * position_encoding
    evidence_features = evidence_features.sum(1)
    evidence_features = evidence_features.view(-1, evidence.size(1), position_encoding.size(2))

    return evidence_features, evidence_for_computation


def get_position_encoding(words_in_sentence, text_latent_size):

    encoding = np.zeros((words_in_sentence, text_latent_size), dtype=np.float32)

    for i in range(0, words_in_sentence):
        for j in range(0, text_latent_size):
            encoding[i, j] = (1-(i+1)/words_in_sentence) - ((j+1)/text_latent_size)*(1-(2*(i+1))/words_in_sentence)

    encoding = torch.from_numpy(encoding)
    return Variable(encoding.unsqueeze(0), requires_grad=False)

'''
 Args:
     e: [N * s * d]
     q: [N * d] where s is the # sentences, d is the dimension of the text's latent representation

 Return:
     - [N * s] weight probabilities
 '''


def compute_evidence_weights(e, q):
    z = e.bmm(q.unsqueeze(2)).squeeze(2)
    softmax = nn.Softmax()
    z = softmax(z)
    return z

''' mean pools evidence items
 Args:
     x: [N * s * d]
     weights: [N * s]

 Return:
     - same as x but each item x_i is weighted by w_i
 '''


def mean_pool(x, weights):
    z = x*weights.unsqueeze(2)
    z = z.sum(1)
    return z


def load_captions(dir='./encoder/pca/pca100_embeddings.npy'):
    return np.load(dir)


'''
 Args:
     evidence: [N * s * w] where s = number of sentences, w = number of words per sentence
     question: [N * w]

 Return:
     - vector of class activations
'''


class vqa_memnet(nn.Module):

    def __init__(self, caption_embeddings, num_species, vocabulary_size, text_latent_size, specie_latent_size):
        super(vqa_memnet, self).__init__()

        # self.caption_embeddings = caption_embeddings  # size 200x4096
        # cap_emb = torch.randn(200, 4096)
        cap_emb = torch.from_numpy(load_captions())
        self.caption_embeddings = Variable(cap_emb.cuda())
        self.num_species = num_species

        # padding_idx=0 is required or else the 0 words (absence of a word) gets mapped to garbage
        # need to be careful with this +1 to embedding dimension. we're counting from 1 instead of 0 with our data
        self.species_emb = nn.Embedding(num_species + 1, specie_latent_size)
        self.question_emb = nn.Embedding(vocabulary_size + 1, text_latent_size, padding_idx=0)
        self.fc1 = nn.Linear(text_latent_size, 2)

        # weight initialization greatly helps convergence
        self.species_emb.weight.data.normal_(0, 0.1)
        self.question_emb.weight.data.normal_(0, 0.1)

        self.softmax = nn.Softmax()

    def forward(self, question_species, question):

        # it may be redundant to be computing all the specie embeddings every loop
        all_species = torch.LongTensor(range(1, self.num_species+1))

        if torch.cuda.is_available():
            all_species = all_species.cuda()

        all_specie_emb = embed_specie(Variable(all_species), self.species_emb)

        question_emb = embed_question(question, self.question_emb)
        question_species_emb = embed_specie(question_species, self.species_emb)

        specie_caption_emb = torch.cat((all_specie_emb, self.caption_embeddings), 1)
        specie_question_emb = torch.cat((question_species_emb, question_emb), 1)

        weights = compute_evidence_weights(
            specie_caption_emb.expand(
                specie_question_emb.size(0),
                specie_caption_emb.size(0),
                specie_caption_emb.size(1)
            ),
            specie_question_emb)
        # _, test = torch.max(weights, 1)
        weighted_evidence = mean_pool(self.caption_embeddings, weights)

        # features = torch.cat((weighted_evidence, question_emb.squeeze(0)))
        features = weighted_evidence + question_emb.squeeze(0)
        output = self.fc1(features)
        return output
