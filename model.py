import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable


def embed_question(question, embed_layer, position_encoding):
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


def final_prediction(features, weights):
    return features.matmul(weights)

'''
 Args:
     evidence: [N * s * w] where s = number of sentences, w = number of words per sentence
     question: [N * w]

 Return:
     - vector of class activations
'''


class vqa_memnet(nn.Module):

    def __init__(self, vocabulary_size, text_latent_size, words_in_question):
        super(vqa_memnet, self).__init__()

        self.position_encoding = get_position_encoding(words_in_question, text_latent_size)

        # self.temporal_enc1 = Parameter(torch.Tensor(num_of_evidences, text_latent_size))
        # self.temporal_enc2 = Parameter(torch.Tensor(num_of_evidences, text_latent_size))
        # padding_idx=0 is required or else the 0 words (absence of a word) gets mapped to garbage
        self.evidence_emb = nn.Embedding(vocabulary_size, text_latent_size, padding_idx=0)
        self.question_emb = nn.Embedding(vocabulary_size, text_latent_size, padding_idx=0)

        # weight initialization greatly helps convergence
        # self.temporal_enc1.data.normal_(0, 0.1)
        # self.temporal_enc2.data.normal_(0, 0.1)
        self.evidence_emb.weight.data.normal_(0, 0.1)
        self.question_emb.weight.data.normal_(0, 0.1)

        self.softmax = nn.Softmax()

    def forward(self, evidence, question):

        question_emb = embed_question(question, self.question_emb, self.position_encoding)
        evidence_feature_emb, evidence_computation_emb \
            = embed_evidence(evidence, self.question_emb, self.evidence_emb, self.position_encoding)

        # evidence_feature_emb = evidence_feature_emb + self.temporal_enc1
        # evidence_computation_emb = evidence_computation_emb + self.temporal_enc2

        weights = compute_evidence_weights(evidence_computation_emb, question_emb)
        weighted_evidence = mean_pool(evidence_feature_emb, weights)

        # features = torch.cat((weighted_evidence, question_emb.squeeze(0)))
        features = weighted_evidence + question_emb.squeeze(0)
        output = final_prediction(features, self.evidence_emb.weight.transpose(0, 1))
        return output
