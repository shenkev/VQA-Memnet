import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable


def embed_question(question, embed_layer, position_encoding):
    question_emb = embed_layer(question)

    question_emb = question_emb * position_encoding

    return question_emb.sum(1)


def embed_evidence(evidence, question_embed_layer, evidence_embed_layer, position_encoding):
    expanded_evidence = evidence.view(-1, evidence.size(2))  # combine batch and sentence dimensions

    evidence_for_computation = question_embed_layer(expanded_evidence)
    evidence_for_computation = evidence_for_computation * position_encoding
    evidence_for_computation = evidence_for_computation.sum(1)
    evidence_for_computation = evidence_for_computation.view(-1, evidence.size(1), position_encoding.size(2))
    # batch X sentences X embedding_size

    evidence_features = evidence_embed_layer(expanded_evidence)
    evidence_features = evidence_features * position_encoding
    evidence_features = evidence_features.sum(1)
    evidence_features = evidence_features.view(-1, evidence.size(1), position_encoding.size(2))
    # batch X sentences X embedding_size

    return evidence_features, evidence_for_computation


def get_position_encoding(words_in_sentence, text_latent_size):

    encoding = np.zeros((words_in_sentence, text_latent_size), dtype=np.float32)

    for i in range(0, words_in_sentence):
        for j in range(0, text_latent_size):
            encoding[i, j] = (1-i/words_in_sentence) - (j/text_latent_size)*(1-(2*i)/words_in_sentence)

    encoding = torch.from_numpy(encoding)
    return Variable(encoding.unsqueeze(0))

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

'''Predict the final answer
 Args:
     x: concatenated evidence and question [N x e+q]

 Return:
     - vector of class activations
'''


class final_prediction(nn.Module):

    def __init__(self, text_latent_size, output_size):
        super(final_prediction, self).__init__()

        self.fc1 = nn.Linear(text_latent_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        z = self.fc1(x)
        z = self.softmax(z)  # not sure if you need to softmax before cross-entropy loss
        return z

'''
 Args:
     evidence: [N * V]
     question: [1 * V]

 Return:
     - vector of class activations
'''


class vqa_memnet(nn.Module):

    def __init__(self, vocabulary_size, text_latent_size, num_of_evidences, words_in_question):
        super(vqa_memnet, self).__init__()

        self.position_encoding = get_position_encoding(words_in_question, text_latent_size)

        self.temporal_enc1 = Parameter(torch.Tensor(num_of_evidences, text_latent_size))
        self.temporal_enc2 = Parameter(torch.Tensor(num_of_evidences, text_latent_size))
        self.evidence_emb = nn.Embedding(vocabulary_size, text_latent_size)
        self.question_emb = nn.Embedding(vocabulary_size, text_latent_size)
        self.answer = final_prediction(text_latent_size, vocabulary_size)

    def forward(self, evidence, question):

        question_emb = embed_question(question, self.question_emb, self.position_encoding)
        evidence_feature_emb, evidence_computation_emb \
            = embed_evidence(evidence, self.question_emb, self.evidence_emb, self.position_encoding)

        evidence_feature_emb = evidence_feature_emb + self.temporal_enc1
        evidence_computation_emb = evidence_computation_emb + self.temporal_enc2

        weights = compute_evidence_weights(evidence_computation_emb, question_emb)
        weighted_evidence = mean_pool(evidence_feature_emb, weights)

        # features = torch.cat((weighted_evidence, question_emb.squeeze(0)))
        features = weighted_evidence + question_emb.squeeze(0)

        return self.answer(features)


# from torch.autograd import Variable
# evidence = Variable(torch.rand(2, 3))
# question = Variable(torch.rand(1, 3))
#
# layer = vqa_memnet(3, 3, 2, 2)
# out = layer(evidence, question)
# loss = torch.sum(out)
# loss.backward()
#
# params = next(layer.parameters())
# print(out)
# print(params.grad)