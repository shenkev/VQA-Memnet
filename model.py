import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
import random
import math
from dominate.tags import *


'''
 Args:
     evidence: [N * s * w] where s = number of sentences, w = number of words per sentence
     question: [N * w]

 Return:
     - vector of class activations
'''


class vqa_memnet(nn.Module):

    def __init__(self, vocab_size, text_latent_size, clues_per_specie):
        super(vqa_memnet, self).__init__()

        # padding_idx=0 is required or else the 0 words (absence of a word) gets mapped to garbage
        self.evidence_emb = nn.Embedding(vocab_size + 1, text_latent_size, padding_idx=0)
        self.question_emb = nn.Embedding(vocab_size + 1, text_latent_size, padding_idx=0)

        # weight initialization greatly helps convergence
        self.evidence_emb.weight.data.normal_(0, 0.1)
        self.question_emb.weight.data.normal_(0, 0.1)

        # For maximum attention weight on 1 clue of z, y clues, the temperature is x=ln(z*(y-1)/(1-z))
        self.attention_temperature = 20*math.log(0.999*(clues_per_specie)/(1-0.999))

        self.fc1 = nn.Linear(text_latent_size, 2)

        self.softmax = nn.Softmax()

    def forward(self, evidence, question, _body=None, iter=None, answer=None):

        question_emb = self.embed_question(question, self.question_emb)
        evidence_feature_emb, evidence_computation_emb = self.embed_evidence(evidence, self.question_emb, self.evidence_emb)

        weights = self.compute_evidence_weights(evidence_computation_emb, question_emb)
        weighted_evidence = self.mean_pool(evidence_feature_emb, weights)

        # features = torch.cat((weighted_evidence, question_emb), 1)
        features = weighted_evidence + question_emb

        # ========================= Logging ========================= #
        if _body is not None and (iter) % 100 == 0:

            randind = random.randint(0, evidence.size(0)-1)
            question_num = question[randind, 0].data[0]
            np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

            _table = table(tr(th('Clue #'), th('Attention weight'),
                              th('Clue contains attribute'), th('Clue attribute numbers')))
            print_out = []
            for i in range(evidence.size(1)):
                att_weight = weights[randind, i].data[0]
                clue_has_att = question_num in to_np(evidence[randind, i])
                clue_atts = evidence[randind, i]

                print_out.append((att_weight, i+1, clue_has_att, clue_atts))

            print_out = sorted(print_out, reverse=True)
            _table.add(tr(td("Final feature to FC: " + str(to_np(features[randind])), colspan="4")))
            for p in print_out:
                _table.add(tr(td(p[1]), td("{0:.4f}".format(p[0])), td(p[2]),
                              td(', '.join(['%d' % n for n in to_np(p[3])]))))

            _body.add(
                div(h2("Attention at iteration {}".format(iter)),
                    h3("Question attribute number: {}, answer: {}".format(question_num, answer[randind].data[0])),
                    _table)
            )
            # ========================= Logging ========================= #

        # output = torch.matmul(features, self.evidence_emb.weight.transpose(0, 1))
        output = self.fc1(features)
        return output

    def subfinder(self, mylist, pattern):
        matches = []
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                matches.append(pattern)
        return matches

    def embed_question(self, question, embed_layer):
        question_emb = embed_layer(question)

        return question_emb.sum(1)

    def embed_evidence(self, evidence, question_embed_layer, evidence_embed_layer):
        expanded_evidence = evidence.view(-1, evidence.size(2))  # combine batch and clue dimensions

        evidence_for_computation = question_embed_layer(expanded_evidence)
        evidence_for_computation = evidence_for_computation.sum(1)
        evidence_for_computation = evidence_for_computation.view(-1, evidence.size(1),
                                                                 evidence_embed_layer.embedding_dim)

        evidence_features = evidence_embed_layer(expanded_evidence)
        evidence_features = evidence_features.sum(1)
        evidence_features = evidence_features.view(-1, evidence.size(1), evidence_embed_layer.embedding_dim)

        return evidence_features, evidence_for_computation

    '''
     Args:
         e: [N * s * d]
         q: [N * d] where s is the # sentences, d is the dimension of the text's latent representation

     Return:
         - [N * s] weight probabilities
     '''

    def compute_evidence_weights(self, e, q):
        # e = e*(1/torch.clamp(torch.norm(e, 2, 2), min=0.1)).unsqueeze(2)  # normalize evidence and question sentences
        # q = q*(1/torch.clamp(torch.norm(q, 2, 1), min=0.1)).unsqueeze(1)  # clamp at 0.1 to prevent dividing by 0
        z = e.bmm(q.unsqueeze(2)).squeeze(2)
        softmax = nn.Softmax()
        z = softmax(z)
        z = softmax(self.attention_temperature * z)
        return z

    ''' mean pools evidence items
     Args:
         x: [N * s * d]
         weights: [N * s]

     Return:
         - [N * d]
     '''

    def mean_pool(self, x, weights):
        z = x * weights.unsqueeze(2)
        z = z.sum(1)
        return z


def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(pattern)
    return matches


def filter_zero(x):
    return [e for e in x if e != 0]


def to_np(x):
    return x.data.cpu().numpy()