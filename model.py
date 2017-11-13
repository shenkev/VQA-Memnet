import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
import random
from dominate.tags import *


'''
 Args:
     evidence: [N * s * w] where s = number of sentences, w = number of words per sentence
     question: [N * w]

 Return:
     - vector of class activations
'''


class vqa_memnet(nn.Module):

    def __init__(self, bin_vec_len, text_latent_size, attention_temperature):
        super(vqa_memnet, self).__init__()

        self.attention_temperature = attention_temperature

        self.fc1 = nn.Linear(2*bin_vec_len, 2)

        self.softmax = nn.Softmax()

    def forward(self, evidence, question, _body=None, iter=None, answer=None):

        weights = self.compute_evidence_weights(evidence, question)
        weighted_evidence = self.mean_pool(evidence, weights)

        features = torch.cat((weighted_evidence, question), 1)
        # features = weighted_evidence + question

        if _body is not None and (iter) % 100 == 0:

            randind = random.randint(0, evidence.size(0)-1)
            _, question_num = torch.max(question[randind], 0)
            question_num = question_num.data[0]
            np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

            _table = table(tr(th('Clue #'), th('Attention weight'),
                              th('Clue contains attribute'), th('Clue attribute numbers')))
            print_out = []
            for i in range(evidence.size(1)):
                att_weight = weights[randind, i].data[0]
                clue_has_att = abs(evidence[randind, i, question_num].data[0] - 1.0) <= 0.001
                clue_atts = np.where(to_np(evidence[randind, i]) == 1)[0]

                print_out.append((att_weight, i+1, clue_has_att, clue_atts))

            print_out = sorted(print_out, reverse=True)
            _table.add(tr(td("Final feature to FC: " + str(to_np(features[randind])), colspan="4")))
            for p in print_out:
                _table.add(tr(td(p[1]), td("{0:.4f}".format(p[0])), td(p[2]), td(', '.join(['%d' % n for n in p[3]]))))

            _body.add(
                div(h2("Attention at iteration {}".format(iter)),
                    h3("Question attribute number: {}, answer: {}".format(question_num, answer[randind].data[0])),
                    _table)
            )

        output = self.fc1(features)
        return output

    '''
     Args:
         e: [N * s * d]
         q: [N * d] where s is the # sentences, d is the dimension of the text's latent representation

     Return:
         - [N * s] weight probabilities
     '''

    def compute_evidence_weights(self, e, q):
        # e = e*(1/torch.clamp(torch.norm(e, 2, 2), 1)).unsqueeze(2)  # normalize evidence and question sentences
        # q = q*(1/torch.clamp(torch.norm(q, 2, 1), 1)).unsqueeze(1)  # clamp at 1 to prevent dividing by 0
        z = e.bmm(q.unsqueeze(2)).squeeze(2)
        softmax = nn.Softmax()
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