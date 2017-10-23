import torch
import torch.nn as nn


'''
 Args:
     x: [N * k]
     q: [1 * k] where k is the dimension of the text's latent representation

 Return:
     - [1* N] weight probabilities
 '''


def compute_evidence_weights(e, q):
    z = e.matmul(q.transpose(0, 1))
    z = z.view(1, z.size(0))
    m = nn.Softmax()
    return m(z)

''' mean pools evidence items
 Args:
     x: [N* l]
     weights: [1 * N]

 Return:
     - same as x but each item x_i is weighted by w_i
 '''


def mean_pool(x, weights):
    weights = weights.view(weights.size(1), 1)
    z = weights*x
    z = z.sum(0)
    return z

'''Predict the final answer
 Args:
     x: concatenated evidence and question [N x e+q]

 Return:
     - vector of class activations
'''


class final_prediction(nn.Module):

    def __init__(self, evidence_size, question_latent_size, output_size):
        super(final_prediction, self).__init__()

        hidden_sizes = [500]
        self.fc1 = nn.Linear(evidence_size + question_latent_size, hidden_sizes[0])
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], output_size)

    def forward(self, x):
        z = self.fc1(x)
        z = self.prelu1(z)
        z = self.fc2(z)
        return z

'''
 Args:
     evidence: [N * V]
     question: [1 * V]

 Return:
     - vector of class activations
'''


class vqa_memnet(nn.Module):

    def __init__(self, evidence_size, question_size, text_latent_size, output_size):
        super(vqa_memnet, self).__init__()

        self.evidence_emb = nn.Linear(evidence_size, text_latent_size)
        self.question_emb = nn.Linear(question_size, text_latent_size)
        self.answer = final_prediction(evidence_size, text_latent_size, output_size)

    def forward(self, evidence, question):

        z_e = self.evidence_emb(evidence)
        z_q = self.question_emb(question)

        w = compute_evidence_weights(z_e, z_q)
        weighted_evidence = mean_pool(evidence, w)

        features = torch.cat((weighted_evidence, z_q.squeeze(0)))

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