from random import randint
import matplotlib

import numpy as np
import torch

import glob
import os

GLOVE_PATH = './dataset/glove.840B.300d.txt'

# make sure models.py is in the working directory
# model = torch.load('infersent.allnli.pickle')
# model = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage.cuda(0))
# This doesn't work unless you specify the GPU https://github.com/facebookresearch/SentEval/issues/3
model = torch.load('infersent.allnli.pickle', map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
# torch.load(..pickle) will use GPU/Cuda by default. If you are on CPU:
# model = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
# On CPU, setting the right number of threads with "torch.set_num_threads(k)" may improve performance

model.set_glove_path(GLOVE_PATH)

# number of works in vocab affects how many words are thrown out of captions
vocab_size = 1000000
model.build_vocab_k_words(K=vocab_size)

path_to_captions = '../birds/captions/*/'
specie_paths = glob.glob(path_to_captions)
specie_paths = sorted(specie_paths)

for i in range(len(specie_paths)):
    sentences = []
    path = specie_paths[i]
    specie = os.path.basename(os.path.normpath(path))
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(filename) as f:
            for line in f:
                sentences.append(line.strip())
    print("=============================" + specie + "=============================")
    print('Num of captions : {0}'.format(len(sentences)))
    embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
    print('nb sentences encoded : {0}'.format(len(embeddings)))
    print('embedding size : {0}'.format(embeddings.shape[1]))

    # just take average embedding
    mean_embedding = np.mean(embeddings, axis=0)
    np.save('mean_embeddings/' + specie, mean_embedding)

    # fit and sample from univariate gaussian
    embedding_mean = np.mean(embeddings, axis=0)
    embedding_cov = np.cov(embeddings, rowvar=0)
    n = 10
    samples = np.random.multivariate_normal(mean_embedding, embedding_cov, n)
    np.save('sampled_embeddings/' + specie, samples)