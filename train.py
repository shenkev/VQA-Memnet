import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import vqa_memnet
from torch.utils.data import DataLoader
from bAbI.dataset import bAbIDataset


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="bAbI/data/tasks_1-20_v1-2/en/",
                        help='the path to the directory of the data')
    parser.add_argument("--task", type=int, default=15,
                        help='the task number for bAbI')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='the batch size for each training iteration using a variant of stochastic gradient descent')
    parser.add_argument("--text_latent_size", type=int, default=25,
                        help='the size of text embedding for question and evidence')
    parser.add_argument("--epochs", type=int, default=100,
                        help='the number of epochs to train for')
    parser.add_argument("--lr", type=float, default=0.001,
                        help='the starting learning rate for the optimizer')
    parser.add_argument("--max_clip", type=float, default=40.0,
                        help='the upperbound for the gradient (for gradient explosions)')

    return parser.parse_args()


def load_data(task, batch_size, dataset_dir='./bAbI/data/tasks_1-20_v1-2/en/'):

    train_data = bAbIDataset(dataset_dir, task)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)

    test_data = bAbIDataset(dataset_dir, task, train=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=False)

    print("Longest sentence length", train_data.sentence_size)
    print("Longest story length", train_data.max_story_size)
    print("Average story length", train_data.mean_story_size)
    print("Number of vocab", train_data.num_vocab)

    return train_loader, test_loader, train_data.num_vocab, train_data.max_story_size, train_data.sentence_size


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def load_model(vocabulary_size, text_latent_size, num_of_evidences, words_in_sentence):
    net = vqa_memnet(vocabulary_size, text_latent_size, num_of_evidences, words_in_sentence)
    if torch.cuda.is_available():
        net.cuda()
    return net


def save_weights(net, path):
    torch.save(net.state_dict(), path)


def load_weights(net, path):
    net.load_state_dict(torch.load(path))


def step(net, optimizer, criterion, evidence, question, answer, step_num):
    optimizer.zero_grad()
    output = net(evidence, question)
    loss = criterion(output, answer)
    loss.backward()
    optimizer.step()

    if (step_num + 1) % 100 == 0:
        _, output_max_index = torch.max(output, 1)
        accuracy = (answer == output_max_index).float().mean()
        print ('Step [%d], Loss: %.4f, Acc: %.2f' % (step_num + 1, loss.data[0], accuracy.data[0]))


def train(epochs, data_loader, net, optimizer, criterion):
    for epoch in range(epochs):

        for i, (evidence, question, answer) in enumerate(data_loader):
            evidence = to_var(evidence)
            question = to_var(question)
            answer = to_var(answer)

            step(net, optimizer, criterion, evidence, question, answer, epoch*32 + i)


if __name__ == "__main__":

    config = parse_config()
    learn_rate = config.lr
    batch_size = config.batch_size
    text_latent_size = config.text_latent_size
    task = config.task
    epochs = config.epochs

    weight_path = './Model/vqamemnet.pkl'

    train_loader, test_loader, vocabulary_size, num_of_evidences, words_in_sentence = load_data(task, batch_size)

    net = load_model(vocabulary_size, text_latent_size, num_of_evidences, words_in_sentence)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

    train(epochs, train_loader, net, optimizer, criterion)

