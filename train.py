import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import vqa_memnet
from torch.utils.data import DataLoader
from birds.dataset import birdCaptionSimpleYesNoDataset, birdEmbeddedCaptionSimpleYesNoDataset


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="bAbI/data/tasks_1-20_v1-2/en/",
                        help='the path to the directory of the data')
    parser.add_argument("--batch_size", type=int, default=256,
                        help='the batch size for each training iteration using a variant of stochastic gradient descent')
    parser.add_argument("--text_latent_size", type=int, default=100,
                        help='the size of text embedding for question and evidence')
    parser.add_argument("--epochs", type=int, default=100,
                        help='the number of epochs to train for')
    parser.add_argument("--lr", type=float, default=0.001,
                        help='the starting learning rate for the optimizer')
    parser.add_argument("--max_clip", type=float, default=40.0,
                        help='the upperbound for the gradient (for gradient explosions)')
    parser.add_argument("--num_species", type=int, default=200,
                        help='the number of bird species')

    return parser.parse_args()


def load_data(batch_size, dataset_dir='/home/shenkev/School/VQA-Memnet/birds'):

    train_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=False, dataset_type="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=False)

    # val_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=False, dataset_type="val")
    # val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=1, shuffle=False)

    test_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=False, dataset_type="test")
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=False)

    vocabulary_size = len(train_data.word_idx)
    max_sentence_size = train_data.sentence_size
    print("Longest caption length", max_sentence_size)
    print("Vocabulary size", vocabulary_size)

    return train_loader, test_loader, vocabulary_size, max_sentence_size


def load_embedded_data(batch_size, dataset_dir='/home/shenkev/School/VQA-Memnet/birds',
                       embedding_dir='/home/shenkev/School/VQA-Memnet/encoder/mean_embeddings'):
    train_data = birdEmbeddedCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, embedding_dir=embedding_dir,
                                                       dataset_type="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=False)

    test_data = birdEmbeddedCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, embedding_dir=embedding_dir,
                                                      dataset_type="test")
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=False)

    vocabulary_size = len(train_data.word_idx)
    max_sentence_size = train_data.sentence_size
    print("Longest question length", max_sentence_size)
    print("Vocabulary size", vocabulary_size)

    caption_embeddings = to_var(torch.from_numpy(train_data.caption_embeddings))

    return train_loader, test_loader, caption_embeddings, vocabulary_size, max_sentence_size


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def load_model(caption_embeddings, num_species, vocabulary_size, text_latent_size):
    specie_latent_size = 20
    net = vqa_memnet(caption_embeddings, num_species, vocabulary_size, text_latent_size, specie_latent_size)
    if torch.cuda.is_available():
        net.cuda()
    return net


def save_weights(net, path):
    torch.save(net.state_dict(), path)


def load_weights(net, path):
    net.load_state_dict(torch.load(path))

    def evaluate(self, data="test"):
        correct = 0
        loader = self.train_loader if data == "train" else self.test_loader
        for step, (story, query, answer) in enumerate(loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)

            if self.config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            pred_prob = self.mem_n2n(story, query)[1]
            pred = pred_prob.data.max(1)[1] # max func return (max, argmax)
            correct += pred.eq(answer.data).cpu().sum()

        acc = float(correct) / len(loader.dataset)
        return acc


def step(net, optimizer, criterion, question_species, question, answer, i):
    optimizer.zero_grad()
    output = net(question_species, question)
    _, answer = torch.max(answer, 1)
    loss = criterion(output, answer)
    loss.backward()
    gradient_noise_and_clip(net.parameters())
    optimizer.step()

    # _, output_max_index = torch.max(output, 1)
    # num_correct = (output_max_index == answer).sum().data[0]

    # if i % 10 == 0:
    #     _, output_max_index = torch.max(output, 1)
    #     num_correct = (output_max_index == answer).sum().data[0]
    #     print('Step : ' + str(i) + ', Accuracy : ' + str(num_correct/256.0))

    return loss.data[0]


def gradient_noise_and_clip(parameters, noise_stddev=1e-4, max_clip=40.0):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    nn.utils.clip_grad_norm(parameters, max_clip)

    for p in parameters:
        noise = torch.randn(p.size()) * noise_stddev
        if torch.cuda.is_available():
            noise = noise.cuda()
        p.grad.data.add_(noise)


def train(epochs, train_loader, test_loader, net, optimizer, criterion):
    for epoch in range(epochs):

        epoch_loss = 0
        for i, (question_species, question, answer) in enumerate(train_loader):
            question_species = to_var(question_species)
            question = to_var(question)
            answer = to_var(answer)

            epoch_loss += step(net, optimizer, criterion, question_species, question, answer, i)

        if (epoch + 1) % 1 == 0:
            train_acc = evaluate(net, train_loader)
            test_acc = evaluate(net, test_loader)
            print(epoch + 1, epoch_loss, train_acc, test_acc)


def evaluate(net, loader):
    correct = 0
    for step, (story, query, answer) in enumerate(loader):
        story = to_var(story)
        query = to_var(query)
        answer = to_var(answer)

        output = net(story, query)
        _, output_max_index = torch.max(output, 1)
        _, answer = torch.max(answer, 1)
        correct += (answer == output_max_index).float().sum()  # really weird, without float() this counter resets to 0

    acc = float(correct.data[0]) / len(loader.dataset)
    return acc


if __name__ == "__main__":

    config = parse_config()
    learn_rate = config.lr
    batch_size = config.batch_size
    text_latent_size = config.text_latent_size
    epochs = config.epochs
    num_species = config.num_species

    weight_path = './Model/vqamemnet.pkl'

    train_loader, test_loader, caption_embeddings, vocabulary_size, words_in_sentence = load_embedded_data(batch_size)

    net = load_model(caption_embeddings, num_species, vocabulary_size, text_latent_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

    train(epochs, train_loader, test_loader, net, optimizer, criterion)
