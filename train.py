import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import vqa_memnet
from torch.utils.data import DataLoader
from birds.dataset import birdCaptionSimpleYesNoDataset
from logger import Logger
import pdb

# Set the logger
run_name = 'threshold_80_ceil_0_floor'
logger = Logger('./logs/' + run_name)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="bAbI/data/tasks_1-20_v1-2/en/",
                        help='the path to the directory of the data')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='the batch size for each training iteration using a variant of stochastic gradient descent')
    parser.add_argument("--text_latent_size", type=int, default=114,
                        help='the size of text embedding for question and evidence')
    parser.add_argument("--epochs", type=int, default=1000,
                        help='the number of epochs to train for')
    parser.add_argument("--lr", type=float, default=0.001,
                        help='the starting learning rate for the optimizer')
    parser.add_argument("--max_clip", type=float, default=40.0,
                        help='the upperbound for the gradient (for gradient explosions)')

    return parser.parse_args()


def tensorboard_logging(batch_loss, train_acc, test_acc, net, iteration):
    # (1) Log the scalar values
    info = {
        'loss': batch_loss,
        'train accuracy': train_acc,
        'test accuracy': test_acc
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, iteration)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), iteration)
        if value.grad is not None:
            logger.histo_summary(tag + '/grad', to_np(value.grad), iteration)


def load_data(batch_size, dataset_dir='/Users/atef/VQA-Memnet/birds'):

    train_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=True, dataset_type="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)

    # val_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=False, dataset_type="val")
    # val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=1, shuffle=False)

    test_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=True, dataset_type="test")
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=False)

    vocabulary_size = len(train_data.word_idx)
    caption_length = train_data.sentence_size
    print("Longest caption length", caption_length)
    print("Number of vocab", vocabulary_size)

    return train_loader, test_loader, vocabulary_size, caption_length


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def load_model(vocabulary_size, text_latent_size, words_in_sentence):
    net = vqa_memnet(vocabulary_size, text_latent_size, words_in_sentence)
    if torch.cuda.is_available():
        net.cuda()
    return net


def save_weights(net, path):
    torch.save(net.state_dict(), path)


def load_weights(net, path):
    net.load_state_dict(torch.load(path))


def step(net, optimizer, criterion, evidence, question, answer, step_num):
    optimizer.zero_grad()
    output = net(evidence, question, logger, step_num)
    loss = criterion(output, answer)
    loss.backward()
    gradient_noise_and_clip(net.parameters())
    optimizer.step()

    return loss.data[0]


def gradient_noise_and_clip(parameters, noise_stddev=1e-3, max_clip=40.0):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    nn.utils.clip_grad_norm(parameters, max_clip)

    for p in parameters:
        noise = torch.randn(p.size()) * noise_stddev
        if torch.cuda.is_available():
            noise = noise.cuda()
        p.grad.data.add_(noise)


def train(epochs, train_loader, test_loader, net, optimizer, criterion):
    total_step = 0
    for epoch in range(epochs):

        epoch_loss = 0
        for i, (captions_species, captions, question_species, question, answer) in enumerate(train_loader):
            captions = to_var(captions)

            # TODO see if varying number of captions helps training
            # captions = torch.index_select(captions, 1, torch.LongTensor(range(0, 10)).cuda())
            # captions = torch.index_select(captions, 2, torch.LongTensor(range(0, 20)).cuda())
            question = to_var(question)
            #question = torch.index_select(question, 1, torch.LongTensor(range(0, 7)).cuda())
            answer = to_var(answer)
            _, answer = torch.max(answer, 1)

            batch_loss = step(net, optimizer, criterion, captions, question, answer, total_step)
            epoch_loss += batch_loss

            if (total_step) % 100 == 0:
                train_acc = evaluate(net, train_loader)
                test_acc = evaluate(net, test_loader)
                print(total_step, batch_loss, train_acc, test_acc)
                tensorboard_logging(batch_loss, train_acc, test_acc, net, total_step)

            total_step = total_step + 1

                # if (epoch + 1) % 1 == 0:
        #     train_acc = evaluate(net, train_loader)
        #     test_acc = evaluate(net, test_loader)
        #     print(epoch + 1, epoch_loss, train_acc, test_acc)


def evaluate(net, loader):
    correct = 0.0

    for step, (captions_species, captions, question_species, question, answer) in enumerate(loader):
        captions = to_var(captions)

        # TODO see if varying number of captions helps training
        # captions = torch.index_select(captions, 1, torch.LongTensor(range(0, 10)).cuda())
        # captions = torch.index_select(captions, 2, torch.LongTensor(range(0, 20)).cuda())
        question = to_var(question)
        #question = torch.index_select(question, 1, torch.LongTensor(range(0, 7)).cuda())
        answer = to_var(answer)
        _, answer = torch.max(answer, 1)

        output = net(captions, question)
        _, output_max_index = torch.max(output, 1)
        correct += (answer == output_max_index).float().sum().data[0]  # really weird, without float() this counter resets to 

    acc = correct / len(loader.dataset)

    return acc


if __name__ == "__main__":

    config = parse_config()
    learn_rate = config.lr
    batch_size = config.batch_size
    text_latent_size = config.text_latent_size
    epochs = config.epochs

    weight_path = './Model/vqamemnet.pkl'
    #pdb.set_trace()
    train_loader, test_loader, vocabulary_size, words_in_sentence = load_data(batch_size)

    net = load_model(vocabulary_size, text_latent_size, words_in_sentence)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

    train(epochs, train_loader, test_loader, net, optimizer, criterion)
