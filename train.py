import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import vqa_memnet
from torch.utils.data import DataLoader
from birds.dataset import birdCaptionSimpleYesNoDataset
from synthetic.dataset import SyntheticDataset
from logger import Logger, initialize_html_logging
from dominate.tags import *

import pdb

# Set the logger
run_name = 'binary_3clues_per_species'
logger = Logger('./logs/' + run_name)
experiment_title = 'Binary + Add the Question + No normalize attention + 100S/100A/100C'

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str,
                        default="synthetic_data_100_species_100_attributes_3_clues_train.pckl")
    parser.add_argument("--test_file", type=str,
                        default="synthetic_data_100_species_100_attributes_3_clues_test.pckl")
    parser.add_argument("--dataset_dir", type=str, default="/home/shenkev/School/VQA-Memnet/synthetic/",
                        help='the path to the directory of the data')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='the batch size for each training iteration using a variant of stochastic gradient descent')
    parser.add_argument("--text_latent_size", type=int, default=50,
                        help='the size of text embedding for question and evidence')
    parser.add_argument("--epochs", type=int, default=15,
                        help='the number of epochs to train for')
    parser.add_argument("--lr", type=float, default=0.001,
                        help='the starting learning rate for the optimizer')
    parser.add_argument("--max_clip", type=float, default=40.0,
                        help='the upperbound for the gradient (for gradient explosions)')
    parser.add_argument("--attention_temperature", type=float, default=4.0,
                        help='the temperature used in the softmax for attention')

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


def load_bird_data(batch_size, dataset_dir='/home/shenkev/School/VQA-Memnet/birds'):

    train_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=True, dataset_type="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)

    test_data = birdCaptionSimpleYesNoDataset(dataset_dir=dataset_dir, limit_to_species=True, dataset_type="test")
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=False)

    vocabulary_size = len(train_data.word_idx)
    caption_length = train_data.sentence_size
    print("Longest caption length", caption_length)
    print("Number of vocab", vocabulary_size)

    return train_loader, test_loader, vocabulary_size, caption_length, train_data.word_idx


def load_synthetic_data(batch_size, config):
    train_dir = config.dataset_dir + config.train_file
    test_dir = config.dataset_dir + config.test_file

    train_data = SyntheticDataset(path_to_dataset=train_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)

    test_data = SyntheticDataset(path_to_dataset=test_dir)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=True)

    bin_vec_len = train_data.sentence_size
    print("Each image/clue is a binary vector of length {}".format(str(bin_vec_len)))

    return train_loader, test_loader, bin_vec_len


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def load_model(bin_vec_len, text_latent_size, attention_temperature):
    net = vqa_memnet(bin_vec_len, text_latent_size, attention_temperature)
    if torch.cuda.is_available():
        net.cuda()
    return net


def save_weights(net, path):
    torch.save(net.state_dict(), path)


def load_weights(net, path):
    net.load_state_dict(torch.load(path))


def step(net, optimizer, criterion, evidence, question, answer, step_num, _body):
    optimizer.zero_grad()
    output = net(evidence, question, _body, step_num, answer)
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


def train(epochs, train_loader, test_loader, net, optimizer, criterion, _body):
    total_step = 0
    # logging for html file
    _table = _body.add(table(
        tr(th('Step'), th('Loss'), th('Train Accuracy (total)'), th('Train Accuracy (0 answers)'),
           th('Train Accuracy (1 answers)'), th('Test Accuracy (total)'), th('Test Accuracy (0 answers)'),
           th('Test Accuracy (1 answers)')),
        cls="table"
    ))
    for epoch in range(epochs):

        epoch_loss = 0
        for i, (captions, question_species, question, answer) in enumerate(train_loader):
            captions = captions.float()
            question = question.float()

            captions = to_var(captions)
            question = to_var(question)
            answer = to_var(answer)
            _, answer = torch.max(answer, 1)

            batch_loss = step(net, optimizer, criterion, captions, question, answer, total_step, _body)
            epoch_loss += batch_loss

            if (total_step) % 100 == 0:

                train_acc = evaluate(net, train_loader)
                test_acc = evaluate(net, test_loader)
                print(total_step, batch_loss, train_acc, test_acc)
                tensorboard_logging(batch_loss, train_acc[0], test_acc[0], net, total_step)

                _table.add(tr(td(total_step), td(batch_loss), td(train_acc[0]), td(train_acc[1]), td(train_acc[2]),
                              td(test_acc[0]), td(test_acc[1]), td(test_acc[2])))

                net.train()

            total_step = total_step + 1


def evaluate(net, loader):
    correct = 0.0
    correct_zero = 0.0
    correct_one = 0.0
    total_zero = 0.0
    total_one = 0.0

    net.eval()
    for step, (captions, question_species, question, answer) in enumerate(loader):
        captions = captions.float()
        question = question.float()

        captions = to_var(captions)
        question = to_var(question)
        answer = to_var(answer)
        _, answer = torch.max(answer, 1)

        output = net(captions, question)
        _, output_max_index = torch.max(output, 1)

        cor_tot = (answer == output_max_index)
        cor_zero = answer[cor_tot]

        correct += cor_tot.float().sum().data[0]  # really weird, without float() this counter resets to
        correct_zero += (cor_tot.float().sum().data[0] - cor_zero.float().sum().data[0])
        correct_one += cor_zero.float().sum().data[0]
        total_zero += (answer == 0).sum().data[0]
        total_one += (answer == 1).sum().data[0]

    acc = correct / (total_zero + total_one)
    acc_zero = correct_zero / total_zero
    acc_one = correct_one / total_one

    return acc, acc_zero, acc_one


if __name__ == "__main__":

    config = parse_config()
    learn_rate = config.lr
    batch_size = config.batch_size
    text_latent_size = config.text_latent_size
    epochs = config.epochs
    attention_temperature = config.attention_temperature

    weight_path = './Model/vqamemnet.pkl'
    #pdb.set_trace()

    train_loader, test_loader, bin_vec_len = load_synthetic_data(batch_size, config)

    net = load_model(bin_vec_len, text_latent_size, attention_temperature)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

    _html, _body = initialize_html_logging(experiment_title, train_loader, net, optimizer, config)

    train(epochs, train_loader, test_loader, net, optimizer, criterion, _body)

    with open("./html/{}.html".format(run_name), "w") as f:
        f.write(_html.render())

