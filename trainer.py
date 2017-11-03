import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from birds.dataset import birdCaptionSimpleYesNoDataset
from model import MemN2N
from logger import Logger

# Set the logger
run_name = 'run'
logger = Logger('./logs/' + run_name)

def to_np(x):
    return x.data.cpu().numpy()

class Trainer():
    def __init__(self, config):
        self.train_data = birdCaptionSimpleYesNoDataset(config.dataset_dir, limit_to_species=True, dataset_type="train")
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True)

        self.test_data = birdCaptionSimpleYesNoDataset(config.dataset_dir, limit_to_species=True, dataset_type="test")
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        num_vocab = len(self.train_data.word_idx)
        sentence_size = self.train_data.sentence_size

        settings = {
            "use_cuda": config.cuda,
            "num_vocab": num_vocab,
            "embedding_dim": config.embed_dim,
            "sentence_size": sentence_size,
            "max_hops": 1
        }

        print("Longest sentence length", sentence_size)
        print("Number of vocab", num_vocab)

        self.mem_n2n = MemN2N(settings)
        self.ce_fn = nn.CrossEntropyLoss(size_average=False)
        self.opt = torch.optim.SGD(self.mem_n2n.parameters(), lr=config.lr)
        print(self.mem_n2n)
            
        if config.cuda:
            self.ce_fn   = self.ce_fn.cuda()
            self.mem_n2n = self.mem_n2n.cuda()

        self.start_epoch = 0
        self.config = config

    def fit(self):
        config = self.config
        for epoch in range(self.start_epoch, config.max_epochs):
            loss = self._train_single_epoch(epoch)
            lr = self._decay_learning_rate(self.opt, epoch)

            if (epoch+1) % 10 == 0:
                train_acc = self.evaluate("train")
                test_acc = self.evaluate("test")
                print(epoch+1, loss, train_acc, test_acc)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'loss': loss,
                    'train accuracy': train_acc,
                    'test accuracy': test_acc
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in self.mem_n2n.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), epoch)
                    if value.grad is not None:
                        logger.histo_summary(tag + '/grad', to_np(value.grad), epoch)

        print(train_acc, test_acc)

    def load(self, directory):
        pass

    def evaluate(self, data):
        correct = 0
        loader = self.train_loader if data == "train" else self.test_loader
        for step, (captions_species, captions, question_species, question, answer) in enumerate(loader):
            captions = Variable(captions)
            question = Variable(question)
            answer = Variable(answer)
            _, answer = torch.max(answer, 1)

            if self.config.cuda:
                captions = captions.cuda()
                question = question.cuda()
                answer = answer.cuda()

            pred_prob = self.mem_n2n(captions, question)[1]
            _, output_max_index = torch.max(pred_prob, 1)
            # TODO
            if output_max_index.sum().data[0] != 0:
                print("OK!" + str(output_max_index.sum().data[0]))
            else:
                print("NOPE")

            correct = correct + (answer == output_max_index).float().sum().data[0]

        acc = correct / len(loader.dataset)
        return acc

    def _train_single_epoch(self, epoch):
        config = self.config
        num_steps_per_epoch = len(self.train_loader)
        for step, (captions_species, captions, question_species, question, answer) in enumerate(self.train_loader):
            captions = Variable(captions)
            question = Variable(question)
            answer = Variable(answer)
            _, answer = torch.max(answer, 1)

            if config.cuda:
                captions = captions.cuda()
                question = question.cuda()
                answer = answer.cuda()
        
            self.opt.zero_grad()
            loss = self.ce_fn(self.mem_n2n(captions, question)[0], answer)
            loss.backward()

            self._gradient_noise_and_clip(self.mem_n2n.parameters(),
                noise_stddev=1e-3, max_clip=config.max_clip)
            self.opt.step()

        return loss.data[0]

    def _gradient_noise_and_clip(self, parameters,
                                 noise_stddev=1e-3, max_clip=40.0):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        nn.utils.clip_grad_norm(parameters, max_clip)

        for p in parameters:
            noise = torch.randn(p.size()) * noise_stddev
            if self.config.cuda:
                noise = noise.cuda()
            p.grad.data.add_(noise)

    def _decay_learning_rate(self, opt, epoch):
        decay_interval = self.config.decay_interval
        decay_ratio    = self.config.decay_ratio

        decay_count = max(0, epoch // decay_interval)
        lr = self.config.lr * (decay_ratio ** decay_count)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        return lr
