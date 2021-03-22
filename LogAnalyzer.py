import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertForMaskedLM
from transformers import pipeline
from pprint import pprint

from tqdm import tqdm, trange
import pandas as pd
import math
import io
import numpy as np
import matplotlib.pyplot as plt


class LogAnalyzer(object):
    def __init__(self, input_train, input_test, mask_train, mask_test):
        self.batch_size = 32

        self.train_data = TensorDataset(input_train, mask_train)
        self.train_sampler = RandomSampler(input_train)
        self.train_data_loader = DataLoader(input_train, sampler=self.train_sampler, batch_size=self.batch_size)

        #self.model = BertForSequenceClassification.from_pretrained()
        self.masked_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def masked_language(self, input_train):
        prediction = self.masked_model(input_train.type(torch.LongTensor))
        loss_fn = torch.nn.CrossEntropyLoss()

        lost = loss_fn(prediction.squeez(), input_train.squeeze()).data
        return math.exp(lost)

    def nlp_model(self):
        nlp = pipeline('fill-mask')
        pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))