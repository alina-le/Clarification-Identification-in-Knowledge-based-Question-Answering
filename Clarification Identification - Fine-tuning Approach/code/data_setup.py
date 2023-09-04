import transformers
#from transformers import BertModel, BertTokenizer
from transformers import AdamW  # use optimizer from Transformers library, not Pytorch
from transformers.optimization import get_linear_schedule_with_warmup  # use scheduler from Transformers library, not Pytorch

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_matthews_corrcoef

import logging 

import math
import numpy as np
import pandas as pd


class ClaquaDataset(Dataset):
    ''' store Claqua Corpus from pandas dataframe for model processing; Dataset architecture inspired by: https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    '''

    def __init__(self, text_items, encoded_text_items, labels):
        ''' Claqua corpus given in form of pandas dataframe columns: one column for text items, one column for labels of the text items;
        convert these dataframe columns to numpy array / torch tensor
        :param text_items: column from pandas dataframe which holds the text items,
        :param encoded_text_items: column from pandas dataframe which holds the text items, which are already already tokenized with BERT tokenizer and padded to max length
        :param labels: column from pandas dataframe which holds the labels for the text items
        '''
        self.text_items = text_items.to_numpy()  # convert dataframe column to numpy for faster access
        self.encoded_text_items = encoded_text_items.to_numpy()  # convert dataframe column to numpy for faster access
        self.labels = torch.tensor(labels, dtype=torch.float).unsqueeze(dim=1)  # convert dataframe column to tensor (convert to numpy: labels.to_numpy())
        self.num_samples = self.labels.shape[0]  # get number of labels == number of text items == length of the dataset

    def __getitem__(self, idx):
        ''' get item from Dataset at index
        :param idx: index of item to retriev
        :return: a dictionary
        '''
        text_item = self.text_items[idx]
        encoded_text = self.encoded_text_items[idx]
        label = self.labels[idx]

        return {
            'text_item' : text_item,
            'input_ids' : encoded_text['input_ids'].squeeze(),  # BERT embeddings for tokens; shape before squeeze: torch.Size([4, 1, 300]), after squeeze: torch.Size([4, 300])
            'attention_mask' : encoded_text['attention_mask'].squeeze(),  # BERT attention mask 
            #'token_type_ids' : encoded_text['token_type_ids'].squeeze(),  # BERT token type IDs (needed? only one token type id in this case, distinction solved via special tokens)
            'labels' : label
        }

    def __len__(self):
        ''' retrieving the number of samples in the dataset
        :return: the number of samples in the dataset
        '''
        return self.num_samples