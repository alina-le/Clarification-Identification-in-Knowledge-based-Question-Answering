import transformers
from transformers import AutoModel, AutoTokenizer
#from transformers import BertModel, BertTokenizer
#from transformers import RobertaModel, RobertaTokenizer
from transformers import AdamW  # use optimizer from Transformers library, not Pytorch
from transformers.optimization import get_linear_schedule_with_warmup  # use scheduler from Transformers library, not Pytorch

import torch
from torchinfo import summary
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

import math
import numpy as np
import pandas as pd
import csv

from matplotlib import pyplot as plt

import random 
import time
from datetime import datetime

import os
import sys
import shutil
import argparse
import logging

import json

# from preprocess_claqua_for_model_training import ClaquaCorpus, process_corpus
from read_claqua_corpus import read_train_dev_test
from data_setup import ClaquaDataset
from models import TransformerClassifier0, TransformerClassifier1, TransformerClassifier1_1, TransformerClassifier1_2, TransformerClassifier1_3, TransformerClassifier2, TransformerClassifier2_1
from train import train
from evaluate import eval_on_test
from utils import set_logger, set_seed, Params, load_checkpoint, remove_checkpoints


def get_freer_gpu():
    ''' get available GPU; taken from https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/5
    '''
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def configure_training(model: torch.nn.Module,
                        total_steps: int, 
                        learning_rate: float,
                        class_weight: torch.Tensor, 
                        device: torch.device,
                        ):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler. Architecture inspired from: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
    
    :param model: A PyTorch classifier with BERT model layer to be trained.
    :param total_steps: An int indicating the number of total training steps.
    :param learning_rate: A float indicating the learning rate.
    :param class_weight: A tensor storing the positive class weight.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").

    :return criterion: A loss function of type torch.nn.Module.
    :return optimizer: An optimizer for training of type torch.optim.Optimizer.
    :return scheduler: A learning rate scheduler for training.
    """
    # Instantiate BERT classifier
    #bert_classifier = BertClassifier(pre_trained_model=pre_trained_model_name, vocab_size=vocab_size, freeze_transformer=freeze_transformer)

    # Send model to device
    model.to(device)

    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,  # learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # Set up loss function with class weight
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)  # BCEWithLogitsLoss because: "This version is more numerically stable than using a plain Sigmoid followed by a BCELoss" (https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
    criterion.to(device)

    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)

    return criterion, optimizer, scheduler


def instantiate_model_tokenizer(model_name: str,
              transformer_name: str,
              freeze_transformer: bool,
              special_tokens: dict,
              model0_dropout: float):
    ''' Map string to class name. Instantiate model class.

    :param model_name (str): Class name of the model, passed as string.
    :param transformer_name (str): Name of the Transformer, passed as string.
    :param freeze_transformer: A boolean indicating whether to freeze BERT layers during training or not.
    :param special_tokens: A dictionary with the special tokens for the Tranformer tokenizer.
    :param model0_dropout: A float indicating the dropout value for the model0 architecture.

    :return model: The model class object.
    :return tokenizer: The tokenizer object.
    '''

    # Pick Transformer matching the config file: specify pre-trained Huggingface model
    match transformer_name:
        case "Bert":
            pre_trained_model = 'bert-base-cased'  # set pre-trained Huggingface BERT model
        case "Roberta":
            pre_trained_model = 'roberta-base'  # set pre-trained Huggingface RoBERTa model (roberta-base is case-sensitive)
        case "Albert":
            pre_trained_model = 'albert-base-v2' # set pre-trained Huggingface ALBERT model (all ALBERT models are uncased)
        case "Distilbert":
            pre_trained_model = 'distilbert-base-cased'  # set pre-trained Huggingface DistilBERT model 
        case _:
            raise ("Transformer version does not exist")

        # for Electra, would need to change model structure first
        #case "Electra":
        #    pre_trained_model = 'google/electra-base-generator'  # set pre-trained Huggingface ELECTRA model

    # Load Transformer tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)  
    tokenizer.add_special_tokens(special_tokens)
    vocab_size = len(tokenizer)  # get tokenizer vocabulary size

    # Pick model matching the config file: load pre-trained Huggingface model and choose model architecture
    match model_name:
        case "TransformerClassifier0":
            dropout = model0_dropout if model0_dropout else 0.1 # for model0, pick dropout value
            model = TransformerClassifier0(pre_trained_model=pre_trained_model, vocab_size=vocab_size, dropout=dropout, freeze_transformer=freeze_transformer)
        case "TransformerClassifier1":
            model = TransformerClassifier1(pre_trained_model=pre_trained_model, vocab_size=vocab_size, freeze_transformer=freeze_transformer)
        case "TransformerClassifier1_1":
            model = TransformerClassifier1_1(pre_trained_model=pre_trained_model, vocab_size=vocab_size, freeze_transformer=freeze_transformer)
        case "TransformerClassifier1_2":
            model = TransformerClassifier1_2(pre_trained_model=pre_trained_model, vocab_size=vocab_size, freeze_transformer=freeze_transformer)
        case "TransformerClassifier1_3":
            model = TransformerClassifier1_3(pre_trained_model=pre_trained_model, vocab_size=vocab_size, freeze_transformer=freeze_transformer)
        case "TransformerClassifier2":
            model = TransformerClassifier2(pre_trained_model=pre_trained_model, vocab_size=vocab_size, freeze_transformer=freeze_transformer)
        case "TransformerClassifier2_1":
            model = TransformerClassifier2_1(pre_trained_model=pre_trained_model, vocab_size=vocab_size, freeze_transformer=freeze_transformer)
        case _:
            raise ("Model version does not exist")
            
    return model, tokenizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/', help='Specifies the path to the CLAQUA corpus. (default: "./data/")')
    parser.add_argument('--config', type=str, help='Specifies the path to json file containing the model hyperparameters and paths for storing models and logs.')
    parser.add_argument('--exp_base_dir', type=str, default='../experiments/', help='Specifies the base folder where experiments are stored (each within subfolders).')
    parser.add_argument('--test', type=bool, default=False, help='If set to True, test code flow on smaller portion of the dataset. Default set to False.')
    args = parser.parse_args()

    # Set up config from file
    params = Params(json_path=args.config)

    # Set up directory to store experiment logs, models etc. in
    if params.class_weight:
        if params.model0_dropout:
            experiment_folder = "dropout" + str(params.model0_dropout) + "_epochs" + str(params.num_epochs) + "_batch" + str(params.batch_size) + "_lr" + str(params.learning_rate) + "_withClassWeight" + "_seed" + str(params.seed) + params.exp_name
        else:  # no model 0 dropout specified (training different model or with default dropoout)
            experiment_folder = "epochs" + str(params.num_epochs) + "_batch" + str(params.batch_size) + "_lr" + str(params.learning_rate) + "_withClassWeight" + "_seed" + str(params.seed) + params.exp_name
    else:  # no class weight
        if params.model0_dropout:
            experiment_folder = "dropout" + str(params.model0_dropout) + "_epochs" + str(params.num_epochs) + "_batch" + str(params.batch_size) + "_lr" + str(params.learning_rate) + "_noClassWeight" + "_seed" + str(params.seed) + params.exp_name
        else:  # no model 0 dropout specified (training different model or with default dropoout)
            experiment_folder = "epochs" + str(params.num_epochs) + "_batch" + str(params.batch_size) + "_lr" + str(params.learning_rate) + "_noClassWeight" + "_seed" + str(params.seed) + params.exp_name
    experiment_dir = os.path.join(args.exp_base_dir, params.corpus_split, params.transformer_version, params.model_version, experiment_folder)  # name of the experiment folder
    #experiment_dir = os.path.join(args.exp_base_dir, params.corpus_split, params.model_version, params.exp_name)  # name of the experiment folder
    experiment_fname = params.corpus_split + '_' + params.transformer_version + '_' + params.model_version + '_' + experiment_folder  # base name for the experiment tracking files
    #experiment_fname = params.corpus_split + '_' + params.model_version + '_' + params.exp_name  # base name for the experiment tracking files
    os.makedirs(experiment_dir, exist_ok=True)
        
    # Set up logger
    set_logger(log_dir=experiment_dir, log_fname=experiment_fname)
    logging.info(f'Start preprocessing for model training. Data split is {params.corpus_split}. \n')

    # Set tokenizer vars
    NUM_MAX_TOKS = params.num_max_toks if params.num_max_toks else 300  # max number of tokens to encode, rest to be filled with padding; BERT max is at 512
    CONTEXT_SEP_TOK = params.con_sep if params.con_sep else '[CONTEXT_SEP]'  # default custom Transformer separator token to separate context from entity information
    ENTITY_SEP_TOK = params.ent_sep if params.ent_sep else '[ENTITY_SEP]'  # default custom Transformer separator token to separate entity1 from entity2 information
    TURN_SEP_TOK = params.turn_sep if params.turn_sep else '[TURN_SEP]'  # default custom Transformer separator token to separate different turns within the context; only needed for multi-turn corpus (single-turn corpus context constists of only one turn)

    # Load Transformer Huggingface model and tokenizer, thereby add special tokens
    FREEZE_TRANSFORMER = params.freeze_transformer if params.freeze_transformer else False
    claqua_special_tokens_dict = {'additional_special_tokens': [ENTITY_SEP_TOK, CONTEXT_SEP_TOK]} if params.corpus_split == 'single-turn' else {'additional_special_tokens': [ENTITY_SEP_TOK, CONTEXT_SEP_TOK, TURN_SEP_TOK]}  # special tokens depend on whether single-turn or multi-turn corpus split is used
    transf_classifier, claqua_tokenizer = instantiate_model_tokenizer(model_name=params.model_version, transformer_name=params.transformer_version, freeze_transformer=FREEZE_TRANSFORMER, special_tokens=claqua_special_tokens_dict, model0_dropout=params.model0_dropout)

    # Load train, dev and test corpus
    train_corpus, dev_corpus, test_corpus = read_train_dev_test(data_path=args.data, corpus_split=params.corpus_split, tokenizer=claqua_tokenizer, num_max_toks=NUM_MAX_TOKS, turn_sep_tok=TURN_SEP_TOK, context_sep_tok=CONTEXT_SEP_TOK, entity_sep_tok=ENTITY_SEP_TOK)

    if args.test == True:  # Test code on smaller portion of dataset
        # Load smaller portion of dataset for testing the model flow
        train_dataset = ClaquaDataset(train_corpus.df['full_text_for_encoding'].head(12), train_corpus.df['enc_full_text_padded'].head(12), train_corpus.df['label'].head(12))
        dev_dataset = ClaquaDataset(dev_corpus.df['full_text_for_encoding'].head(12), dev_corpus.df['enc_full_text_padded'].head(12), dev_corpus.df['label'].head(12))
        test_dataset = ClaquaDataset(test_corpus.df['full_text_for_encoding'].head(12), test_corpus.df['enc_full_text_padded'].head(12), test_corpus.df['label'].head(12))
    else: 
        # Create dataset instances
        train_dataset = ClaquaDataset(train_corpus.df['full_text_for_encoding'], train_corpus.df['enc_full_text_padded'], train_corpus.df['label'])
        dev_dataset = ClaquaDataset(dev_corpus.df['full_text_for_encoding'], dev_corpus.df['enc_full_text_padded'], dev_corpus.df['label'])
        test_dataset = ClaquaDataset(test_corpus.df['full_text_for_encoding'], test_corpus.df['enc_full_text_padded'], test_corpus.df['label'])
        
    logging.info(f"Length train data: {len(train_dataset)}")
    logging.info(f"Length dev data: {len(dev_dataset)}")
    logging.info(f"Length test data: {len(test_dataset)}")
    
    # Set model vars
    NUM_EPOCHS = params.num_epochs if params.num_epochs else 2  # Number of training epochs
    BATCH_SIZE = params.batch_size if params.batch_size else 8  # For fine-tuning Transformer, the authors recommend a batch size of 16 or 32.
    LEARNING_RATE = params.learning_rate

    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Learning rate: {LEARNING_RATE}")

    # Create the DataLoader for train set
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    TOTAL_STEPS = len(train_dataloader) * NUM_EPOCHS  # Total number of training steps
    TOTAL_SAMPLES = len(train_dataset)
    #NUM_ITERATIONS = math.ceil(TOTAL_SAMPLES/BATCH_SIZE) 

    # Set device and random seed
    device_idx = get_freer_gpu()
    device_name = "cuda:" + str(device_idx)
    device = device_name if torch.cuda.is_available() else "cpu"
    set_seed(params.seed)  # set random seed 
    logging.info(f"Seed: {params.seed}")
    # Set class weight
    CLASS_WEIGHT = train_corpus.positive_class_weight if params.class_weight is True else None 
    logging.info(f"Class weight: {CLASS_WEIGHT}")

    # Instantiate training
    criterion, optimizer, scheduler = configure_training(model=transf_classifier, total_steps=TOTAL_STEPS, learning_rate=LEARNING_RATE, class_weight=CLASS_WEIGHT, device=device)
    logging.info(f"Transformer model: {params.transformer_version}")
    logging.info(summary(transf_classifier))
    logging.info(f'Classifier architecture on top of Transformer model: {transf_classifier.classifier}')  #print(bert_classifier)
    
    # Train model
    train(model=transf_classifier, train_dataloader=train_dataloader, val_dataloader=dev_dataloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, epochs=NUM_EPOCHS, device=device, checkpoint_dir=experiment_dir, checkpoint_fname=experiment_fname)

    # Evaluate model on test data
    eval_on_best_model = True  # if set to True, eval on best model checkpoint; if set to False, eval on last model checkpoint
    test_transf_classifier, _ = instantiate_model_tokenizer(model_name=params.model_version, transformer_name=params.transformer_version, freeze_transformer=FREEZE_TRANSFORMER, special_tokens=claqua_special_tokens_dict, model0_dropout=params.model0_dropout)  # set up new empty model; eval function loads it from checkpoint
    test_criterion, _, _ =  configure_training(model=test_transf_classifier, total_steps=TOTAL_STEPS, learning_rate= LEARNING_RATE, class_weight=CLASS_WEIGHT, device=device)
    
    eval_on_test(model=test_transf_classifier, test_dataloader=test_dataloader, criterion=test_criterion, device=device, checkpoint_dir=experiment_dir, checkpoint_fname=experiment_fname, load_best=eval_on_best_model)

    # After training and evaluation, delete model checkpoints.
    remove_checkpoints(checkpoint_dir=experiment_dir, checkpoint_fname=experiment_fname)
    # Move config file to experiment directory and to directory of processed configs
    processed_configs_dir = '../configs/done/'  # directory for processed configs
    if not os.path.exists(processed_configs_dir):
        os.makedirs(processed_configs_dir)
    shutil.copy(args.config, processed_configs_dir)
    shutil.move(args.config, os.path.join(experiment_dir, experiment_fname + '.config.json'))


if __name__ == "__main__":
    main()
