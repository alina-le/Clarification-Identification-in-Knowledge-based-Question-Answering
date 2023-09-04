import transformers
from transformers import BertModel, BertTokenizer
from transformers import AdamW  # use optimizer from Transformers library, not Pytorch
from transformers.optimization import get_linear_schedule_with_warmup  # use scheduler from Transformers library, not Pytorch

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter

import math
import numpy as np

import pandas as pd
import csv

import random 
import time

import logging

from sklearn.metrics import classification_report

from evaluate import evaluate
from utils import save_checkpoint, load_checkpoint, save_results_to_csv, plot_train_loss, plot_train_acc, plot_train_f1, plot_train_metrics, plot_prec_vs_recall, plot_loss_vs_f1, store_train_classification_metrics_full


def train_epoch(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer: transformers,
                scheduler: transformers,
                device: torch.device,
                epoch: int):
    ''' Trains a Pytorch model for a single epoch. 
    Architecture inspired from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb and https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb and https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

    Turns model to training mode and then runs through all of the required training steps (forward pass, loss calculation, optimizer step).

    :param model: A PyTorch classifier with BERT model layer to be trained.
    :param train_dataloader: A DataLoader instance for the model to be trained on.
    :param criterion: A PyTorch loss function to minimize.
    :param optimizer: A Transformers optimizer to help minimize the loss function.
    :param scheduler: A Transformers scheduler to let learning rate decrease linearly.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    :param epoch: An integer indicating the number of the current training epoch.

    '''
    # Reset tracking variables at the beginning of each epoch
    batch_loss, batch_cnt, total_loss = 0, 0, 0  # track number of batches s.t. batch loss can be calculated every 20 batches and track total loss to get average loss for one epoch
    train_accuracies = 0
    train_precisions = 0
    train_recalls = 0
    train_f1_scores = 0

    # Store true and predicted labels to calculate classification metrics
    y_true_labels = []
    y_pred_labels = []

    # Measure the elapsed time of each epoch
    t0_epoch, t0_batch = time.time(), time.time()

    # Set model to train mode
    model.train()

    # Loop through train data batches
    for step, batch in enumerate(train_dataloader):
        batch_cnt += 1

        # Unpack batch and load to device
        x_input_ids = batch['input_ids'].to(device)
        x_attention_mask = batch['attention_mask'].to(device) 
        y_labels = batch['labels'].to(device)

        # Forward pass
        train_logits = model(input_ids=x_input_ids, attention_mask=x_attention_mask)

        # Transform model logits into labels (for classification metrics report)
        pred_labels = torch.round(torch.sigmoid(train_logits))  # logits -> pred probs -> pred labls
        y_pred_labels.extend(pred_labels.cpu().detach().numpy())
        y_true_labels.extend(y_labels.cpu().numpy())

        # Calculate the loss
        loss = criterion(train_logits, y_labels)  # give raw model outputs to loss function, BCEWithLogitsLoss applies a sigmoid
        batch_loss += loss.item()
        total_loss += loss.item()

        # Zero out previously calculated gradients
        model.zero_grad()

        # Backward pass
        loss.backward()

        # Clip the norm of the gradients to 1.0 to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and the learning rate
        optimizer.step()
        scheduler.step()

        # Print the loss values and time elapsed for every 20 batches
        #if 1==1:
        if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            # Calculate time elapsed for 20 batches
            time_elapsed = time.time() - t0_batch

            # Print training results
            #print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_cnt:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
            #print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_cnt:^12.6f} | {'-':^9} | {'-':^10} | {'-':^9} | {'-':^11} | {'-':^9} | {'-':^6} | {time_elapsed:^9.2f}")
            
            logging.info(f"Epoch: {epoch+1} | "
                f"step: {step} | "
                f"batch_loss: {batch_loss / batch_cnt:.4f} | "
                f"time: {time_elapsed:.2f} | "
                )
            
            # Reset batch tracking variables
            batch_loss, batch_cnt = 0, 0
            t0_batch = time.time()

    # Calculate the average loss over the entire training data
    avg_train_loss = total_loss / len(train_dataloader)

    #logging.info("Train report:")
    #logging.info(f"\n {classification_report(y_true=y_true_labels, y_pred=y_pred_labels, digits=4)}")

    # Calculate time elapsed for one epoch
    time_elapsed = time.time() - t0_epoch

    return avg_train_loss, y_pred_labels, y_true_labels, time_elapsed



def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          criterion: torch.nn.Module,
          optimizer: transformers,
          scheduler: transformers,
          epochs: int,
          device: torch.device,
          checkpoint_dir: str,
          checkpoint_fname: str):
        ''' Trains a PyTorch model. During each epoch, performs training step and then evaluates on validation data.

        :param model: A PyTorch classifier with BERT model layer to be trained.
        :param train_dataloader: A DataLoader instance for the model to be trained on.
        :param val_dataloader: A DataLoader instance for the model to be evaluated on.
        :param criterion: A PyTorch loss function to minimize.
        :param optimizer: A Transformers optimizer to help minimize the loss function.
        :param scheduler: A Transformers scheduler to let learning rate decrease linearly.
        :param epochs: An integer indicating the number of epochs to train for. 
        :param device: A target device to compute on (e.g. "cuda" or "cpu").
        :param checkpoint_dir: A directory name where the model checkpoints are stored.
        :param checkpoint_fname: A filename where the model checkpoint is stored.

        '''
        # Set up Tensorboard writer to log results
        tb_writer = SummaryWriter(log_dir=checkpoint_dir + '/' + "tensorboard_" + checkpoint_fname) 
        
        # Set up results storage
        results = {"epoch": [],
                "train_loss": [],
                "train_acc": [],
                "train_macro_precision": [],
                "train_macro_recall": [],
                "train_macro_f1": [],
                "train_weighted_precision": [],
                "train_weighted_recall": [],
                "train_weighted_f1": [],
                "train_class0_precision": [],
                "train_class0_recall": [],
                "train_class0_f1": [],
                "train_class1_precision": [],
                "train_class1_recall": [],
                "train_class1_f1": [],
                "val_loss": [],
                "val_acc": [],
                "val_macro_precision": [],
                "val_macro_recall": [],
                "val_macro_f1": [],
                "val_weighted_precision": [],
                "val_weighted_recall": [],
                "val_weighted_f1": [],
                "val_class0_precision": [],
                "val_class0_recall": [],
                "val_class0_f1": [],
                "val_class1_precision": [],
                "val_class1_recall": [],
                "val_class1_f1": [],
                "test_loss": [],
                "test_acc": [],
                "test_macro_precision": [],
                "test_macro_recall": [],
                "test_macro_f1": [],
                "test_weighted_precision": [],
                "test_weighted_recall": [],
                "test_weighted_f1": [],
                "test_class0_precision": [],
                "test_class0_recall": [],
                "test_class0_f1": [],
                "test_class1_precision": [],
                "test_class1_recall": [],
                "test_class1_f1": []
        }
        
        best_val_f1 = 0.0  # store best validation accuracy so far for best model checkpoint

        # Start training loop
        logging.info(f"Start training on device: {device} ...\n")
        # Loop through epochs
        for epoch_i in range(epochs):

            # Train epoch
            train_loss, train_pred_labels, train_true_labels, train_time = train_epoch(model=model, train_dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch_i)

            # Evaluate
            val_loss, val_pred_labels, val_true_labels = evaluate(model=model, val_dataloader=val_dataloader, criterion=criterion, device=device)

            # Store results
            train_metrics = classification_report(y_true=train_true_labels, y_pred=train_pred_labels, output_dict=True)
            val_metrics = classification_report(y_true=val_true_labels, y_pred=val_pred_labels, output_dict=True)
            results = store_train_classification_metrics_full(results=results, epoch=epoch_i+1, train_metrics=train_metrics, train_loss=train_loss, val_metrics=val_metrics, val_loss=val_loss)
    
            # Save last model checkpoint; save best model checkpoint based on validation F1-score
            is_best = val_metrics['macro avg']['f1-score'] >= best_val_f1 
            save_checkpoint({'epoch': epoch_i + 1,
                             'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict()},
                             is_best=is_best,
                             checkpoint_dir=checkpoint_dir,
                             checkpoint_fname=checkpoint_fname)
            # If best_eval, best_save_path
            if is_best:
                #logging.info("Found new best accuracy")
                best_val_f1 = val_metrics['macro avg']['f1-score']

            # Log results for Tensorboard
            tb_writer.add_scalars(main_tag="Loss",
                                  tag_scalar_dict={"train_loss":train_loss,
                                                   "validation_loss":val_loss},
                                  global_step=epoch_i+1)
            tb_writer.add_scalars(main_tag="Accuracy",
                                  tag_scalar_dict={"train_acc":train_metrics['accuracy'],
                                                   "validation_acc":val_metrics['accuracy']},
                                  global_step=epoch_i+1)
            
            # Write results to logging
            logging.info(f"Epoch: {epoch_i+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_metrics['accuracy']:.4f} | "
                f"train_macro_p: {train_metrics['macro avg']['precision']:.4f} | "
                f"train_macro_r: {train_metrics['macro avg']['recall']:.4f} | "
                f"train_macro_f1: {train_metrics['macro avg']['f1-score']:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_metrics['accuracy']:.4f} | "
                f"val_macro_p: {val_metrics['macro avg']['precision']:.4f} | "
                f"val_macro_r: {val_metrics['macro avg']['recall']:.4f} | "
                f"val_macro_f1: {val_metrics['macro avg']['f1-score']:.4f} | "
                )
            logging.info("Classification report on validation data:")
            logging.info(f"\n {classification_report(y_true=val_true_labels, y_pred=val_pred_labels, digits=4)}")

        logging.info("Training complete!")

        # Write results to CSV file
        save_results_to_csv(csv_dir=checkpoint_dir, csv_fname=checkpoint_fname, results=results)

        # Plot training vs validation loss and metrics
        plot_train_loss(checkpoint_dir=checkpoint_dir, checkpoint_fname=checkpoint_fname, num_epochs=epochs, train_loss_per_epoch=results['train_loss'], val_loss_per_epoch=results['val_loss'])
        plot_train_acc(checkpoint_dir=checkpoint_dir, checkpoint_fname=checkpoint_fname, num_epochs=epochs, train_acc_per_epoch=results['train_acc'], val_acc_per_epoch=results['val_acc'])
        plot_train_f1(checkpoint_dir=checkpoint_dir, checkpoint_fname=checkpoint_fname, num_epochs=epochs, train_macro_precision_per_epoch=results['train_macro_precision'], train_macro_recall_per_epoch=results['train_macro_recall'], train_macro_f1_per_epoch=results['train_macro_f1'], val_macro_precision_per_epoch=results['val_macro_precision'], val_macro_recall_per_epoch=results['val_macro_recall'], val_macro_f1_per_epoch=results['val_macro_f1'])
        plot_train_metrics(checkpoint_dir=checkpoint_dir, checkpoint_fname=checkpoint_fname, num_epochs=epochs, train_acc_per_epoch=results['train_acc'], train_macro_precision_per_epoch=results['train_macro_precision'], train_macro_recall_per_epoch=results['train_macro_recall'], train_macro_f1_per_epoch=results['train_macro_f1'], val_acc_per_epoch=results['val_acc'], val_macro_precision_per_epoch=results['val_macro_precision'], val_macro_recall_per_epoch=results['val_macro_recall'], val_macro_f1_per_epoch=results['val_macro_f1'])
        plot_prec_vs_recall(checkpoint_dir=checkpoint_dir, checkpoint_fname=checkpoint_fname, num_epochs=epochs, train_macro_f1_per_epoch=results['train_macro_f1'], val_macro_precision_per_epoch=results['val_macro_precision'], val_macro_recall_per_epoch=results['val_macro_recall'], val_macro_f1_per_epoch=results['val_macro_f1'], val_class0_precision_per_epoch=results['val_class0_precision'], val_class0_recall_per_epoch=results['val_class0_recall'], val_class0_f1_per_epoch=results['val_class0_f1'], val_class1_precision_per_epoch=results['val_class1_precision'], val_class1_recall_per_epoch=results['val_class1_recall'], val_class1_f1_per_epoch=results['val_class1_f1'])
        plot_loss_vs_f1(checkpoint_dir=checkpoint_dir, checkpoint_fname=checkpoint_fname, num_epochs=epochs, train_macro_f1_per_epoch=results['train_macro_f1'], val_macro_f1_per_epoch=results['val_macro_f1'], train_loss_per_epoch=results['train_loss'], val_loss_per_epoch=results['val_loss'])
    
        # Close Tensorboard writer
        tb_writer.close()

        