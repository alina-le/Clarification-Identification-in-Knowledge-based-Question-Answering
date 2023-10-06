import transformers
from transformers import BertModel, BertTokenizer
from transformers import AdamW  # use optimizer from Transformers library, not Pytorch
from transformers.optimization import get_linear_schedule_with_warmup  # use scheduler from Transformers library, not Pytorch

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_matthews_corrcoef

from sklearn.metrics import classification_report

import logging

from utils import load_checkpoint, save_results_to_csv, store_test_classification_metrics_full

def evaluate(model: torch.nn.Module, 
              val_dataloader: torch.utils.data.DataLoader, 
              criterion: torch.nn.Module,
              device: torch.device):
    ''' Tests a PyTorch model for a single epoch.
    Architecture inspired from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb and https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb and https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset.

    :param model: A PyTorch classifier with BERT model layer to be evaluated.
    :param val_dataloader: A DataLoader instance for the model to be evaluated on.
    :param criterion: A PyTorch loss function to calculate loss on the validation or test data.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    
    '''
    # Tracking variables
    val_losses = 0

    # Store true and predicted labels to calculate classification metrics
    y_true_labels = []
    y_pred_labels = []

    # Set model to eval mode
    model.eval()

    # Turn on inference context manager
    with torch.inference_mode():

        # Loop through eval data batches
        for batch in val_dataloader:

            # Unpack batch and load to device
            x_input_ids = batch['input_ids'].to(device)
            x_attention_mask = batch['attention_mask'].to(device) 
            y_labels = batch['labels'].to(device)

            # Get model predictions (logits)
            val_logits = model(input_ids=x_input_ids, attention_mask=x_attention_mask)

            # Calculate the loss
            loss = criterion(val_logits, y_labels)  # give raw model outputs to loss function, BCEWithLogitsLoss applies a sigmoid

            val_losses += loss.item()

            # Transform model logits into labels (for classification metrics report)
            pred_labels = torch.round(torch.sigmoid(val_logits))
            y_pred_labels.extend(pred_labels.cpu().numpy())
            y_true_labels.extend(y_labels.cpu().numpy())

        # Calculate the average loss over the entire validation data
        val_loss = val_losses / len(val_dataloader)

        return val_loss, y_pred_labels, y_true_labels
    

def eval_on_test(model: torch.nn.Module,
                test_dataloader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                device: torch.device,
                checkpoint_dir: str,
                checkpoint_fname: str,
                load_best: bool):
    ''' Runs evaluation of a Pytorch model on test data. Loads trained model from best or last checkpoint to evaluate on the test data.
    
    :param model: A PyTorch model instantiated to be loaded from checkpoint.
    :param test_dataloader: A DataLoader instance for the model to be evaluated on.
    :param criterion: A PyTorch loss function to calculate loss on the validation or test data.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    :param checkpoint_dir: A directory name where the checkpoints are stored.
    :param checkpoint_fname: A filename where the checkpoint is stored.
    :param load_best: A boolean to indicate whether to load the best checkpoint. If set to False, instead the last model checkpoint is loaded.

    '''
    # Set up results storage
    test_results = {"epoch": [],
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

    logging.info("Evaluating model on the test data.")

    # Load model from checkpoint
    checkpoint = load_checkpoint(load_best=load_best, 
                                 checkpoint_dir=checkpoint_dir,
                                 checkpoint_fname=checkpoint_fname)
    
    checkpoint_model_epoch = checkpoint['epoch']
    logging.info(f"Model checkpoint loaded from trained model saved at epoch {checkpoint_model_epoch}.")
    
    model.load_state_dict(checkpoint['state_dict'])

    test_loss, test_pred_labels, test_true_labels = evaluate(model=model, val_dataloader=test_dataloader, criterion=criterion, device=device)
    test_metrics = classification_report(y_true=test_true_labels, y_pred=test_pred_labels, output_dict=True)
            
    # Store results
    test_results = store_test_classification_metrics_full(results=test_results, epoch=checkpoint_model_epoch, test_metrics=test_metrics, test_loss=test_loss)

    # Write results to CSV file
    save_results_to_csv(csv_dir=checkpoint_dir, csv_fname=checkpoint_fname, results=test_results, append=True)

    # Write results to logging
    logging.info("Classification report on test data:")
    logging.info(f"\n {classification_report(y_true=test_true_labels, y_pred=test_pred_labels, digits=4)}")

    logging.info(f"Test loss: {test_loss:.6f}")
    logging.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Test macro precision: {test_metrics['macro avg']['precision']:.4f}")
    logging.info(f"Test macro recall: {test_metrics['macro avg']['recall']:.4f}")
    logging.info(f"Test macro F1 score: {test_metrics['macro avg']['f1-score']:.4f}")