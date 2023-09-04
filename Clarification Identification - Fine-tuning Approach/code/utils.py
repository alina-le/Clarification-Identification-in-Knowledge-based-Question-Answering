import os
import logging

import random
import numpy as np

import torch

import json
import shutil

import csv

from matplotlib import pyplot as plt


def set_seed(seed=42):
    ''' Set random seed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_dir, log_fname):
    '''Set the logger to log info in terminal and file `log_path`. Implementation adapted from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    log_path = os.path.join(log_dir, log_fname + '.log')

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode="a")  # with mode x: throw an error if log file exists already
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


class Params():
    """Class that loads hyperparameters from a json file. Implementation adapted from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    

def save_results_to_csv(csv_dir, csv_fname, results, append=False):
    ''' Stores model results in CSV file.
    :param results (dict): A dictionary of model results.
    :param csv_dir: A directory name where the results in csv form are stored.
    :param csv_fname: A filename where the the results in csv form is stored.
    '''
    csv_filepath = os.path.join(csv_dir, csv_fname + '.csv')

    if append:  # Append to existing file
        with open(csv_filepath, "a") as f:
            writer = csv.writer(f)
            writer.writerows(zip(*results.values()))
    else:  # Write new file
        with open(csv_filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(results.keys()) # write header
            writer.writerows(zip(*results.values()))


def save_checkpoint(state, is_best, checkpoint_dir, checkpoint_fname):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'.  Implementation adapted from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
    
    :param state (dict): contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
    :param is_best (bool): True if it is the best model seen till now
    :param checkpoint_dir: A directory name where the checkpoints are stored.
    :param checkpoint_fname: A filename where the checkpoint is stored.
    """
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.last.pth.tar')

    torch.save(state, checkpoint_filepath)
    if is_best:
        shutil.copyfile(checkpoint_filepath, os.path.join(checkpoint_dir, checkpoint_fname + '.best.pth.tar'))


def load_checkpoint(load_best, checkpoint_dir, checkpoint_fname):
    '''Loads model checkpoint from file. 

    :param load_best (bool): A boolean to indicate whether to load the best checkpoint. If set to False, instead the last model checkpoint is loaded.
    :param checkpoint_dir: A directory name where the checkpoints are stored.
    :param checkpoint_fname: A filename where the checkpoint is stored.
    :return: The model checkpoint.
    '''
    
    if load_best:  # Load best model
        logging.info("Loading checkpoint from best model.")
        checkpoint_filepath  = os.path.join(checkpoint_dir, checkpoint_fname + '.best.pth.tar')
    else:  # Load last model
        logging.info("Loading checkpoint from last model.")
        checkpoint_filepath  = os.path.join(checkpoint_dir, checkpoint_fname + '.last.pth.tar')
    
    if not os.path.exists(checkpoint_filepath):
        raise ("File doesn't exist {}".format(checkpoint_filepath))
    checkpoint = torch.load(checkpoint_filepath)

    return checkpoint


def remove_checkpoints(checkpoint_dir, checkpoint_fname):
    ''' Deletes model checkpoints to save disk space.
    :param checkpoint_dir: A directory name where the checkpoints are stored.
    :param checkpoint_fname: A filename where the checkpoint is stored.
    '''
    best_checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.best.pth.tar')
    last_checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.last.pth.tar')

    if os.path.exists(best_checkpoint_filepath):
        os.remove(best_checkpoint_filepath)
        logging.info("Removed best model checkpoint from disk.")

    if os.path.exists(last_checkpoint_filepath):
        os.remove(last_checkpoint_filepath)
        logging.info("Removed last model checkpoint from disk.")


def plot_train_loss(checkpoint_dir, checkpoint_fname, num_epochs, train_loss_per_epoch, val_loss_per_epoch):
    ''' Plots training and validation loss over epochs.
    '''
    epochs = list(range(1, num_epochs+1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss_per_epoch, label='train loss')
    ax.plot(epochs, val_loss_per_epoch, label='val loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.xticks(epochs)
    ax.legend(loc='lower right', fontsize="small")
    plt.show()
    # Save plot
    plot_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.loss_plot.pdf')
    plt.savefig(plot_filepath)


def plot_train_acc(checkpoint_dir, checkpoint_fname, num_epochs, train_acc_per_epoch, val_acc_per_epoch):
    ''' Plots training and validation accuracy over epochs.
    '''
    epochs = list(range(1, num_epochs+1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_acc_per_epoch, label ='train acc')
    ax.plot(epochs, val_acc_per_epoch, label = 'val acc')
    ax.set_title('Training and Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    plt.xticks(epochs)
    ax.legend(loc='lower right', fontsize="small")
    plt.show()
    # Save plot
    plot_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.acc_plot.pdf')
    plt.savefig(plot_filepath)


def plot_train_f1(checkpoint_dir, checkpoint_fname, num_epochs, train_macro_precision_per_epoch, train_macro_recall_per_epoch, train_macro_f1_per_epoch, val_macro_precision_per_epoch, val_macro_recall_per_epoch, val_macro_f1_per_epoch):
    ''' Plots training and validation precision, recall and F1 over epochs.
    '''
    epochs = list(range(1, num_epochs+1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_macro_precision_per_epoch, label ='train macro precision')
    ax.plot(epochs, train_macro_recall_per_epoch, label ='train macro recall')
    ax.plot(epochs, train_macro_f1_per_epoch, label ='train macro f1')
    ax.plot(epochs, val_macro_precision_per_epoch, label = 'val macro precision')
    ax.plot(epochs, val_macro_recall_per_epoch, label = 'val macro recall')
    ax.plot(epochs, val_macro_f1_per_epoch, label = 'val macro f1')
    ax.set_title('Training and Validation Precision, Recall and F1-score')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Precision / Recall / F1')
    plt.xticks(epochs)
    ax.legend(loc='lower right', fontsize="small")
    plt.show()
    # Save plot
    plot_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.f1_plot.pdf')
    plt.savefig(plot_filepath)


def plot_train_metrics(checkpoint_dir, checkpoint_fname, num_epochs, train_acc_per_epoch, train_macro_precision_per_epoch, train_macro_recall_per_epoch, train_macro_f1_per_epoch, val_acc_per_epoch, val_macro_precision_per_epoch, val_macro_recall_per_epoch, val_macro_f1_per_epoch):
    ''' Plots training and validation metrics over epochs.
    '''
    epochs = list(range(1, num_epochs+1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_acc_per_epoch, label ='train acc')
    ax.plot(epochs, train_macro_precision_per_epoch, label ='train macro precision')
    ax.plot(epochs, train_macro_recall_per_epoch, label ='train macro recall')
    ax.plot(epochs, train_macro_f1_per_epoch, label ='train macro f1')
    ax.plot(epochs, val_acc_per_epoch, label = 'val acc')
    ax.plot(epochs, val_macro_precision_per_epoch, label = 'val macro precision')
    ax.plot(epochs, val_macro_recall_per_epoch, label = 'val macro recall')
    ax.plot(epochs, val_macro_f1_per_epoch, label = 'val macro f1')
    ax.set_title('Training and Validation Accuracy, Precision, Recall and F1-score')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy / Precision / Recall / F1')
    plt.xticks(epochs)
    ax.legend(loc='lower right', fontsize="small")
    plt.show()
    # Save plot
    plot_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.metrics_plot.pdf')
    plt.savefig(plot_filepath)


def plot_prec_vs_recall(checkpoint_dir, checkpoint_fname, num_epochs, train_macro_f1_per_epoch, val_macro_precision_per_epoch, val_macro_recall_per_epoch, val_macro_f1_per_epoch, val_class0_precision_per_epoch, val_class0_recall_per_epoch, val_class0_f1_per_epoch, val_class1_precision_per_epoch, val_class1_recall_per_epoch, val_class1_f1_per_epoch):
    ''' Plots training and validation metrics over epochs.
    '''
    epochs = list(range(1, num_epochs+1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_macro_f1_per_epoch, label ='train macro f1')
    ax.plot(epochs, val_macro_precision_per_epoch, label = 'val macro precision')
    ax.plot(epochs, val_macro_recall_per_epoch, label = 'val macro recall')
    ax.plot(epochs, val_macro_f1_per_epoch, label = 'val macro f1')
    ax.plot(epochs, val_class0_precision_per_epoch, label = 'val class 0 precision')
    ax.plot(epochs, val_class0_recall_per_epoch, label = 'val class 0 recall')
    ax.plot(epochs, val_class0_f1_per_epoch, label = 'val class 0 f1')
    ax.plot(epochs, val_class1_precision_per_epoch, label = 'val class 1 precision')
    ax.plot(epochs, val_class1_recall_per_epoch, label = 'val class 1 recall')
    ax.plot(epochs, val_class1_f1_per_epoch, label = 'val class 1 f1') 
    ax.set_title('Training and Validation Precision vs. Recall')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Precision / Recall / F1')
    plt.xticks(epochs)
    ax.legend(loc='lower right', fontsize="x-small")
    plt.show()
    # Save plot
    plot_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.prec_vs_recall_plot.pdf')
    plt.savefig(plot_filepath)


def plot_loss_vs_f1(checkpoint_dir, checkpoint_fname, num_epochs, train_macro_f1_per_epoch, val_macro_f1_per_epoch, train_loss_per_epoch, val_loss_per_epoch):
    ''' Plots training and validation metrics over epochs.
    '''
    epochs = list(range(1, num_epochs+1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_macro_f1_per_epoch, label ='train macro f1')
    ax.plot(epochs, val_macro_f1_per_epoch, label = 'val macro f1')
    ax.plot(epochs, train_loss_per_epoch, label='train loss')
    ax.plot(epochs, val_loss_per_epoch, label='val loss')
    ax.set_title('Training and Validation Loss and F1-score')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss / F1')
    plt.xticks(epochs)
    ax.legend(loc='lower right', fontsize="x-small")
    plt.show()
    # Save plot
    plot_filepath = os.path.join(checkpoint_dir, checkpoint_fname + '.loss_f1_plot.pdf')
    plt.savefig(plot_filepath)


def store_train_classification_metrics_full(results, epoch, train_metrics, train_loss, val_metrics, val_loss, digits=4):
    ''' Reads the train classification report metrics from sklearn classification report and stores them in a dictionary.
    Metrics are stored for train as well as for validation data.
    :param results: The dictionary of results to append to.
    :param epoch: An int indicating the current training epoch.
    :param train_metrics: Sklearn classification report dictionary calculated on train data.
    :param train_loss: The training loss of current epoch.
    :param val_metrics: Sklearn classification report dictionary calculated on validation data.
    :param val_loss: The validation loss of current epoch.
    :param digits: The number of digits for formatting output floating point values. Default is set to 4.
    '''

    # Store results
    results["epoch"].append(epoch)

    # Store train metrics
    results["train_loss"].append(round(train_loss,digits))
    results["train_acc"].append(round(train_metrics['accuracy'],digits))
    results["train_macro_precision"].append(round(train_metrics['macro avg']['precision'],digits))
    results["train_macro_recall"].append(round(train_metrics['macro avg']['recall'],digits))
    results["train_macro_f1"].append(round(train_metrics['macro avg']['f1-score'],digits))
    results["train_weighted_precision"].append(round(train_metrics['weighted avg']['precision'],digits))
    results["train_weighted_recall"].append(round(train_metrics['weighted avg']['recall'],digits))
    results["train_weighted_f1"].append(round(train_metrics['weighted avg']['f1-score'],digits))
    results["train_class0_precision"].append(round(train_metrics['0.0']['precision'],digits))
    results["train_class0_recall"].append(round(train_metrics['0.0']['recall'],digits))
    results["train_class0_f1"].append(round(train_metrics['0.0']['f1-score'],digits))
    results["train_class1_precision"].append(round(train_metrics['1.0']['precision'],digits))
    results["train_class1_recall"].append(round(train_metrics['1.0']['recall'],digits))
    results["train_class1_f1"].append(round(train_metrics['1.0']['f1-score'],digits))
    
    # Store validation metrics
    results["val_loss"].append(round(val_loss,digits))
    results["val_acc"].append(round(val_metrics['accuracy'],digits))
    results["val_macro_precision"].append(round(val_metrics['macro avg']['precision'],digits))
    results["val_macro_recall"].append(round(val_metrics['macro avg']['recall'],digits))
    results["val_macro_f1"].append(round(val_metrics['macro avg']['f1-score'],digits))
    results["val_weighted_precision"].append(round(val_metrics['weighted avg']['precision'],digits))
    results["val_weighted_recall"].append(round(val_metrics['weighted avg']['recall'],digits))
    results["val_weighted_f1"].append(round(val_metrics['weighted avg']['f1-score'],digits))
    results["val_class0_precision"].append(round(val_metrics['0.0']['precision'],digits))
    results["val_class0_recall"].append(round(val_metrics['0.0']['recall'],digits))
    results["val_class0_f1"].append(round(val_metrics['0.0']['f1-score'],digits))
    results["val_class1_precision"].append(round(val_metrics['1.0']['precision'],digits))
    results["val_class1_recall"].append(round(val_metrics['1.0']['recall'],digits))
    results["val_class1_f1"].append(round(val_metrics['1.0']['f1-score'],digits))

    # Store empty test metrics
    results["test_loss"].append('-')
    results["test_acc"].append('-')
    results["test_macro_precision"].append('-')
    results["test_macro_recall"].append('-')
    results["test_macro_f1"].append('-')
    results["test_weighted_precision"].append('-')
    results["test_weighted_recall"].append('-')
    results["test_weighted_f1"].append('-')
    results["test_class0_precision"].append('-')
    results["test_class0_recall"].append('-')
    results["test_class0_f1"].append('-')
    results["test_class1_precision"].append('-')
    results["test_class1_recall"].append('-')
    results["test_class1_f1"].append('-')

    # Return filled results dictionary
    return results


def store_test_classification_metrics_full(results, epoch, test_metrics, test_loss, digits=4):
    ''' Reads the test classification report metrics from sklearn classification report and stores them in a dictionary.
    Metrics are stored for train as well as for validation data.
    :param results: The dictionary of results to append to.
    :param epoch: An int indicating the training epoch where the model was saved and loaded from to evaluate on test data.
    :param test_metrics: Sklearn classification report dictionary calculated on test data.
    :param test_loss: The test loss.
    :param digits: The number of digits for formatting output floating point values. Default is set to 4.
    '''
    
    # Store results
    results["epoch"].append(epoch)

    # Store train metrics
    results["train_loss"].append('-')
    results["train_acc"].append('-')
    results["train_macro_precision"].append('-')
    results["train_macro_recall"].append('-')
    results["train_macro_f1"].append('-')
    results["train_weighted_precision"].append('-')
    results["train_weighted_recall"].append('-')
    results["train_weighted_f1"].append('-')
    results["train_class0_precision"].append('-')
    results["train_class0_recall"].append('-')
    results["train_class0_f1"].append('-')
    results["train_class1_precision"].append('-')
    results["train_class1_recall"].append('-')
    results["train_class1_f1"].append('-')
    
    # Store validation metrics
    results["val_loss"].append('-')
    results["val_acc"].append('-')
    results["val_macro_precision"].append('-')
    results["val_macro_recall"].append('-')
    results["val_macro_f1"].append('-')
    results["val_weighted_precision"].append('-')
    results["val_weighted_recall"].append('-')
    results["val_weighted_f1"].append('-')
    results["val_class0_precision"].append('-')
    results["val_class0_recall"].append('-')
    results["val_class0_f1"].append('-')
    results["val_class1_precision"].append('-')
    results["val_class1_recall"].append('-')
    results["val_class1_f1"].append('-')

    # Store empty test metrics
    results["test_loss"].append(round(test_loss,digits))
    results["test_acc"].append(round(test_metrics['accuracy'],digits))
    results["test_macro_precision"].append(round(test_metrics['macro avg']['precision'],digits))
    results["test_macro_recall"].append(round(test_metrics['macro avg']['recall'],digits))
    results["test_macro_f1"].append(round(test_metrics['macro avg']['f1-score'],digits))
    results["test_weighted_precision"].append(round(test_metrics['weighted avg']['precision'],digits))
    results["test_weighted_recall"].append(round(test_metrics['weighted avg']['recall'],digits))
    results["test_weighted_f1"].append(round(test_metrics['weighted avg']['f1-score'],digits))
    results["test_class0_precision"].append(round(test_metrics['0.0']['precision'],digits))
    results["test_class0_recall"].append(round(test_metrics['0.0']['recall'],digits))
    results["test_class0_f1"].append(round(test_metrics['0.0']['f1-score'],digits))
    results["test_class1_precision"].append(round(test_metrics['1.0']['precision'],digits))
    results["test_class1_recall"].append(round(test_metrics['1.0']['recall'],digits))
    results["test_class1_f1"].append(round(test_metrics['1.0']['f1-score'],digits))

    # Return filled results dictionary
    return results
