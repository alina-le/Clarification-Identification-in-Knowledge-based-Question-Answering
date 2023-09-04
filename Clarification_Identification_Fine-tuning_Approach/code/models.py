import transformers
from transformers import AutoModel, AutoTokenizer
#from transformers import BertModel, BertTokenizer
from transformers import AdamW  # use optimizer from Transformers library, not Pytorch
from transformers.optimization import get_linear_schedule_with_warmup  # use scheduler from Transformers library, not Pytorch

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_matthews_corrcoef


# model0: one linear layer for classifier 
class TransformerClassifier0(nn.Module):
    ''' Bert Classifier; architecture inspired from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f and https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    and https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
    '''

    def __init__(self, pre_trained_model, vocab_size, dropout, freeze_transformer=False):
        ''' 
        :param pre_trained_model: name of pre-trained Transformer model object
        :param vocab_size: length of Transformer tokenizer - can differ from pre-trained Transformer tokenizer when special tokens are added
        '''
        super(TransformerClassifier0, self).__init__()

        self.num_classes = 1  # binary classification

        # Instantiate Transformer model
        self.transformer = AutoModel.from_pretrained(pre_trained_model)
        self.transformer.resize_token_embeddings(vocab_size)  # adjust vocab size of pre-trained Transformer for special tokens that were added during tokenization
        self.transformer_hidden_size = 768

        # Instantiate classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # dropout layer for regularization
            nn.Linear(self.transformer_hidden_size, self.num_classes)  # fully-connected layer for output
            # sigmoid layer is automatically calles in torch.nn.BCEWithLogitsLoss
        )
        
        # Freeze Transformer model to prevent fine-tuning Transformer and instead only train the classifier
        if freeze_transformer:
            logging.info("Transformer layers frozen during training.")
            for param in self.transformer.parameters():
                param.requires_grad = False
                

    def forward(self, input_ids, attention_mask):
        ''' 
        Feed input from Transformer tokenizer sequences first to Transformer, then to classifier to compute logits
        :param input_ids (torch.Tensor): input ids from Transformer pretrained tokenizer with shape (batch_size, max_length), the token embeddings, padded until max len
        :param attention_mask (torch.Tensor): binary tensor from Transformer tokenizer with shape (batch_size, max_length), indicating position of the padded indices so that the model does not attend to them
        :return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''

        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs['last_hidden_state'][:,0,:]  # take last hidden state of [CLS] token for classification task
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(cls_embedding)

        return logits


# model1: two linear layers for classifier, relu activation, dropout 0.3
class TransformerClassifier1(nn.Module):
    ''' Bert Classifier; architecture inspired from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f and https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    and https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
    '''

    def __init__(self, pre_trained_model, vocab_size, freeze_transformer=False):
        ''' 
        :param pre_trained_model: name of pre-trained Transformer model object
        :param vocab_size: length of Transformer tokenizer - can differ from pre-trained Transformer tokenizer when special tokens are added
        '''
        super(TransformerClassifier1, self).__init__()

        self.num_classes = 1  # binary classification

        # Instantiate Transformer model
        self.transformer = AutoModel.from_pretrained(pre_trained_model)
        self.transformer.resize_token_embeddings(vocab_size)  # adjust vocab size of pre-trained Transformer for special tokens that were added during tokenization
        self.transformer_hidden_size = 768
        self.hidden_size = 50 

        # Instantiate classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # dropout layer for regularization
            nn.Linear(self.transformer_hidden_size, self.hidden_size), 
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
            # sigmoid layer is automatically calles in torch.nn.BCEWithLogitsLoss
        )
        
        # Freeze Transformer model to prevent fine-tuning Transformer and instead only train the classifier
        if freeze_transformer:
            logging.info("Transformer layers frozen during training.")
            for param in self.transformer.parameters():
                param.requires_grad = False
                

    def forward(self, input_ids, attention_mask):
        ''' 
        Feed input from Transformer tokenizer sequences first to Transformer, then to classifier to compute logits
        :param input_ids (torch.Tensor): input ids from Transformer pretrained tokenizer with shape (batch_size, max_length), the token embeddings, padded until max len
        :param attention_mask (torch.Tensor): binary tensor from Transformer tokenizer with shape (batch_size, max_length), indicating position of the padded indices so that the model does not attend to them
        :return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''

        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs['last_hidden_state'][:,0,:]  # take last hidden state of [CLS] token for classification task
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(cls_embedding)

        return logits



# model1_1: two linear layers for classifier, tanh activation, dropout 0.3
class TransformerClassifier1_1(nn.Module):
    ''' Bert Classifier; architecture inspired from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f and https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    and https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
    '''

    def __init__(self, pre_trained_model, vocab_size, freeze_transformer=False):
        ''' 
        :param pre_trained_model: name of pre-trained Transformer model object
        :param vocab_size: length of Transformer tokenizer - can differ from pre-trained Transformer tokenizer when special tokens are added
        '''
        super(TransformerClassifier1_1, self).__init__()

        self.num_classes = 1  # binary classification

        # Instantiate Transformer model
        self.transformer = AutoModel.from_pretrained(pre_trained_model)
        self.transformer.resize_token_embeddings(vocab_size)  # adjust vocab size of pre-trained Tranformer for special tokens that were added during tokenization
        self.transformer_hidden_size = 768
        self.hidden_size = 50 

        # Instantiate classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # dropout layer for regularization
            nn.Linear(self.transformer_hidden_size, self.hidden_size), 
            nn.Tanh(),  #nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
            # sigmoid layer is automatically calles in torch.nn.BCEWithLogitsLoss
        )
        
        # Freeze Transformer model to prevent fine-tuning Transformer and instead only train the classifier
        if freeze_transformer:
            logging.info("Transformer layers frozen during training.")
            for param in self.transformer.parameters():
                param.requires_grad = False
                

    def forward(self, input_ids, attention_mask):
        ''' 
        Feed input from Transformer tokenizer sequences first to Transformer, then to classifier to compute logits
        :param input_ids (torch.Tensor): input ids from Transformer pretrained tokenizer with shape (batch_size, max_length), the token embeddings, padded until max len
        :param attention_mask (torch.Tensor): binary tensor from Transformer tokenizer with shape (batch_size, max_length), indicating position of the padded indices so that the model does not attend to them
        :return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''

        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs['last_hidden_state'][:,0,:]  # take last hidden state of [CLS] token for classification task
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(cls_embedding)

        return logits


# model1_2: two linear layers for classifier, relu activation, dropout 0.1
class TransformerClassifier1_2(nn.Module):
    ''' Bert Classifier; architecture inspired from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f and https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    and https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
    '''

    def __init__(self, pre_trained_model, vocab_size, freeze_transformer=False):
        ''' 
        :param pre_trained_model: name of pre-trained Transformer model object
        :param vocab_size: length of Transformer tokenizer - can differ from pre-trained Transformer tokenizer when special tokens are added
        '''
        super(TransformerClassifier1_2, self).__init__()

        self.num_classes = 1  # binary classification

        # Instantiate BERT model
        self.transformer = AutoModel.from_pretrained(pre_trained_model)
        self.transformer.resize_token_embeddings(vocab_size)  # adjust vocab size of pre-trained Transformer for special tokens that were added during tokenization
        self.transformer_hidden_size = 768
        self.hidden_size = 50 

        # Instantiate classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),  # dropout layer for regularization
            nn.Linear(self.transformer_hidden_size, self.hidden_size), 
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
            # sigmoid layer is automatically calles in torch.nn.BCEWithLogitsLoss
        )
        
        # Freeze Transformer model to prevent fine-tuning Transformer and instead only train the classifier
        if freeze_transformer:
            logging.info("Transformer layers frozen during training.")
            for param in self.transformer.parameters():
                param.requires_grad = False
                

    def forward(self, input_ids, attention_mask):
        ''' 
        Feed input from Transformer tokenizer sequences first to Transformer, then to classifier to compute logits
        :param input_ids (torch.Tensor): input ids from Transformer pretrained tokenizer with shape (batch_size, max_length), the token embeddings, padded until max len
        :param attention_mask (torch.Tensor): binary tensor from Transformer tokenizer with shape (batch_size, max_length), indicating position of the padded indices so that the model does not attend to them
        :return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''

        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs['last_hidden_state'][:,0,:]  # take last hidden state of [CLS] token for classification task
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(cls_embedding)

        return logits


# model1_3: two linear layers for classifier, tanh activation, dropout 0.1
class TransformerClassifier1_3(nn.Module):
    ''' Bert Classifier; architecture inspired from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f and https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    and https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
    '''

    def __init__(self, pre_trained_model, vocab_size, freeze_transformer=False):
        ''' 
        :param pre_trained_model: name of pre-trained Transformer model object
        :param vocab_size: length of Transformer tokenizer - can differ from pre-trained Transformer tokenizer when special tokens are added
        '''
        super(TransformerClassifier1_3, self).__init__()

        self.num_classes = 1  # binary classification

        # Instantiate Transformer model
        self.transformer = AutoModel.from_pretrained(pre_trained_model)
        self.transformer.resize_token_embeddings(vocab_size)  # adjust vocab size of pre-trained Transformer for special tokens that were added during tokenization
        self.transformer_hidden_size = 768
        self.hidden_size = 50 

        # Instantiate classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),  # dropout layer for regularization
            nn.Linear(self.transformer_hidden_size, self.hidden_size), 
            nn.Tanh(),  #nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
            # sigmoid layer is automatically calles in torch.nn.BCEWithLogitsLoss
        )
        
        # Freeze Transformer model to prevent fine-tuning Transformer and instead only train the classifier
        if freeze_transformer:
            logging.info("Transformer layers frozen during training.")
            for param in self.transformer.parameters():
                param.requires_grad = False
                

    def forward(self, input_ids, attention_mask):
        ''' 
        Feed input from Transformer tokenizer sequences first to Transformer, then to classifier to compute logits
        :param input_ids (torch.Tensor): input ids from Transformer pretrained tokenizer with shape (batch_size, max_length), the token embeddings, padded until max len
        :param attention_mask (torch.Tensor): binary tensor from Transformer tokenizer with shape (batch_size, max_length), indicating position of the padded indices so that the model does not attend to them
        :return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''

        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs['last_hidden_state'][:,0,:]  # take last hidden state of [CLS] token for classification task
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(cls_embedding)

        return logits


# model2: three linear layers for classifier
class TransformerClassifier2(nn.Module):
    ''' Bert Classifier; architecture inspired from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f and https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    and https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
    '''

    def __init__(self, pre_trained_model, vocab_size, freeze_transformer=False):
        ''' 
        :param pre_trained_model: name of pre-trained Transformer model object
        :param vocab_size: length of Transformer tokenizer - can differ from pre-trained Transformer tokenizer when special tokens are added
        '''
        super(TransformerClassifier2, self).__init__()

        self.num_classes = 1  # binary classification

        # Instantiate Transformer model
        self.transformer = AutoModel.from_pretrained(pre_trained_model)
        self.transformer.resize_token_embeddings(vocab_size)  # adjust vocab size of pre-trained Transformer for special tokens that were added during tokenization
        self.transformer_hidden_size = 768
        self.hidden_size_1 = 300 
        self.hidden_size_2 = 50 

        # Instantiate classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # dropout layer for regularization
            nn.Linear(self.transformer_hidden_size, self.hidden_size_1), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size_1, self.hidden_size_2), 
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.num_classes)
            # sigmoid layer is automatically calles in torch.nn.BCEWithLogitsLoss
        )
        
        # Freeze Transformer model to prevent fine-tuning Transformer and instead only train the classifier
        if freeze_transformer:
            logging.info("Transformer layers frozen during training.")
            for param in self.transformer.parameters():
                param.requires_grad = False
                

    def forward(self, input_ids, attention_mask):
        ''' 
        Feed input from Transformer tokenizer sequences first to Transformer, then to classifier to compute logits
        :param input_ids (torch.Tensor): input ids from Transformer pretrained tokenizer with shape (batch_size, max_length), the token embeddings, padded until max len
        :param attention_mask (torch.Tensor): binary tensor from Transformer tokenizer with shape (batch_size, max_length), indicating position of the padded indices so that the model does not attend to them
        :return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''

        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs['last_hidden_state'][:,0,:]  # take last hidden state of [CLS] token for classification task
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(cls_embedding)

        return logits


# model2_1: three linear layers for classifier, tanh instead of relu activation
class TransformerClassifier2_1(nn.Module):
    ''' Bert Classifier; architecture inspired from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f and https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb
    and https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
    '''

    def __init__(self, pre_trained_model, vocab_size, freeze_transformer=False):
        ''' 
        :param pre_trained_model: name of pre-trained Transformer model object
        :param vocab_size: length of Transformer tokenizer - can differ from pre-trained Transformer tokenizer when special tokens are added
        '''
        super(TransformerClassifier2_1, self).__init__()

        self.num_classes = 1  # binary classification

        # Instantiate Transformer model
        self.transformer = AutoModel.from_pretrained(pre_trained_model)
        self.transformer.resize_token_embeddings(vocab_size)  # adjust vocab size of pre-trained Transformer for special tokens that were added during tokenization
        self.transformer_hidden_size = 768
        self.hidden_size_1 = 300 
        self.hidden_size_2 = 50 

        # Instantiate classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # dropout layer for regularization
            nn.Linear(self.transformer_hidden_size, self.hidden_size_1), 
            nn.Tanh(), #nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size_1, self.hidden_size_2), 
            nn.Tanh(), # nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.num_classes)
            # sigmoid layer is automatically calles in torch.nn.BCEWithLogitsLoss
        )
        
        # Freeze Transformer model to prevent fine-tuning Transformer and instead only train the classifier
        if freeze_transformer:
            logging.info("Transformer layers frozen during training.")
            for param in self.transformer.parameters():
                param.requires_grad = False
                

    def forward(self, input_ids, attention_mask):
        ''' 
        Feed input from Transformer tokenizer sequences first to Transformer, then to classifier to compute logits
        :param input_ids (torch.Tensor): input ids from Transformer pretrained tokenizer with shape (batch_size, max_length), the token embeddings, padded until max len
        :param attention_mask (torch.Tensor): binary tensor from Transformer tokenizer with shape (batch_size, max_length), indicating position of the padded indices so that the model does not attend to them
        :return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''

        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs['last_hidden_state'][:,0,:]  # take last hidden state of [CLS] token for classification task
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(cls_embedding)

        return logits
