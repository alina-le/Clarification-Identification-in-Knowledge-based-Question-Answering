import transformers
from transformers import AutoTokenizer
import torch

import logging

import math
import numpy as np
import pandas as pd


class ClaquaCorpusSingleTurn:
    ''' class for processing Single-turn split of CLAQUA corpus: preprocessing of corpus for Transformer tokenization and modeling
    '''

    def __init__(self):
        ''' constructor for CLAQUA corpus multi-turn instance
        :param use_class_weight (bool): indicate whether class weights should be used (in model training) or not; if True, the class weight will be calculated during corpus preprocessing 
        '''
        # pandas dataframe of corpus class instance
        self.df = None  

        # class weights for positive class
        self.positive_class_weight = None

        # vars for tokenizer
        self.tokenizer = None  # Transformer huggingface tokenizer  

        # set Transformer default tokens: [CLS] and [SEP] token
        self.transformer_sep_tok = '[SEP]'  # Transformer [SEP] token
        self.transformer_cls_tok = '[CLS]'  # Transformer [CLS] token
        self.num_transformer_tok_positions = 2  # number of positions taken by Transformer toks: [CLS] text [SEP]
        
        # set special tokens to add to Transformer encoding for separating the input, in case not the Transformer default toks are used
        # single-turn context looks like: question_a (no turn separator needed)
        self.context_sep_tok = None  # Transformer special token to separate context and entity info (items look like: context - ent1 - ent2)
        self.entity_sep_tok = None  # Transformer special token to separate entity1 and entity2 (items look like: context - ent1 - ent2)
        self.special_toks = None  # set of special toks
        self.num_special_tok_positions = 2  # number of special tokens added to separate context and entities: context [CONTEXT_SEP] ent1 [ENTITY_SEP] ent2
        
        # set maximum number of tokens for Transformer model encoding; default BERT max toks = 512
        self.num_max_toks = None  


    # getter methods

    def get(self, idx):
        ''' 
        :return: dataframe row at idx
        '''
        return self.df.iloc[idx]
    
    def get_context(self, idx):
        ''' 
        :return: context column of dataframe row at idx
        '''
        return self.df.iloc[idx]['context']
    
    def get_ent1(self, idx):
        ''' 
        :return: entity1 column of dataframe row at idx
        '''
        return self.df.iloc[idx]['entity1']

    def get_ent2(self, idx):
        ''' 
        :return: entity2 column of dataframe row at idx
        '''
        return self.df.iloc[idx]['entity2']

    # read corpus methods and set vars

    def read_corpus_data(self, data_path, df_name):
        ''' reads corpus from csv file and stores it in pandas dataframe; if class weight should be used, calculates class weight for positive class
        '''
        self.df = pd.read_csv(data_path, sep=';')  # read corpus from csv file
        self.df.name = df_name  # set dataframe name 

        self.positive_class_weight = self.get_positive_class_weight()  # calculate class weight for positive class (dataset is imbalanced)

    def remove_entity_attrbts(self, ent_col_name):
        '''
        takes string entry from pandas df column; removes entity attributes from string with entity name, entity text and entity attributes; entity attributes are enclosed in '<S>' tag
        '''
        self.df[ent_col_name] = self.df[ent_col_name].str.replace(r'<S> .* <S>', '', regex=True)    # entity attributes in corpus are enclosed in '<S>' tag

    def set_tokenizer_vars(self, tokenizer, context_sep_tok, entity_sep_tok, num_max_toks):
        ''' set variables needed for specification of tokenizer details such that dataframe can be accustomed to it
        '''
        self.tokenizer =  tokenizer  # Transformer huggingface tokenizer  
        
        self.context_sep_tok = context_sep_tok  # Transformer special token to separate context and entity info
        self.entity_sep_tok = entity_sep_tok  # Transformer special token to separate entity1 and entity2
        
        self.num_max_toks = num_max_toks # maximum number of tokens for Transformer model encoding

    # calculcate class weights

    def get_positive_class_weight(self):
        ''' Calculate the class weight for the positive class in a binary classification problem, which is needed for torch.nn.BCEWithLogitsLoss
        :param class_labels (pandas.core.series.Series): column from pandas dataframe which holds the class labels of the datasets
        :return positive_weight (torch.Tensor): pos_weight for torch.nn.BCEWithLogitsLoss, the class weight for the positive class
        '''
        positive_weight = self.df[self.df['label'] == 0].shape[0] / self.df[self.df['label'] == 1].shape[0]  # divide nr of negative examples in dataset by nr of positive examples to get weight for the postive class
        return torch.Tensor([positive_weight])
    
    # helper methods for tokenizer, calculation and text merge operations on dataframe

    def encode_text(self, text_item):
        ''' apply Transformer tokenizer to text string
        :return: tokenizer object with input_ids, token_type_ids and attention_mask
        '''
        encoded_toks = self.tokenizer.encode_plus(text=text_item, return_tensors='pt')
        return encoded_toks

    def pad_encode_text(self, text_item):
        ''' apply Transformer tokenizer to text string with extended settings: using max length and padding
        :return: tokenizer object with input_ids, token_type_ids and attention_mask
        '''
        encoded_toks = self.tokenizer.encode_plus(text=text_item, truncation=True, max_length=self.num_max_toks, padding='max_length', return_tensors='pt')
        return encoded_toks    

    def get_num_encoded_toks(self, toks_encoded, num_toks_to_ignore=None):
        ''' counts the number of encoded tokens in Tokenizer-encoded sequence; subtracts the default Transformer tokens which are always needed for encoding
        :param toks_encoded (transformers.tokenization_utils_base.BatchEncoding): Tokenizer-encoded sequence
        :param num_toks_to_ignore: number of tokens to subtract from total number of encoded tokens; unless otherwise specified, num_toks_to_ignore is set to default value of the number of default Transformer toks; this is because these default tokens are always added to the sequence: [CLS] text [SEP], they are not manually added to the text that is encoded
        :return: number of encoded tokens
        '''
        if num_toks_to_ignore is None:  
            num_toks_to_ignore = self.num_transformer_tok_positions  # unless otherwise specified, num_toks_to_ignore is set to default value of the number of default Transformer toks: [CLS] and [SEP]
        num_toks = toks_encoded['input_ids'].shape[1]  # count number of encoded tokens from tokenizer object
        num_toks -= num_toks_to_ignore  # subtract a number of tokens (e.g. special tokens) that should be ignored when counting
        return num_toks

    def exceeds_max_len(self, cnt_toks_context_column, cnt_toks_ent1_column, cnt_toks_ent2_column):
        ''' calculate if max length for Transformer encoding is exceeded when entity1 and entity2 text are both fully encoded; thereby add number of special tokens which always need be included in the tokens
        :return: True if max length is exceeded --> needs truncation; False if max length is not exceeded --> does not need truncation
        '''
        num_toks_encoded = cnt_toks_context_column + cnt_toks_ent1_column + cnt_toks_ent2_column  # get total length of one text item, consisting of context and two entities, in its encoded state
        num_toks_encoded_with_transformer_toks = num_toks_encoded + self.num_transformer_tok_positions  # add number of default Transformer toks needed in every encoding to it: [CLS] context ent1 ent2 [SEP]
        num_toks_encoded_with_transformer_and_special_toks = num_toks_encoded_with_transformer_toks + self.num_special_tok_positions  # add number of special tokens to it: [CLS] context [CONTEXT_SEP] ent1 [ENTITY_SEP] ent2 [SEP]

        max_len_exceeded = num_toks_encoded_with_transformer_and_special_toks > self.num_max_toks  # check if length exceeds the max length of tokens to assign minus the special and Transformer tokens which are always needed
        return max_len_exceeded

    def calc_truncation_ratio(self, cnt_toks_context_column, cnt_toks_ent1_column, cnt_toks_ent2_column):
        ''' calculate ratio for number of tokens to use for each entity so the total does not exceed the max len
        :return: floating point ratio (not rounded but cut off)
        '''
        sum_entity_toks = cnt_toks_ent1_column + cnt_toks_ent2_column
        num_max_toks_without_context = self.num_max_toks - cnt_toks_context_column  # get number of tokens free to assign after subtracting context tokens from max num of tokens
        num_max_toks_without_context_transformer_special_toks = num_max_toks_without_context - self.num_transformer_tok_positions - self.num_special_tok_positions  #  from number of tokens free to assign after subtracting context, also subtract num of Transformer default and special tokens needed
        
        trunc_ratio = num_max_toks_without_context_transformer_special_toks / sum_entity_toks  # divide number of free token spots by number of needed token spots when encoding the full entities to get a ratio
        trunc_ratio_cut = str(trunc_ratio)[:5]  # cut ratio instead of rounding (number of encoded tokens should never exceed max length)

        return float(trunc_ratio_cut)
    
    def calc_trunc_cutoff(self, trunc_ratio, cnt_toks):
        ''' for the entity1 and entity2 texts which together with context exceed the Transformer max length, multiply their number of tokens by truncation ratio to get the number of free token spots; this can be used for cutting the encoded sequences
        :return: truncated length of the sequence, rounded down 
        '''
        num_free_toks = trunc_ratio * cnt_toks 
        rounded_num_free_toks = math.floor(num_free_toks)
        return rounded_num_free_toks

    def truncate_ent(self, enc_ent, trunc_cutoff):
        ''' cut entity encoding at truncation cutoff point (which is the number of free token spots available for this entity s.t. the total encoding of entity1, entity2 and context does not exceed the max len for Transformer encoding
        :return: weiß noch nicht
        '''
        trunc_cutoff = int(trunc_cutoff)  # need to transform rounded down float of token length to integer
        ent_tok_ids_without_special = enc_ent['input_ids'][0][1:-1]  # get encoded tokens for this entity, cut the [CLS] at the beginning and [SEP] token at the end
        ent_tok_ids_cut = ent_tok_ids_without_special[0:trunc_cutoff]  # truncate at cutoff point
        ent_toks_cut = self.tokenizer.decode(ent_tok_ids_cut) # decode tokens IDs back to words
        return ent_toks_cut   

    def merge_text_for_enc(self, context_column, ent1_column, ent2_column):
        ''' take context, entity1 and entity2 text (original versions or truncated version);
        concat them with context separator token and entity separator token s.t. tokenizer can encode them in next step (CLS and SEP token will be added automatically in encoding step)
        :return: full concated text with special tokens inbetween    
        '''
        # old (worked for BERT, because whitespace makes no difference there): full_text = context_column + " " + self.context_sep_tok + " " + ent1_column + " " + self.entity_sep_tok + " " + ent2_column  # put whitespace inbetween in case there is none and Transformer would cut the tokens incorrectly; tokenizer will take care of multiple whitespaces
        full_text = context_column + self.context_sep_tok + ent1_column + self.entity_sep_tok + ent2_column
        return full_text
        
    # methods for dataframe operations

    def apply_count_encode_text(self, text_column, enc_text_column, cnt_toks_column, num_ignore=None):
        ''' apply function to use tokenizer to encode to the text column of dataframe and then apply function to count the number of encoded tokens
        :return: encoded context tokens and number of encoded context tokens
        '''
        self.df[enc_text_column] = self.df.apply(lambda x: self.encode_text(x[text_column]), axis=1)
        self.df[cnt_toks_column] = self.df.apply(lambda x: self.get_num_encoded_toks(x[enc_text_column]), axis=1)
    
    def apply_get_truncation_ratio(self, needs_trunc_column, trunc_ratio_column, cnt_toks_context_column, cnt_toks_ent1_column, cnt_toks_ent2_column):
        ''' call methods for checking if text item row needs truncation, and if yes, calculate the truncation ratio
        '''
        self.df[needs_trunc_column] = self.df.apply(lambda x: self.exceeds_max_len(x[cnt_toks_context_column], x[cnt_toks_ent1_column], x[cnt_toks_ent2_column]), axis=1)
        self.df[trunc_ratio_column] = self.df.apply(lambda x: self.calc_truncation_ratio(x[cnt_toks_context_column], x[cnt_toks_ent1_column], x[cnt_toks_ent2_column]) if x[needs_trunc_column] is True else None, axis=1)

    def apply_entity_truncation(self, trunc_ent_column, enc_ent_column, needs_trunc_column, trunc_ratio_column, trunc_cutoff_ent_column, cnt_toks_ent_column):
        ''' apply calculation of cutoff point for too long sequences of entity1 and entity2 
        '''
        # get entity truncation cutoff point
        self.df[trunc_cutoff_ent_column] = self.df.apply(lambda x: self.calc_trunc_cutoff(x[trunc_ratio_column], x[cnt_toks_ent_column]) if x[needs_trunc_column] is True else None, axis=1)
        # truncate entity
        self.df[trunc_ent_column] = self.df.apply(lambda x: self.truncate_ent(x[enc_ent_column], x[trunc_cutoff_ent_column]) if x[needs_trunc_column] is True else None, axis=1)

    def apply_merge_text_and_encode(self, context_column, ent1_column, ent2_column, needs_trunc_column, ent1_trunc_column, ent2_trunc_column, encoded_final_column, cnt_encoded_final_column):
        ''' merge context, entity1 and entity2 texts, with special separator tokens inbetween; depending on whether they need truncation, use truncated version of entity1 and entity2
        '''
        # apply merge of texts depending on whether truncated or full texts of entity1 and entity2 are needed
        self.df['full_text_for_encoding'] = self.df.apply(lambda x: self.merge_text_for_enc(context_column=x[context_column], ent1_column=x[ent1_trunc_column], ent2_column=x[ent2_trunc_column]) if x[needs_trunc_column] is True else self.merge_text_for_enc(context_column=x[context_column], ent1_column=x[ent1_column], ent2_column=x[ent2_column]), axis=1)
        
        self.df[encoded_final_column] = self.df.apply(lambda x: self.encode_text(x['full_text_for_encoding']), axis=1) 

        self.df[cnt_encoded_final_column] = self.df.apply(lambda x: self.get_num_encoded_toks(x[encoded_final_column], num_toks_to_ignore=0), axis=1)

        self.df[encoded_final_column + '_padded'] = self.df.apply(lambda x: self.pad_encode_text(x['full_text_for_encoding']), axis=1)  # encoding final text needs max length and padding


def process_single_turn_corpus(corpus, tokenizer, context_sep_tok, entity_sep_tok, num_max_toks):
    logging.info(f'Processing {corpus.df.name}')

    # remove entity attributes in entity1 and entity2 from corpus, they are not needed at this point
    logging.info(f'\t Removing entity attributes ...')
    corpus.remove_entity_attrbts('entity1')
    corpus.remove_entity_attrbts('entity2')

    # set tokenizer vars for processing in dataframe
    corpus.set_tokenizer_vars(tokenizer=tokenizer, context_sep_tok=context_sep_tok, entity_sep_tok=entity_sep_tok, num_max_toks=num_max_toks)

    # encode context, entity1 and entity2
    logging.info(f'\t Encoding text with Transformer tokenizer...')
    corpus.apply_count_encode_text(text_column='context', enc_text_column='enc_context', cnt_toks_column='num_enc_context_toks')
    corpus.apply_count_encode_text(text_column='entity1', enc_text_column='enc_entity1', cnt_toks_column='num_enc_entity1_toks')
    corpus.apply_count_encode_text(text_column='entity2', enc_text_column='enc_entity2', cnt_toks_column='num_enc_entity2_toks')

    # check whether truncation is needed (for all entities whose text, together with context, exceeds the specified max len) and if yes, calculate the truncation ratio 
    logging.info(f'\t Truncating entities to specified maximum length of {num_max_toks}...')
    corpus.apply_get_truncation_ratio(trunc_ratio_column='truncation_ratio', needs_trunc_column='needs_truncation', cnt_toks_context_column='num_enc_context_toks', cnt_toks_ent1_column='num_enc_entity1_toks', cnt_toks_ent2_column='num_enc_entity2_toks')

    # apply entity truncation where needed, using the truncation ratio 
    corpus.apply_entity_truncation(trunc_ent_column='entity1_trunc_text', enc_ent_column='enc_entity1', trunc_cutoff_ent_column='trunc_cutoff_entity1', trunc_ratio_column='truncation_ratio', needs_trunc_column='needs_truncation', cnt_toks_ent_column='num_enc_entity1_toks')
    corpus.apply_entity_truncation(trunc_ent_column='entity2_trunc_text', enc_ent_column='enc_entity2', trunc_cutoff_ent_column='trunc_cutoff_entity2', trunc_ratio_column='truncation_ratio', needs_trunc_column='needs_truncation', cnt_toks_ent_column='num_enc_entity2_toks')

    # merge context with entities (truncated entities where needed, else the full entity text) and special tokens inbetween, encode the full text for Transformer model 
    logging.info(f'\t Merging context and truncated entity texts with Transformer special tokens, encode with padding...')
    corpus.apply_merge_text_and_encode(context_column='context', ent1_column='entity1', ent2_column='entity2', needs_trunc_column='needs_truncation', ent1_trunc_column='entity1_trunc_text', ent2_trunc_column='entity2_trunc_text', encoded_final_column='enc_full_text', cnt_encoded_final_column='num_enc_full_text')
    logging.info('Done! \n')
