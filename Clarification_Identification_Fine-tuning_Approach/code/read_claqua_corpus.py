import transformers
from transformers import AutoTokenizer
from transformers import AdamW  # use optimizer from Transformers library, not Pytorch
from transformers.optimization import get_linear_schedule_with_warmup  # use scheduler from Transformers library, not Pytorch

import logging

# Import Claqua preprocessing scripts for Single-Turn and Multi-Turn split of the corpus
from preprocess_claqua_single_turn import ClaquaCorpusSingleTurn, process_single_turn_corpus
from preprocess_claqua_multi_turn import ClaquaCorpusMultiTurn, process_multi_turn_corpus


def read_single_turn_train_dev_test(data_path: str,
                tokenizer: transformers,
                num_max_toks: int,
                context_sep_tok: str,
                entity_sep_tok: str,
                ):
    ''' Reads training, development and test data path to create instances of class ClaquaCorpusSingleTurn.

    :param data_path: Specifies the path to the CLAQUA corpus.
    :param tokenizer: A Huggingface transformers tokenizer to tokenize the corpus data.
    :param num_max_toks: An integer indicating the maximum number of tokens to encode with the tokenizer.
    :param context_sep_tok: A string specifying the special token which separates context from entity information for tokenization and Transformer model training.
    :param entity_sep_tok: A string specifying the special token which separates two entities in  the entity information for tokenization and Transformer model training.
    :param num_special_toks: An integer specifying the number of special tokens for the tokenizer.
    :param single_turn: A boolean indicating the corpus-split: single-turn or multi-turn. Default is set to single-turn.
    
    :return: Three ClaquaCorpusSingleTurn instances: the first holding the train corpus, the second the development and the third the test corpus.
    '''

    # Set corpus vars
    train_corpus, dev_corpus, test_corpus = None, None, None

    # Set single-turn corpus paths
    train_path = data_path + '/corpus_preprocessed/single-turn_train_classifier.csv'
    dev_path = data_path + '/corpus_preprocessed/single-turn_dev_classifier.csv'
    test_path = data_path + '/corpus_preprocessed/single-turn_test_classifier.csv'

    # Create single-turn corpus instances
    train_corpus = ClaquaCorpusSingleTurn()
    train_corpus.read_corpus_data(train_path, 'Single-turn train data corpus')

    dev_corpus = ClaquaCorpusSingleTurn()
    dev_corpus.read_corpus_data(dev_path, 'Single-turn development data corpus')

    test_corpus = ClaquaCorpusSingleTurn()
    test_corpus.read_corpus_data(test_path, 'Single-turn test data corpus')

    # Process all single-turn corpora: train, dev and test 
    corpora = [train_corpus, dev_corpus, test_corpus]

    for corpus in corpora:
        # Fill ClaquaCorpusSingleTurn instance
        process_single_turn_corpus(corpus, tokenizer=tokenizer, context_sep_tok=context_sep_tok, entity_sep_tok=entity_sep_tok, num_max_toks=num_max_toks)
        # Check if truncation was successful and no dataframe items exceeds max len
        logging.info(f"Customized truncation to {num_max_toks} tokens successful: {corpus.df.loc[corpus.df['num_enc_full_text'] > num_max_toks].empty}.")
        trunc_success=corpus.df.loc[corpus.df['num_enc_full_text'] > num_max_toks].empty
        if trunc_success == False:
            logging.info(f"Highest token number is {corpus.df.loc[corpus.df['num_enc_full_text'].idxmax()]['num_enc_full_text']}. Huggingface tokenizer will truncate to {num_max_toks} tokens.")
    

    return train_corpus, dev_corpus, test_corpus


def read_multi_turn_train_dev_test(data_path: str,
                tokenizer: transformers,
                num_max_toks: int,
                turn_sep_tok: str, 
                context_sep_tok: str,
                entity_sep_tok: str,
                ):
    ''' Reads training, development and test data path to create instances of class ClaquaCorpusMultiTurn.

    :param data_path: Specifies the path to the CLAQUA corpus.
    :param tokenizer: A Huggingface transformers tokenizer to tokenize the corpus data.
    :param num_max_toks: An integer indicating the maximum number of tokens to encode with the tokenizer.
    :param turn_sep_tok: A string specifying the special token which separates turn within the context (replacing the <EOS> token in the corpus) for tokenization and Transformer model training.
    :param context_sep_tok: A string specifying the special token which separates context from entity information for tokenization and Transformer model training.
    :param entity_sep_tok: A string specifying the special token which separates two entities in  the entity information for tokenization and Transformer model training.
    :param num_special_toks: An integer specifying the number of special tokens for the tokenizer.
    :param single_turn: A boolean indicating the corpus-split: single-turn or multi-turn. Default is set to single-turn.
    
    :return: Three ClaquaCorpusMultiTurn instances: the first holding the train corpus, the second the development and the third the test corpus.
    '''

    # Set corpus vars
    train_corpus, dev_corpus, test_corpus = None, None, None

    # Set multi-turn corpus paths
    train_path = data_path + '/corpus_preprocessed/multi-turn_train_classifier.csv'
    dev_path = data_path + '/corpus_preprocessed/multi-turn_dev_classifier.csv'
    test_path = data_path + '/corpus_preprocessed/multi-turn_test_classifier.csv'

    # Create multi-turn corpus instances
    train_corpus = ClaquaCorpusMultiTurn()
    train_corpus.read_corpus_data(train_path, 'Multi-turn train data corpus')

    dev_corpus = ClaquaCorpusMultiTurn()
    dev_corpus.read_corpus_data(dev_path, 'Multi-turn development data corpus')

    test_corpus = ClaquaCorpusMultiTurn()
    test_corpus.read_corpus_data(test_path, 'Multi-turn test data corpus')

    # Process all multi-turn corpora: train, dev and test 
    corpora = [train_corpus, dev_corpus, test_corpus]

    for corpus in corpora:
        # Fill ClaquaCorpusMultiTurn instance
        process_multi_turn_corpus(corpus, tokenizer=tokenizer, turn_sep_tok=turn_sep_tok, context_sep_tok=context_sep_tok, entity_sep_tok=entity_sep_tok, num_max_toks=num_max_toks)
        # Check if truncation was successful and no dataframe items exceeds max len
        logging.info(f"Customized truncation to {num_max_toks} tokens successful: {corpus.df.loc[corpus.df['num_enc_full_text'] > num_max_toks].empty}. ")
        trunc_success=corpus.df.loc[corpus.df['num_enc_full_text'] > num_max_toks].empty
        if trunc_success == False:
            logging.info(f"Highest token number is {corpus.df.loc[corpus.df['num_enc_full_text'].idxmax()]['num_enc_full_text']}. Huggingface tokenizer will truncate to {num_max_toks} tokens.")
    
    return train_corpus, dev_corpus, test_corpus


def read_train_dev_test(corpus_split: str,
                data_path: str,
                tokenizer: transformers,
                num_max_toks: int,
                turn_sep_tok: str, 
                context_sep_tok: str,
                entity_sep_tok: str,
                ):
    ''' Calls either single-turn or multi-turn read method to read training, development and test data path and create instances of either class ClaquaCorpusSingleTurn or class ClaquaCorpusMultiTurn.
    
    :param corpus_split: A string to specify which split of the corpus is used. Either 'single-turn' or 'multi-turn'. 'single-turn' is the default. All other string arguments will result in processing the multi-turn split of the corpus.
    :param data_path: Specifies the path to the CLAQUA corpus.
    :param tokenizer: A Huggingface transformers tokenizer to tokenize the corpus data.
    :param num_max_toks: An integer indicating the maximum number of tokens to encode with the tokenizer.
    :param turn_sep_tok: Only needed for multi-turn split. A string specifying the special token which separates turn within the context (replacing the <EOS> token in the corpus) for tokenization and Transformer model training.
    :param context_sep_tok: A string specifying the special token which separates context from entity information for tokenization and Transformer model training.
    :param entity_sep_tok: A string specifying the special token which separates two entities in  the entity information for tokenization and Transformer model training.
    
    :return: Three instances of either ClaquaCorpusSingleTurn or ClaquaCorpusMultiTurn: the first holding the train corpus, the second the development and the third the test corpus.
    '''

    train_corpus, dev_corpus, test_corpus = None, None, None

    if corpus_split == 'single-turn':  # Process single-turn split of corpus
        train_corpus, dev_corpus, test_corpus = read_single_turn_train_dev_test(data_path=data_path, tokenizer=tokenizer, num_max_toks=num_max_toks, context_sep_tok=context_sep_tok, entity_sep_tok=entity_sep_tok)
    else:  # Process single-turn split of corpus
        train_corpus, dev_corpus, test_corpus = read_multi_turn_train_dev_test(data_path=data_path, tokenizer=tokenizer, num_max_toks=num_max_toks, turn_sep_tok=turn_sep_tok, context_sep_tok=context_sep_tok, entity_sep_tok=entity_sep_tok)

    return train_corpus, dev_corpus, test_corpus