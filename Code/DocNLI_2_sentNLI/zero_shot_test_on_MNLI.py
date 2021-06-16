# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax

from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.models.roberta.modeling_roberta import RobertaModel#RobertaForSequenceClassification


p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from load_data import load_harsh_data, get_FEVER_examples

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'

def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('starting model storing....')
    # model.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    # tokenizer.save_pretrained(output_dir)
    print('store succeed')

class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single



class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_MNLI_train_and_dev(self, train_filename, dev_filename_list):
        '''
        classes: ["entailment", "neutral", "contradiction"]
        '''
        examples_per_file = []
        for filename in [train_filename]+dev_filename_list:
            examples=[]
            readfile = codecs.open(filename, 'r', 'utf-8')
            line_co=0
            for row in readfile:
                if line_co>0:
                    line=row.strip().split('\t')
                    guid = "train-"+str(line_co-1)
                    # text_a = 'MNLI. '+line[8].strip()
                    text_a = line[8].strip()
                    text_b = line[9].strip()
                    label = line[-1].strip() #["entailment", "neutral", "contradiction"]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
            readfile.close()
            print('loaded  MNLI size:', len(examples))
            examples_per_file.append(examples)
        dev_examples = []
        for listt in examples_per_file[1:]:
            dev_examples+=listt
        return examples_per_file[0], dev_examples #train, dev

    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()










def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--data_label",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    args = parser.parse_args()


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))



    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    _, threeway_dev_examples = processor.get_MNLI_train_and_dev('/export/home/Dataset/glue_data/MNLI/train.tsv', ['/export/home/Dataset/glue_data/MNLI/dev_mismatched.tsv', '/export/home/Dataset/glue_data/MNLI/dev_matched.tsv'])
    '''preprocessing: binary classification, randomly sample 20k for testing data'''
    # train_examples = []
    # for ex in threeway_train_examples:
    #     if ex.label == 'neutral' or ex.label == 'contradiction':
    #         ex.label = 'neutral'
    #     train_examples.append(ex)
    # train_examples = train_examples[:10000]
    dev_examples = []
    for ex in threeway_dev_examples:
        if ex.label == 'neutral' or ex.label == 'contradiction':
            ex.label = 'not_entailment'
        dev_examples.append(ex)
    random.shuffle(dev_examples)
    test_examples = dev_examples[:13000]
    dev_examples = dev_examples[13000:]

    label_list = ["entailment", "not_entailment"]#, "contradiction"]
    num_labels = len(label_list)
    print('num_labels:', num_labels,  'dev size:', len(dev_examples), ' test size:', len(test_examples))

    # device = torch.device('cpu')
    model = RobertaForSequenceClassification(num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_epoch_0.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_epoch_1.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_epoch_2.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_epoch_3.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_epoch_0.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_epoch_1.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_epoch_2.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_epoch_3.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_Curation_epoch_0.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_Curation_epoch_1.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_Curation_epoch_2.pt', map_location=device))
    model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_CNNDailyMail_DUC_Curation_SQUAD_epoch_1.pt', map_location=device))

    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_epoch_0.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_epoch_1.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_DUC_epoch_0.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_DUC_epoch_1.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_DUC_Curation_epoch_0.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_DUC_Curation_epoch_1.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_DUC_Curation_SQUAD_epoch_0.pt', map_location=device))
    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/ANLI_CNNDailyMail_DUC_Curation_SQUAD_epoch_1.pt', map_location=device))

    # model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021/160k_ANLI_epoch_4.pt', map_location=device))
    model.to(device)


    '''load test set'''
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)



    '''
    start evaluate on dev set after this epoch
    '''
    model.eval()
    final_test_performance = evaluation(test_dataloader, device, model)
    print('final_test_performance:', final_test_performance)

def evaluation(dev_dataloader, device, model):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    gold_label_ids = []
    # print('Evaluating...')
    for input_ids, input_mask, segment_ids, label_ids in dev_dataloader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        gold_label_ids+=list(label_ids.detach().cpu().numpy())

        with torch.no_grad():
            logits = model(input_ids, input_mask)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        nb_eval_steps+=1
        print('eval_steps:', nb_eval_steps, '/', len(dev_dataloader))

    preds = preds[0]

    pred_probs = softmax(preds,axis=1)
    pred_label_ids = list(np.argmax(pred_probs, axis=1))

    gold_label_ids = gold_label_ids
    assert len(pred_label_ids) == len(gold_label_ids)
    hit_co = 0
    for k in range(len(pred_label_ids)):
        if pred_label_ids[k] == gold_label_ids[k]:
            hit_co +=1
    test_acc = hit_co/len(gold_label_ids)
    return test_acc


if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=1 python -u zero_shot_test_on_MNLI.py --task_name rte --do_train --do_lower_case --data_label DUC --num_train_epochs 20 --train_batch_size 32 --eval_batch_size 128 --learning_rate 1e-6 --max_seq_length 128 --seed 42


'''
