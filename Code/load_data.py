# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json_lines
import codecs
import random
import json
from transformers.data.processors.utils import InputExample


def deal_with_block(block_line_list, filter_label_set, hypo_only=False):
    examples = []
    premise = ''

    if not block_line_list[0].startswith('document>>'):
        return [], 0, 0
    first_line_parts = block_line_list[0].strip().split('\t')
    # premise = first_line_parts[1].strip()
    premise = first_line_parts[2].strip()
    if len(premise) == 0:
        return [], 0, 0

    pos_hypo_list = []
    neg_hypo_list = []
    for line in block_line_list[1:]:
        if len(line.strip())>0:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                filter_label = parts[1].strip()
                if parts[0] == 'positive>>':
                    pos_hypo = parts[2].strip() # harsh version
                    if filter_label not in filter_label_set and len(pos_hypo) >0:
                        pos_hypo_list.append(pos_hypo)
                elif parts[0] == 'negative>>':
                    neg_hypo = parts[2].strip()
                    '''we do not need filter any negative summary in train, dev, and test'''
                    if len(neg_hypo) >0:
                        neg_hypo_list.append(neg_hypo)


    for pos_hypo in pos_hypo_list:

        if hypo_only:
            examples.append(InputExample(guid='ex', text_a=pos_hypo, text_b=None, label='entailment'))
        else:
            examples.append(InputExample(guid='ex', text_a=premise, text_b=pos_hypo, label='entailment'))

    for neg_hypo in neg_hypo_list:

        if hypo_only:
            examples.append(InputExample(guid='ex', text_a=neg_hypo, text_b=None, label='not_entailment'))
        else:
            examples.append(InputExample(guid='ex', text_a=premise, text_b=neg_hypo, label='not_entailment'))

    return examples, len(pos_hypo_list), len(neg_hypo_list)

def get_summary_examples(path, prefix, hypo_only=False):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    # path = '/export/home/Dataset/para_entail_datasets/DUC/'
    filename = path+prefix+'_in_entail.harsh.v2.txt'
    print('loading ...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')

    examples = []
    pos_size = 0
    neg_size = 0

    if prefix == 'train':
        filter_label_set = set([])
    else:
        filter_label_set = set(['#FakePlus2FakeIsPos#>>'])

    block_line_list = []
    for line in readfile:
        if line.strip().startswith('document>>'):
            '''if a block is ready'''
            if len(block_line_list)> 0:
                example_from_block, pos_size_block, neg_size_block = deal_with_block(block_line_list, filter_label_set, hypo_only=hypo_only)
                examples+=example_from_block
                pos_size+=pos_size_block
                neg_size+=neg_size_block
                '''this is especially for CNN'''
                # if len(examples) >=160000:
                #     break

            '''start a new block'''
            block_line_list=[line.strip()]
        else:
            if len(line.strip()) > 0: # in case some noice lines are emtpy
                block_line_list.append(line.strip())

    print('>>pos:neg: ', pos_size, neg_size)
    print('size:', len(examples))
    return examples, pos_size

# def get_DUC_examples(prefix, hypo_only=False):
#     #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
#     path = '/export/home/Dataset/para_entail_datasets/DUC/'
#     filename = path+prefix+'_in_entail.harsh.txt'
#     print('loading DUC...', filename)
#     readfile = codecs.open(filename, 'r', 'utf-8')
#
#     examples = []
#     pos_size = 0
#     neg_size = 0
#
#     if prefix == 'train':
#         filter_label_set = set([])
#     else:
#         filter_label_set = set(['#neg2negIsPos#>>', '#negInserted2negIsPos#>>'])
#
#     block_line_list = []
#     for line in readfile:
#         if line.strip().startswith('document>>'):
#             '''if a block is ready'''
#             if len(block_line_list)> 0:
#                 example_from_block, pos_size_block, neg_size_block = deal_with_block(block_line_list, filter_label_set, hypo_only=hypo_only)
#                 examples+=example_from_block
#                 pos_size+=pos_size_block
#                 neg_size+=neg_size_block
#
#             else:
#                 block_line_list.append(line.strip())
#         else:
#             if len(line.strip()) > 0: # in case some noice lines are emtpy
#                 block_line_list.append(line.strip())
#
#     print('>>pos:neg: ', pos_size, neg_size)
#     print('DUC size:', len(examples))
#     return examples, pos_size



def get_DUC_examples(prefix, hypo_only=False):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/DUC/'
    filename = path+prefix+'_in_entail.harsh.txt'
    print('loading DUC...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    guid_id = -1
    pos_size = 0
    neg_size = 0
    for line in readfile:
        if len(line.strip()) == 0:
            start = False
            '''to avoid that no examples loaded in this block, but the premise was falsely kept for the next block'''
            premise = ''
        else:
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                start = True
                premise = parts[1].strip()
            elif parts[0] == 'positive>>':
                guid_id+=1
                # pos_hypo = parts[1].strip()
                pos_hypo = parts[2].strip() # harsh version
                if len(premise) == 0 or len(pos_hypo)==0:
                    # print('DUC premise:', premise)
                    # print('hypothesis:', pos_hypo)
                    continue
                if prefix !='train' and parts[1].strip() == '#neg2negIsPos#>>' or parts[1].strip()=='#negInserted2negIsPos#>>':
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=pos_hypo, text_b=None, label='entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()
                if len(premise) == 0 or len(neg_hypo)==0:
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=None, label='not_entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                neg_size+=1
                # if filename.find('train_in_entail') > -1:
                #     examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                #     neg_size+=1
                # else:
                #     rand_prob = random.uniform(0, 1)
                #     if rand_prob > 3/4:
                #         examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                #         neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    print('DUC size:', len(examples))
    # if prefix == 'train':
    #     new_examples = []
    #     new_pos_size = 0
    #     new_neg_size = 0
    #     for ex in examples:
    #         if ex.label == 'not_entailment':
    #             if random.uniform(0.0, 1.0) <= pos_size/neg_size:
    #                 new_examples.append(ex)
    #                 new_neg_size+=1
    #         else:
    #             new_examples.append(ex)
    #             new_pos_size+=1
    #     print('>>new pos:neg: ', new_pos_size, new_neg_size)
    #     return new_examples, new_pos_size
    # else:
    #     return examples, pos_size
    return examples, pos_size

def get_Curation_examples(prefix, hypo_only=False):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/Curation/'
    filename = path+prefix+'_in_entail.harsh.txt'
    print('loading Curation...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    guid_id = -1
    pos_size = 0
    neg_size = 0
    for line in readfile:
        if len(line.strip()) == 0:
            start = False
            premise = ''
        else:
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                start = True
                premise = parts[1].strip()
            elif parts[0] == 'positive>>':
                guid_id+=1
                pos_hypo = parts[2].strip()
                if len(premise) == 0 or len(pos_hypo)==0:
                    continue

                if prefix !='train' and parts[1].strip() == '#neg2negIsPos#>>' or parts[1].strip()=='#negInserted2negIsPos#>>':
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=pos_hypo, text_b=None, label='entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()
                if len(premise) == 0 or len(neg_hypo)==0:
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=None, label='not_entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    print('Curation size:', len(examples))
    # if prefix == 'train':
    #     new_examples = []
    #     new_pos_size = 0
    #     new_neg_size = 0
    #     for ex in examples:
    #         if ex.label == 'not_entailment':
    #             if random.uniform(0.0, 1.0) <= pos_size/neg_size:
    #                 new_examples.append(ex)
    #                 new_neg_size+=1
    #         else:
    #             new_examples.append(ex)
    #             new_pos_size+=1
    #     print('>>new pos:neg: ', new_pos_size, new_neg_size)
    #     return new_examples, new_pos_size
    # else:
    #     return examples, pos_size
    return examples, pos_size



def get_CNN_DailyMail_examples(prefix, hypo_only=False):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/CNN_DailyMail/'
    filename = path+prefix+'_in_entail.harsh.txt'
    print('loading CNN_DailyMail...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    guid_id = -1
    pos_size = 0
    neg_size = 0

    load_size = 0
    for line in readfile:
        if len(line.strip()) == 0:
            start = False
            premise = ''
            load_size+=1
            '''we currently only use 60K as training set in CNN/DailyMail'''
            if load_size == 600000 and prefix == 'train':
                break
        else:
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                start = True
                premise = parts[1].strip()
            elif parts[0] == 'positive>>':
                guid_id+=1
                pos_hypo = parts[2].strip()
                if len(premise) == 0 or len(pos_hypo)==0:
                    continue

                if prefix !='train' and parts[1].strip() == '#neg2negIsPos#>>' or parts[1].strip()=='#negInserted2negIsPos#>>':
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=pos_hypo, text_b=None, label='entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()

                # if filename.find('train_in_entail') > -1:
                if len(premise) == 0 or len(neg_hypo)==0:
                    # print('CNN premise:', premise)
                    # print('neg_hypo:', neg_hypo)
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=None, label='not_entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                neg_size+=1


    print('>>pos:neg: ', pos_size, neg_size)
    print('CNN size:', len(examples))
    # if prefix == 'train':
    #     new_examples = []
    #     new_pos_size = 0
    #     new_neg_size = 0
    #     for ex in examples:
    #         if ex.label == 'not_entailment':
    #             if random.uniform(0.0, 1.0) <= pos_size/neg_size:
    #                 new_examples.append(ex)
    #                 new_neg_size+=1
    #         else:
    #             new_examples.append(ex)
    #             new_pos_size+=1
    #     print('>>new pos:neg: ', new_pos_size, new_neg_size)
    #     return new_examples, new_pos_size
    # else:
    #     return examples, pos_size
    return examples, pos_size


def get_SQUAD_examples(prefix, hypo_only=False):
    path = '/export/home/Dataset/para_entail_datasets/SQUAD/'
    filename = path+prefix+'.txt'
    print('loading SQUAD...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    guid_id = 0
    pos_size = 0
    neg_size = 0
    examples = []
    for line in readfile:
        guid_id+=1
        parts = line.strip().split('\t')
        if len(parts) ==3:
            premise = parts[1]
            hypothesis = parts[2]
            label = 'entailment' if parts[0] == 'entailment' else 'not_entailment'
            if len(premise) == 0 or len(hypothesis)==0:
                continue

            if label == 'entailment':
                pos_size+=1
            else:
                neg_size+=1
            if hypo_only:
                examples.append(InputExample(guid=prefix+str(guid_id), text_a=hypothesis, text_b=None, label=label))
            else:
                examples.append(InputExample(guid=prefix+str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('SQUAD size:', len(examples))
    # if prefix == 'train':
    #     new_examples = []
    #     new_pos_size = 0
    #     new_neg_size = 0
    #     for ex in examples:
    #         if ex.label == 'not_entailment':
    #             if random.uniform(0.0, 1.0) <= pos_size/neg_size:
    #                 new_examples.append(ex)
    #                 new_neg_size+=1
    #         else:
    #             new_examples.append(ex)
    #             new_pos_size+=1
    #     print('>>new pos:neg: ', new_pos_size, new_neg_size)
    #     return new_examples, new_pos_size
    # else:
    #     return examples, pos_size
    return examples, pos_size



def get_MCTest_examples(prefix, hypo_only=False):
    path = '/export/home/Dataset/para_entail_datasets/MCTest/'
    filename = path+prefix+'_in_entail.txt'
    print('loading MCTest...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    guid_id = 0
    pos_size = 0
    neg_size = 0
    examples = []
    for line in readfile:
        guid_id+=1
        parts = line.strip().split('\t')
        if len(parts) ==3:
            premise = parts[1]
            hypothesis = parts[2]
            label = 'entailment' if parts[0] == 'entailment' else 'not_entailment'
            # if label == 'entailment':
            #     pos_size+=1
            if len(premise) == 0 or len(hypothesis)==0:
                # print('MCTest premise:', premise)
                # print('hypothesis:', hypothesis)
                continue

            if label == 'entailment':
                pos_size+=1
            else:
                neg_size+=1
            if hypo_only:
                examples.append(InputExample(guid=prefix+str(guid_id), text_a=hypothesis, text_b=None, label=label))
            else:
                examples.append(InputExample(guid=prefix+str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('MCTest size:', len(examples))
    # if prefix == 'train':
    #     new_examples = []
    #     new_pos_size = 0
    #     new_neg_size = 0
    #     for ex in examples:
    #         if ex.label == 'not_entailment':
    #             if random.uniform(0.0, 1.0) <= pos_size/neg_size:
    #                 new_examples.append(ex)
    #                 new_neg_size+=1
    #         else:
    #             new_examples.append(ex)
    #             new_pos_size+=1
    #     print('>>new pos:neg: ', new_pos_size, new_neg_size)
    #     return new_examples, new_pos_size
    # else:
    #     return examples, pos_size
    return examples, pos_size

def get_FEVER_examples(prefix, hypo_only=False):
    '''
    train_fitems.jsonl, dev_fitems.jsonl, test_fitems.jsonl
    dev_fitems.label.recovered.jsonl
    '''
    examples = []
    path = '/export/home/Dataset/para_entail_datasets/nli_FEVER/nli_fever/'
    filename = path+prefix+'_fitems.jsonl'
    if prefix == 'test' or prefix == 'dev':
        filename = path+'dev_fitems.label.recovered.jsonl'
    print('loading FEVER...', filename)
    guid_id = 0
    pos_size = 0
    with open(filename, 'r') as f:
        for line in json_lines.reader(f):
            guid_id+=1
            premise = line.get('context')
            hypothesis = line.get('query')
            label = 'entailment' if line.get('label') == 'SUPPORTS' else 'not_entailment'
            if label == 'entailment':
                pos_size+=1
            if len(premise) == 0 or len(hypothesis)==0:
                continue

            if hypo_only:
                examples.append(InputExample(guid=str(guid_id), text_a=hypothesis, text_b=None, label=label))
            else:
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('FEVER size:', len(examples))
    return examples, pos_size

def get_ANLI_examples(prefix, hypo_only=False):
    folders = ['R1', 'R2', 'R3']
    examples = []
    guid_id = 0
    pos_size = 0
    neg_size = 0
    path = '/export/home/Dataset/para_entail_datasets/ANLI/anli_v0.1/'
    for folder in folders:
        filename = path+folder+'/'+prefix+'.jsonl'
        print('loading ANLI...', filename)
        with open(filename, 'r') as f:
            for line in json_lines.reader(f):
                guid_id+=1
                premise = line.get('context')
                hypothesis = line.get('hypothesis')
                label = 'entailment' if line.get('label') == 'e' else 'not_entailment'
                if len(premise) == 0 or len(hypothesis)==0:
                    continue
                if label == 'entailment':
                    pos_size+=1
                else:
                    neg_size+=1
                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=hypothesis, text_b=None, label=label))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('>>pos:neg: ', pos_size, neg_size)
    print('ANLI size:', len(examples))
    # if prefix == 'train':
    #     new_examples = []
    #     new_pos_size = 0
    #     new_neg_size = 0
    #     for ex in examples:
    #         if ex.label == 'not_entailment':
    #             if random.uniform(0.0, 1.0) <= pos_size/neg_size:
    #                 new_examples.append(ex)
    #                 new_neg_size+=1
    #         else:
    #             new_examples.append(ex)
    #             new_pos_size+=1
    #     print('>>new pos:neg: ', new_pos_size, new_neg_size)
    #     return new_examples, new_pos_size
    # else:
    #     return examples, pos_size
    return examples, pos_size




def load_harsh_data(prefix, need_data_list, hypo_only=False):

    # '''DUC'''
    # duc_examples, duc_pos_size = get_DUC_examples('train', hypo_only=hypo_only)
    # '''CNN'''
    # cnn_examples, cnn_pos_size = get_CNN_DailyMail_examples('train', hypo_only=hypo_only)
    # '''MCTest'''
    # mctest_examples, mctest_pos_size = get_MCTest_examples('train', hypo_only=hypo_only)
    # '''Curation'''
    # curation_examples, curation_pos_size = get_Curation_examples('train', hypo_only=hypo_only)
    # '''ANLI'''
    # anli_examples, anli_pos_size = get_ANLI_examples('train', hypo_only=hypo_only)
    # prefix = 'train'
    train_examples_list = []
    pos_size_list = []


    summary_path_list = [
                '/export/home/Dataset/para_entail_datasets/DUC/',
                '/export/home/Dataset/para_entail_datasets/Curation/',
                '/export/home/Dataset/para_entail_datasets/CNN_DailyMail/'
                ]
    for path in summary_path_list:
        summary_examples_i, summary_pos_size_i = get_summary_examples(path, prefix, hypo_only=hypo_only)
        train_examples_list.append(summary_examples_i)
        pos_size_list.append(summary_pos_size_i)

    # '''MCTest'''
    # mctest_examples, mctest_pos_size = get_MCTest_examples(prefix, hypo_only=hypo_only)
    # train_examples+=mctest_examples
    # pos_size+=mctest_pos_size

    '''SQUAD'''
    squada_examples, squada_pos_size = get_SQUAD_examples(prefix, hypo_only=hypo_only)
    train_examples_list.append(squada_examples)
    pos_size_list.append(squada_pos_size)


    '''ANLI'''
    anli_examples, anli_pos_size = get_ANLI_examples(prefix, hypo_only=hypo_only)
    train_examples_list.append(anli_examples)
    pos_size_list.append(anli_pos_size)

    data_label_list = ['DUC', 'Curation', 'CNNDailyMail', 'SQUAD', 'ANLI']
    assert len(data_label_list) == len(train_examples_list)
    train_examples = []
    pos_size = 0
    for data_label in need_data_list:
        train_examples+=train_examples_list[data_label_list.index(data_label)]
        pos_size+=pos_size_list[data_label_list.index(data_label)]

    print('train size:', len(train_examples), ' pos size:', pos_size, ' ratio:', pos_size/len(train_examples))


    return train_examples



def load_dev_data(hypo_only=False):
    '''test size: 125646  pos size: 14309; 11.38%'''
    '''DUC'''
    duc_examples, duc_pos_size = get_DUC_examples('dev', hypo_only=hypo_only)
    '''CNN'''
    cnn_examples, cnn_pos_size = get_CNN_DailyMail_examples('dev', hypo_only=hypo_only)
    '''SQUAD'''
    squada_examples, squada_pos_size  = get_SQUAD_examples('dev', hypo_only=hypo_only)
    '''Curation'''
    curation_examples, curation_pos_size = get_Curation_examples('dev', hypo_only=hypo_only)
    '''ANLI'''
    anli_examples, anli_pos_size = get_ANLI_examples('dev', hypo_only=hypo_only)

    dev_examples = (
                        duc_examples+
                        cnn_examples+
                        squada_examples+
                        curation_examples+
                        anli_examples
                        )
    pos_size = (
                duc_pos_size+
                cnn_pos_size+
                squada_pos_size+
                curation_pos_size+
                anli_pos_size
                )

    print('dev size:', len(dev_examples), ' pos size:', pos_size, ' ratio:', pos_size/len(dev_examples))
    return dev_examples


def load_test_data(hypo_only=False):
    '''test size: 125646  pos size: 14309; 11.38%'''
    '''DUC'''
    duc_examples, duc_pos_size = get_DUC_examples('test', hypo_only=hypo_only)
    '''CNN'''
    cnn_examples, cnn_pos_size = get_CNN_DailyMail_examples('test', hypo_only=hypo_only)
    '''SQUAD'''
    squada_examples, squada_pos_size  = get_SQUAD_examples('test', hypo_only=hypo_only)
    '''Curation'''
    curation_examples, curation_pos_size = get_Curation_examples('test', hypo_only=hypo_only)
    '''ANLI'''
    anli_examples, anli_pos_size = get_ANLI_examples('test', hypo_only=hypo_only)

    test_examples = (
                        duc_examples+
                        cnn_examples+
                        squada_examples+
                        curation_examples+
                        anli_examples
                        )
    pos_size = (
                duc_pos_size+
                cnn_pos_size+
                squada_pos_size+
                curation_pos_size+
                anli_pos_size
                )

    print('test size:', len(test_examples), ' pos size:', pos_size, ' ratio:', pos_size/len(test_examples))
    return test_examples


def load_DocNLI(prefix, hypo_only=False):
    readfile = codecs.open('/export/home/Dataset/para_entail_datasets/'+prefix+'.json', 'r', 'utf-8')

    data = json.load(readfile)
    examples = []
    for dic in data:
        premise = dic.get('premise')
        hypothesis = dic.get('hypothesis')
        label  = dic.get('label')
        if hypo_only:
            examples.append(InputExample(guid='ex', text_a=hypothesis, text_b=None, label=label))
        else:
            examples.append(InputExample(guid='ex', text_a=premise, text_b=hypothesis, label=label))
    return examples

if __name__ == "__main__":
    # load_train_data()
    load_test_data()

    '''
    train size: 1404446  pos size: 304352
    test size: 123857  pos size: 20975
    '''
