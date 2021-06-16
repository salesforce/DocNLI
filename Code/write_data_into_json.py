# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import codecs
from load_data import load_harsh_data, load_test_data, load_dev_data
import json

data_label = 'ANLI CNNDailyMail DUC Curation SQUAD'
# examples = load_harsh_data('train', data_label.split(),  hypo_only=False)
# examples = load_test_data()
examples = load_dev_data()
writefile = codecs.open("/export/home/Dataset/para_entail_datasets/dev.json", "w", 'utf-8')
writefile.write('[')
pos_size = 0
neg_size = 0
for id, ex in enumerate(examples):
    if len(ex.text_a.strip())>0 and len(ex.text_b.strip())>0:
        if ex.label == 'entailment':
            pos_size+=1
        else:
            neg_size+=1
        dictt= {'premise':ex.text_a, 'hypothesis':ex.text_b, 'label':ex.label}
        json.dump(dictt, writefile)
        if id < len(examples)-1:
            writefile.write(',\n')
        else:
            writefile.write(']')

writefile.close()
print('write over. pos_size:', pos_size, ' neg_size:', neg_size)
