# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

export BATCHSIZE=4 #32 is also ok
export EPOCHSIZE=5
export LEARNINGRATE=1e-6
export MAXLEN=512


CUDA_VISIBLE_DEVICES=6 python -u train_docNLI_2_FEVER_RoBERTa.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'DUC' \
    --seed 42 > log.DUC.docNLI.2.FEVER.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 python -u train_docNLI_2_FEVER_RoBERTa.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'CNNDailyMail' \
    --seed 42 > log.CNNDailyMail.docNLI.2.FEVER.txt 2>&1 &
