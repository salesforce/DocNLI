# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

export BATCHSIZE=32 #32 is also ok
export EPOCHSIZE=5
export LEARNINGRATE=1e-6
export MAXLEN=128

# CUDA_VISIBLE_DEVICES=0 python -u train_docNLI_RoBERTa_storeModel.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --data_label 'ANLI CNNDailyMail' \
#     --seed 42 > log.160k.ANLI.and.CNNDailyMail.docNLI.store.model.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=3 python -u train_docNLI_RoBERTa_storeModel.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --data_label 'ANLI CNNDailyMail DUC' \
#     --seed 42 > log.160k.ANLI.and.CNNDailyMail.and.DUC.docNLI.store.model.txt 2>&1 &
#
#
# CUDA_VISIBLE_DEVICES=5 python -u train_docNLI_RoBERTa_storeModel.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --data_label 'ANLI CNNDailyMail DUC Curation' \
#     --seed 42 > log.160k.ANLI.and.CNNDailyMail.and.DUC.and.Curation.docNLI.store.model.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=7 python -u train_docNLI_RoBERTa_storeModel.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --data_label 'ANLI CNNDailyMail DUC Curation SQUAD' \
#     --seed 42 > log.160k.ANLI.and.CNNDailyMail.and.DUC.and.Curation.and.SQUAD.docNLI.store.model.txt 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u train_docNLI_RoBERTa_storeModel.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'ANLI' \
    --seed 42 > log.160k.ANLI.docNLI.store.model.txt 2>&1 &
