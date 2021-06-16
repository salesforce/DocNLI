# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

export BATCHSIZE=32 #32 is also ok
export EPOCHSIZE=5
export LEARNINGRATE=1e-6


# CUDA_VISIBLE_DEVICES=3 python -u train_SciTail_2_SciTail_BERT.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length 128 \
#     --seed 42 > log.BERT.scitail.2.scitail.seed.42.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=4 python -u train_SciTail_2_SciTail_BERT.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length 128 \
#     --seed 16 > log.BERT.scitail.2.scitail.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train_SciTail_2_SciTail_BERT.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 32 > log.BERT.scitail.2.scitail.seed.32.txt 2>&1 &
