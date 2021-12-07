# encoding=utf-8
import sys
import time
import os
import __init__

sys.path.append(__init__.config['data_path'])  # add your data path here
from datasets import as_dataset
from tf_trainer import Trainer
from tf_models import AIM
import tensorflow as tf
import traceback
import random
import numpy as np
from itertools import combinations

seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
         0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
data_name = 'avazu'
dataset = as_dataset(data_name)
backend = 'tf'
batch_size = 2000

train_data_param = {
    'gen_type': 'train',
    'random_sample': True,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}
test_data_param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}


def run_one_model(model=None, learning_rate=1e-3, decay_rate=1.0, epsilon=1e-8, ep=5, grda_c=0.005,
                  grda_mu=0.51, learning_rate2=1e-3, decay_rate2=1.0, retrain_stage=0):
    n_ep = ep * 1
    train_param = {
        'opt1': 'adam',
        'opt2': 'grda',
        'loss': 'weight',
        'pos_weight': 1.0,
        'n_epoch': n_ep,
        'train_per_epoch': dataset.train_size / ep,  # split training data
        'test_per_epoch': dataset.test_size,
        'early_stop_epoch': int(0.5 * ep),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'learning_rate2': learning_rate2,
        'decay_rate2': decay_rate2,
        'epsilon': epsilon,
        'load_ckpt': False,
        'ckpt_time': 10000,
        'grda_c': grda_c,
        'grda_mu': grda_mu,
        'test_every_epoch': int(ep / 5),
        'retrain_stage': retrain_stage,
    }
    train_gen = dataset.batch_generator(train_data_param)
    test_gen = dataset.batch_generator(test_data_param)
    trainer = Trainer(model=model, train_gen=train_gen, test_gen=test_gen, **train_param)
    trainer.fit()
    trainer.session.close()


def generate_pairs(ranges=range(1, 100), mask=None, order=2):
    res = []
    for i in range(order):
        res.append([])
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or mask[i] == 1:
            for j in range(order):
                res[j].append(pair[j])
    print("generated pairs", len(res[0]))
    return res


def get_third_order(alpha1=[], num_inputs=24, top_num_inputs=24):
    tmp_left1, tmp_right1 = generate_pairs(range(num_inputs), mask=None)
    num_pair = int(num_inputs * (num_inputs - 1) / 2)
    alpha1 = np.abs(alpha1)
    k = 0
    second_order_operation = [[] for i in range(4)]
    for i in range(4):
        alpha1_dict = {}
        for j in range(len(tmp_left1)):
            alpha1_dict[(tmp_left1[j], tmp_right1[j])] = alpha1[k]
            k += 1
        sort_alpha_dict = sorted(alpha1_dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(top_num_inputs):
            if sort_alpha_dict[j][1] != 0:
                second_order_operation[i].append(sort_alpha_dict[j][0])
                # print(len(second_order_operation[i]))
    third_order_operation = []
    for i in range(4):
        tmp_third_order_operation = []
        for j in range(len(second_order_operation[i])):
            second1, second2 = second_order_operation[i][j]
            for k in range(num_inputs):
                if k != second1 and k != second2:
                    tmp_third = [second1, second2, k]
                    tmp_third.sort()
                    tmp_third_order_operation.append(tuple(tmp_third))
        tmp_third_order_operation = list(set(tmp_third_order_operation))
        tmp_third_order_operation.sort()
        third_order_operation.append(tmp_third_order_operation)
        # print(len(third_order_operation[i]))

    return third_order_operation


def get_topK(choose_set=[], input_alpha=[], top_num_inputs=24):
    k = 0
    topK_inter = [[] for i in range(4)]
    for i in range(4):
        alpha1_dict = {}
        for j in range(len(choose_set[i])):
            if input_alpha[k] != 0:
                alpha1_dict[choose_set[i][j]] = input_alpha[k]
            k += 1
        sort_alpha_dict = sorted(alpha1_dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(min(top_num_inputs, len(sort_alpha_dict))):
            topK_inter[i].append(sort_alpha_dict[j][0])
        print(len(topK_inter[i]))
    return topK_inter


def get_high_order(input_set=[], num_inputs=24, input_order=3):
    high_order_operation = []
    for i in range(4):
        tmp_high_order_operation_list = []
        for j in range(len(input_set[i])):
            tmp_inter = input_set[i][j]
            for k in range(num_inputs):
                flag = False
                for l in range(input_order):
                    if k == tmp_inter[l]:
                        flag = True
                        break
                if flag:
                    continue
                tmp_high_order_interaction = [tmp_inter[i] for i in range(input_order)]
                tmp_high_order_interaction.append(k)
                tmp_high_order_interaction.sort()
                tmp_high_order_operation_list.append(tuple(tmp_high_order_interaction))
        tmp_high_order_operation_list = list(set(tmp_high_order_operation_list))
        tmp_high_order_operation_list.sort()
        high_order_operation.append(tmp_high_order_operation_list)
        print(len(high_order_operation[i]))
    return high_order_operation


import math

if __name__ == "__main__":
    # general parameter
    embedding_size = 40
    l2_v = 0.0
    learning_rate = 1e-3
    dc = 0.75
    split_epoch = 5
    weight_base = 0.6  # the initial value of alpha

    # second-order parameter
    # learn important interactions
    prune = True
    comb_mask = None  # which interactions are reserved
    # learn important embedding dimensions
    beta_prune = False
    beta_comb_mask = None  # which dimensions are reserved

    # third-order parameter
    third_set = [[], [], [], []]  # contain third interactions with four different IFCs
    # for example, third_set = [[(0,1,2),(1,2,3)],[(1,2,5)],[(2,3,4),(2,6,7)],[(1,7,8)]]
    third_prune = False  # whether consider third-order feature interaction
    third_beta_prune = False  # whether consider third-order embedding dimensions
    third_beta_comb_mask = None

    # fourth-order parameter
    fourth_set = [[], [], [], []]  # contain fourth interactions with four different IFCs
    # for example, fourth_set = [[(0,1,2,3),(1,2,3,7)],[(1,2,5,6)],[(2,3,4,9),(2,6,7,10)],[(1,7,8,11)]]
    fourth_prune = False  # whether consider fourth-order feature interaction
    fourth_beta_prune = False  # whether consider fourth-order embedding dimensions
    fourth_beta_comb_mask = None

    # search_stage or retrain_stage; 0 represents search stage and 1 represents retrain stage
    retrain_stage = 0
    # in retrain stage, optimize all parameters by adam Optimizer \
    # and you need to mask interactions and embeddings by comb_mask and beta_comb_mask \
    # high-order interactions could be directly mask in third_set and fourth_set \bn
    # we add an MLP layer in AutoFIFS
    if (prune and beta_prune) or (third_prune and third_beta_prune) or (fourth_prune and fourth_beta_prune):
        print("Error: cannot both optimize alpha and beta due to the coupling of BN")
        exit(-1)

    # grda parameter
    grda_c = 0.0005
    grda_mu = 0.8
    learning_rate2 = 1.0  # learning rate for alpha in research stage
    dc2 = 0.75
    # which parameters need selection and are optimized by GRDA optimizer
    selection_parameter = {"second_interaction": 1, "third_interaction": 0, "fourth_interaction": 0,
                           "second_dimension": 0, "third_dimension": 0, "fourth_dimension": 0}
    model = AIM(init="xavier", num_inputs=dataset.max_length, input_dim=dataset.num_features, l2_v=l2_v,
                embed_size=embedding_size, weight_base=weight_base, retrain_stage=retrain_stage,
                prune=prune, comb_mask=comb_mask, beta_prune=beta_prune, beta_comb_mask=beta_comb_mask,
                third_set=third_set, third_prune=third_prune, third_beta_prune=third_beta_prune,
                third_beta_comb_mask=third_beta_comb_mask, fourth_set=fourth_set, fourth_prune=fourth_prune,
                fourth_beta_prune=fourth_beta_prune, fourth_beta_comb_mask=fourth_beta_comb_mask,
                selection_parameter=selection_parameter)

    run_one_model(model=model, learning_rate=learning_rate, epsilon=1e-8,
                  decay_rate=dc, ep=split_epoch, grda_c=grda_c, grda_mu=grda_mu,
                  learning_rate2=learning_rate2, decay_rate2=dc2, retrain_stage=retrain_stage)

















