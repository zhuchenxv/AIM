from __future__ import print_function

from abc import abstractmethod
from itertools import combinations
import numpy as np
import tensorflow as tf

import __init__
from tf_utils import row_col_fetch, row_col_expand, batch_kernel_product, \
    batch_mlp, create_placeholder, drop_out, embedding_lookup, linear, output, bin_mlp, get_variable, \
    layer_normalization, batch_normalization, get_l2_loss, split_data_mask, embedding_lookup_interaction, \
    get_gather_transpose_embedding

dtype = __init__.config['dtype']

if dtype.lower() == 'float32' or dtype.lower() == 'float':
    dtype = tf.float32
elif dtype.lower() == 'float64':
    dtype = tf.float64

class Model:
    inputs = None
    outputs = None
    logits = None
    labels = None
    learning_rate = None
    loss = None
    l2_loss = None
    optimizer = None
    grad = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__

def generate_pairs(ranges=range(1, 100), mask=None, order=2):
    res = []
    for i in range(order):
        res.append([])
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or mask[i]==1:
            for j in range(order):
                res[j].append(pair[j])
    print("generated pairs", len(res[0]))
    return res


class AutoFM(Model):
    def __init__(self, init='xavier', num_inputs=None, input_dim=None, embed_size=None, l2_w=None, l2_v=None,
                 norm=False, real_inputs=None, comb_mask=None, weight_base=0.6, third_prune=False, 
                 comb_mask_third=None, weight_base_third=0.6, retrain_stage=0):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_ps = l2_v
        self.third_prune = third_prune
        self.retrain_stage = retrain_stage

        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)

        inputs, mask, flag, num_inputs = split_data_mask(self.inputs, num_inputs, norm=norm, real_inputs=real_inputs)

        self.xw, self.xv, b, self.xps = embedding_lookup(init=init, input_dim=input_dim, factor=embed_size, inputs=inputs,
                                               apply_mask=flag, mask=mask, third_order=third_prune)

        l = linear(self.xw)
        self.cols, self.rows = generate_pairs(range(self.xv.shape[1]),mask=comb_mask)
        t_embedding_matrix = tf.transpose(self.xv, perm=[1, 0, 2])
        left = tf.transpose(tf.gather(t_embedding_matrix, self.rows), perm=[1, 0, 2])
        right = tf.transpose(tf.gather(t_embedding_matrix, self.cols), perm=[1, 0, 2])
        level_2_matrix = tf.reduce_sum(tf.multiply(left, right), axis=-1)
        with tf.variable_scope("edge_weight", reuse=tf.AUTO_REUSE):
            self.edge_weights = tf.get_variable('weights', shape=[len(self.cols)],
                                                initializer=tf.random_uniform_initializer(
                                                minval=weight_base - 0.001,
                                                maxval=weight_base + 0.001))
            normed_wts = tf.identity(self.edge_weights, name="normed_wts")
            tf.add_to_collection("structure", self.edge_weights)
            tf.add_to_collection("edge_weights", self.edge_weights)
            mask = tf.identity(normed_wts, name="unpruned_mask")
            mask = tf.expand_dims(mask, axis=0)
        level_2_matrix = tf.layers.batch_normalization(level_2_matrix, axis=-1, training=self.training,
                                                    reuse=tf.AUTO_REUSE, scale=False, center=False, name='prune_BN')
        level_2_matrix *= mask                                          
        if third_prune:
            self.first, self.second, self.third = generate_pairs(range(self.xps.shape[1]), mask=comb_mask_third, order=3)
            t_embedding_matrix = tf.transpose(self.xps, perm=[1, 0, 2])
            first_embed = tf.transpose(tf.gather(t_embedding_matrix, self.first), perm=[1, 0, 2])
            second_embed = tf.transpose(tf.gather(t_embedding_matrix, self.second), perm=[1, 0, 2])
            third_embed = tf.transpose(tf.gather(t_embedding_matrix, self.third), perm=[1, 0, 2])
            level_3_matrix = tf.reduce_sum(tf.multiply(tf.multiply(first_embed, second_embed), third_embed), axis=-1)
            with tf.variable_scope("third_edge_weight", reuse=tf.AUTO_REUSE):
                self.third_edge_weights = tf.get_variable('third_weights', shape=[len(self.first)],
                                                          initializer=tf.random_uniform_initializer(
                                                              minval=weight_base_third - 0.001,
                                                              maxval=weight_base_third + 0.001))
                third_normed_wts = tf.identity(self.third_edge_weights, name="third_normed_wts")
                tf.add_to_collection("third_structure", self.third_edge_weights)
                tf.add_to_collection("third_edge_weights", self.third_edge_weights)
                third_mask = tf.identity(third_normed_wts, name="third_unpruned_mask")
                third_mask = tf.expand_dims(third_mask, axis=0)
            level_3_matrix = tf.layers.batch_normalization(level_3_matrix, axis=-1, training=self.training,
                                                           reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                           name="level_3_matrix_BN")
            level_3_matrix *= third_mask

        fm_out = tf.reduce_sum(level_2_matrix, axis=-1)
        if third_prune:
            fm_out2 = tf.reduce_sum(level_3_matrix, axis=-1)
        if third_prune:
            self.logits, self.outputs = output([l, fm_out,fm_out2, b, ])
        else:
            self.logits, self.outputs = output([l, fm_out, b, ])

    def analyse_structure(self, sess, print_full_weight=False, epoch=None):
        import numpy as np
        wts, mask = sess.run(["edge_weight/normed_wts:0", "edge_weight/unpruned_mask:0"])
        if print_full_weight:
            outline = ""
            for j in range(wts.shape[0]):
                outline += str(wts[j]) + ","
            outline += "\n"
            print("log avg auc all weights for(epoch:%s)" % (epoch), outline)
        print("wts", wts[:10])
        print("mask", mask[:10])
        zeros_ = np.zeros_like(mask, dtype=np.float32)
        zeros_[mask == 0] = 1
        print("masked edge_num", sum(zeros_))
        if self.third_prune:
            wts, mask = sess.run(["third_edge_weight/third_normed_wts:0", "third_edge_weight/third_unpruned_mask:0"])
            if print_full_weight:
                outline = ""
                for j in range(wts.shape[0]):
                    outline += str(wts[j]) + ","
                outline += "\n"
                print("third log avg auc all third weights for(epoch:%s)" % (epoch), outline)
            print("third wts", wts[:10])
            print("third mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("third masked edge_num", sum(zeros_))

    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                if self.third_prune:
                    self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.l2_ps],
                                               [self.xw, self.xv, self.xps])
                else:
                    self.l2_loss = get_l2_loss([self.l2_w, self.l2_v],
                                               [self.xw, self.xv])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                if self.retrain_stage:
                    all_variable = [v for v in tf.trainable_variables()]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable)
                else:
                    if self.third_prune:
                        weight_second_var = list(set(tf.get_collection("edge_weights")))
                        weight_third_var = list(set(tf.get_collection("third_edge_weights")))
                        weight_var = weight_second_var + weight_third_var
                        weight_var = list(set(weight_var))
                    else:
                        weight_var = list(set(tf.get_collection("edge_weights")))
                    all_variable = [v for v in tf.trainable_variables()]
                    other_var = [i for i in all_variable if i not in weight_var]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=other_var)
                    self.optimizer2 = optimizer2.minimize(loss=_loss_, var_list=weight_var)


class AutoDeepFM(Model):
    def __init__(self, init='xavier', num_inputs=None, input_dim=None, embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, norm=False, real_inputs=None,
                 batch_norm=False, layer_norm=False, comb_mask=None, weight_base=0.6, third_prune=False, 
                 comb_mask_third=None, weight_base_third=0.6, retrain_stage=0):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_ps = l2_v
        self.layer_l2 = layer_l2
        self.retrain_stage = retrain_stage
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        layer_keeps = drop_out(self.training, layer_keeps)
        inputs, mask, flag, num_inputs = split_data_mask(self.inputs, num_inputs, norm=norm, real_inputs=real_inputs)

        self.xw, xv, _, self.xps = embedding_lookup(init=init, input_dim=input_dim, factor=embed_size, inputs=inputs,
                                            apply_mask=flag, mask=mask, use_b=False, third_order=third_prune)
        self.third_prune = third_prune
        self.xv = xv
        h = tf.reshape(xv, [-1, num_inputs * embed_size])
        h, self.layer_kernels, _ = bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, num_inputs * embed_size,
                                           batch_norm=batch_norm, layer_norm=layer_norm, training=self.training)
        h = tf.squeeze(h)

        l = linear(self.xw)
        self.cols, self.rows = generate_pairs(range(self.xv.shape[1]),mask=comb_mask)
        t_embedding_matrix = tf.transpose(self.xv, perm=[1, 0, 2])
        left = tf.transpose(tf.gather(t_embedding_matrix, self.rows), perm=[1, 0, 2])
        right = tf.transpose(tf.gather(t_embedding_matrix, self.cols), perm=[1, 0, 2])
        level_2_matrix = tf.reduce_sum(tf.multiply(left, right), axis=-1)
        with tf.variable_scope("edge_weight", reuse=tf.AUTO_REUSE):
            self.edge_weights = tf.get_variable('weights', shape=[len(self.cols)],
                                                initializer=tf.random_uniform_initializer(
                                                minval=weight_base - 0.001,
                                                maxval=weight_base + 0.001))
            normed_wts = tf.identity(self.edge_weights, name="normed_wts")
            tf.add_to_collection("structure", self.edge_weights)
            tf.add_to_collection("edge_weights", self.edge_weights)
            mask = tf.identity(normed_wts, name="unpruned_mask")
            mask = tf.expand_dims(mask, axis=0)
        level_2_matrix = tf.layers.batch_normalization(level_2_matrix, axis=-1, training=self.training,
                                                    reuse=tf.AUTO_REUSE, scale=False, center=False, name='prune_BN')
        level_2_matrix *= mask                                          
        if third_prune:
            self.first, self.second, self.third = generate_pairs(range(self.xps.shape[1]), mask=comb_mask_third, order=3)
            t_embedding_matrix = tf.transpose(self.xps, perm=[1, 0, 2])
            first_embed = tf.transpose(tf.gather(t_embedding_matrix, self.first), perm=[1, 0, 2])
            second_embed = tf.transpose(tf.gather(t_embedding_matrix, self.second), perm=[1, 0, 2])
            third_embed = tf.transpose(tf.gather(t_embedding_matrix, self.third), perm=[1, 0, 2])
            level_3_matrix = tf.reduce_sum(tf.multiply(tf.multiply(first_embed, second_embed), third_embed), axis=-1)
            with tf.variable_scope("third_edge_weight", reuse=tf.AUTO_REUSE):
                self.third_edge_weights = tf.get_variable('third_weights', shape=[len(self.first)],
                                                          initializer=tf.random_uniform_initializer(
                                                              minval=weight_base_third - 0.001,
                                                              maxval=weight_base_third + 0.001))
                third_normed_wts = tf.identity(self.third_edge_weights, name="third_normed_wts")
                tf.add_to_collection("third_structure", self.third_edge_weights)
                tf.add_to_collection("third_edge_weights", self.third_edge_weights)
                third_mask = tf.identity(third_normed_wts, name="third_unpruned_mask")
                third_mask = tf.expand_dims(third_mask, axis=0)
            level_3_matrix = tf.layers.batch_normalization(level_3_matrix, axis=-1, training=self.training,
                                                           reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                           name="level_3_matrix_BN")
            level_3_matrix *= third_mask

        fm_out = tf.reduce_sum(level_2_matrix, axis=-1)
        if third_prune:
            fm_out2 = tf.reduce_sum(level_3_matrix, axis=-1)
        if third_prune:
            self.logits, self.outputs = output([l, fm_out,fm_out2, h, ])
        else:
            self.logits, self.outputs = output([l, fm_out, h, ])

    def analyse_structure(self, sess, print_full_weight=False, epoch=None):
        import numpy as np
        wts, mask = sess.run(["edge_weight/normed_wts:0", "edge_weight/unpruned_mask:0"])
        if print_full_weight:
            outline = ""
            for j in range(wts.shape[0]):
                outline += str(wts[j]) + ","
            outline += "\n"
            print("log avg auc all weights for(epoch:%s)" % (epoch), outline)
        print("wts", wts[:10])
        print("mask", mask[:10])
        zeros_ = np.zeros_like(mask, dtype=np.float32)
        zeros_[mask == 0] = 1
        print("masked edge_num", sum(zeros_))
        if self.third_prune:
            wts, mask = sess.run(["third_edge_weight/third_normed_wts:0", "third_edge_weight/third_unpruned_mask:0"])
            if print_full_weight:
                outline = ""
                for j in range(wts.shape[0]):
                    outline += str(wts[j]) + ","
                outline += "\n"
                print("third log avg auc all third weights for(epoch:%s)" % (epoch), outline)
            print("third wts", wts[:10])
            print("third mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("third masked edge_num", sum(zeros_))

    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                if self.third_prune:
                    self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.l2_ps, self.layer_l2],
                                               [self.xw, self.xv, self.xps, self.layer_kernels])
                else:
                    self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.layer_l2],
                                               [self.xw, self.xv, self.layer_kernels])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                if self.retrain_stage:
                    all_variable = [v for v in tf.trainable_variables()]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable)
                else:
                    all_variable = [v for v in tf.trainable_variables()]
                    if self.third_prune:
                        print("optimizer")
                        weight_second_var = list(set(tf.get_collection("edge_weights")))
                        weight_third_var = list(set(tf.get_collection("third_edge_weights")))
                        weight_var = weight_second_var + weight_third_var
                        weight_var = list(set(weight_var))
                        # weight_var = list(set(tf.get_collection("third_edge_weights")))
                    else:
                        weight_var = list(set(tf.get_collection("edge_weights")))
                    other_var = [i for i in all_variable if i not in weight_var]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=other_var)
                    self.optimizer2 = optimizer2.minimize(loss=_loss_, var_list=weight_var)


class AIM(Model):
    def __init__(self, init='xavier', num_inputs=None, input_dim=None, embed_size=None, l2_w=None, l2_v=None,
                 norm=False, real_inputs=None, weight_base=0.6, retrain_stage=0, comb_mask=None, prune=True,
                 beta_comb_mask=None, beta_prune=False, third_set=None, third_prune=False, third_beta_comb_mask=None,
                 third_beta_prune=False, fourth_set=None, fourth_prune=False, fourth_beta_comb_mask=None,
                 fourth_beta_prune=False, selection_parameter=None):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_ps = l2_v
        self.retrain_stage = retrain_stage
        self.prune = prune
        self.third_prune = third_prune
        self.fourth_prune = fourth_prune
        self.beta_prune = beta_prune
        self.third_beta_prune = third_beta_prune
        self.fourth_beta_prune = fourth_beta_prune
        self.selection_parameter = selection_parameter
        third_order = False
        if third_set is not None:
            if len(third_set[0]) + len(third_set[1]) + len(third_set[2]) + len(third_set[3]) != 0:
                third_order = True
        fourth_order = False
        if fourth_set is not None:
            if len(fourth_set[0])+len(fourth_set[1])+len(fourth_set[2])+len(fourth_set[3]) != 0:
                fourth_order = True
        if fourth_order is False:
            self.fourth_prune = False
            self.fourth_beta_prune = False

        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        inputs, mask, flag, num_inputs = split_data_mask(self.inputs, num_inputs, norm=norm, real_inputs=real_inputs)
        inter_func_num = 4
        inter_num = int(num_inputs * (num_inputs - 1) / 2)
        if comb_mask is None:
            all_embed_size = inter_num * inter_func_num * embed_size
            pair_num = inter_num * inter_func_num
        else:
            all_embed_size = sum(comb_mask) * embed_size
            pair_num = sum(comb_mask)
        self.xw, self.xv_list, b, self.third_xv_list, self.fourth_xv_list, _ = embedding_lookup_interaction(init=init, input_dim=input_dim,
                                        factor=embed_size, inputs=inputs, apply_mask=flag, mask=mask, interaction_function_num=inter_func_num,
                                        third_order=third_order, fourth_order=fourth_order, fifth_order=False)

        l = linear(self.xw)
        # get embedding
        pnn_input_vector_list = []
        if comb_mask is None:
            tmp_left1, tmp_right1 = generate_pairs(range(num_inputs), mask=comb_mask)
            self.left_pos = [tmp_left1 for i in range(inter_func_num)]
            self.right_pos = [tmp_right1 for i in range(inter_func_num)]
        else:
            self.left_pos = []
            self.right_pos = []
            for i in range(inter_func_num):
                tmp_comb_mask = comb_mask[inter_num * i: inter_num * (i + 1)]
                tmp_left1, tmp_right1 = generate_pairs(range(num_inputs), mask=tmp_comb_mask)
                self.left_pos.append(tmp_left1)
                self.right_pos.append(tmp_right1)
        left = []
        right = []
        if beta_comb_mask:
            print("second beta num:", sum(beta_comb_mask))
            beta_comb_mask_tmp = np.reshape(np.array(beta_comb_mask), (inter_func_num, num_inputs, embed_size))
            for i in range(inter_func_num):
                self.xv_list[i] *= beta_comb_mask_tmp[i]
        if beta_prune:
            left1 = []
            right1 = []
            for i in range(inter_func_num):
                tmp_embedding_matrix = tf.transpose(self.xv_list[i], perm=[1, 0, 2])
                tmp_left2 = get_gather_transpose_embedding(tmp_embedding_matrix, self.left_pos[i])
                tmp_right2 = get_gather_transpose_embedding(tmp_embedding_matrix, self.right_pos[i])
                if tmp_left2 is not None:
                    left1.append(tmp_left2)
                    right1.append(tmp_right2)
            all_left = tf.reshape(tf.concat(left1, axis=1), shape=(-1, all_embed_size))  # batch, pair*k
            all_right = tf.reshape(tf.concat(right1, axis=1), shape=(-1, all_embed_size))  # batch, pair*k
            all_left = tf.layers.batch_normalization(all_left, axis=-1, training=self.training,
                                                     reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                     name='left_beta_BN')
            all_right = tf.layers.batch_normalization(all_right, axis=-1, training=self.training,
                                                      reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                      name='right_beta_BN')
            all_left = tf.reshape(all_left, shape=(-1, pair_num, embed_size))
            all_right = tf.reshape(all_right, shape=(-1, pair_num, embed_size))
            tmp_num = 0
            for i in range(inter_func_num):
                if len(self.left_pos[i]) > 0:
                    left.append(all_left[:, tmp_num: tmp_num + len(self.left_pos[i])])
                    right.append(all_right[:, tmp_num: tmp_num + len(self.right_pos[i])])
                    tmp_num += len(self.left_pos[i])
                else:
                    left.append(None)
                    right.append(None)
            with tf.variable_scope("beta_weight", reuse=tf.AUTO_REUSE):
                self.beta_weight = tf.get_variable('beta_weights', shape=[inter_func_num, num_inputs, embed_size],
                                                initializer=tf.random_uniform_initializer(
                                                minval=weight_base - 0.001,
                                                maxval=weight_base + 0.001))
                tf.add_to_collection("beta_edge_weights", self.beta_weight)
                for i in range(inter_func_num):
                    if len(self.left_pos[i]) > 0:
                        left_mask = tf.gather(self.beta_weight[i], self.left_pos[i])  # pair, k
                        left[i] *= left_mask  # batch, pair, k
                        right_mask = tf.gather(self.beta_weight[i], self.right_pos[i])  # pair, k
                        right[i] *= right_mask  # batch, pair, k
        else:
            for i in range(inter_func_num):
                tmp_embedding_matrix = tf.transpose(self.xv_list[i], perm=[1, 0, 2])  # field size, batch, k
                left.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.left_pos[i]))
                right.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.right_pos[i]))

        # compute inner product, kernel product, outer product
        all_level_2_matrix_list = []
        for f_index in range(inter_func_num):
            if len(self.left_pos[f_index]) == 0:
                continue
            if f_index == 0:
                level_2_inner_product = tf.multiply(left[f_index], right[f_index])  # batch, interaction, embed
                pnn_input_vector_list.append(
                    tf.reshape(level_2_inner_product, (-1, len(self.left_pos[f_index]) * embed_size)))
                level_2_inner_product = tf.reduce_sum(level_2_inner_product, axis=-1)
                all_level_2_matrix_list.append(level_2_inner_product)
            elif f_index == 1:
                kernel_vec = get_variable('tnormal', name='kernel_vec',
                                          shape=[1, len(self.left_pos[f_index]), embed_size])
                level_2_vec_kernel = left[f_index] * right[f_index] * kernel_vec
                pnn_input_vector_list.append(
                    tf.reshape(level_2_vec_kernel, (-1, len(self.left_pos[f_index]) * embed_size)))
                level_2_vec_kernel = tf.reduce_sum(level_2_vec_kernel, axis=-1)
                all_level_2_matrix_list.append(level_2_vec_kernel)
            elif f_index == 2:
                kernel_num = get_variable('tnormal', name='kernel_num', shape=[1, len(self.right_pos[f_index]), 1])
                level_2_num_kernel = left[f_index] * right[f_index] * kernel_num
                pnn_input_vector_list.append(
                    tf.reshape(level_2_num_kernel, (-1, len(self.left_pos[f_index]) * embed_size)))
                level_2_num_kernel = tf.reduce_sum(level_2_num_kernel, axis=-1)
                all_level_2_matrix_list.append(level_2_num_kernel)
            elif f_index == 3:
                outer_w1 = get_variable(init, name='outer_w1', shape=[embed_size, 1])
                outer_w2 = get_variable(init, name='outer_w2', shape=[embed_size, 1])
                level_2_outer1 = tf.squeeze(tf.tensordot(tf.expand_dims(left[f_index], 2), outer_w1, axes=1), [2, 3])
                level_2_outer2 = tf.squeeze(tf.tensordot(tf.expand_dims(right[f_index], 2), outer_w2, axes=1), [2, 3])
                level_2_outer = tf.multiply(level_2_outer1, level_2_outer2)
                all_level_2_matrix_list.append(level_2_outer)
                pnn_input_vector_list.append(level_2_outer)
        all_level_2_matrix = tf.concat(all_level_2_matrix_list, axis=-1)

        if prune:
            all_level_2_matrix = tf.layers.batch_normalization(all_level_2_matrix, axis=-1, training=self.training,
                                                               reuse=tf.AUTO_REUSE, scale=False,
                                                               center=False, name='prune_BN')
            with tf.variable_scope("edge_weight", reuse=tf.AUTO_REUSE):
                print("all interaction num:", pair_num)
                self.edge_weights = tf.get_variable('weights', shape=[pair_num],
                                                    initializer=tf.random_uniform_initializer(
                                                    minval=weight_base - 0.001,
                                                    maxval=weight_base + 0.001))
                normed_wts = tf.identity(self.edge_weights, name="normed_wts")
                tf.add_to_collection("structure", self.edge_weights)
                tf.add_to_collection("edge_weights", self.edge_weights)
                mask = tf.identity(normed_wts, name="unpruned_mask")
                mask = tf.expand_dims(mask, axis=0)
            all_level_2_matrix *= mask
        fm_out = tf.reduce_sum(all_level_2_matrix, axis=-1)

        print("third order")
        if third_order:
            third_pair_num = 0
            for i in range(inter_func_num):
                third_pair_num += len(third_set[i])
            print("all third interaction num:", third_pair_num)
            third_all_embed_size = third_pair_num * embed_size
            third1 = []
            third2 = []
            third3 = []
            self.third1_pos = [[] for i in range(inter_func_num)]
            self.third2_pos = [[] for i in range(inter_func_num)]
            self.third3_pos = [[] for i in range(inter_func_num)]
            for i in range(inter_func_num):
                for j in range(len(third_set[i])):
                    self.third1_pos[i].append(third_set[i][j][0])
                    self.third2_pos[i].append(third_set[i][j][1])
                    self.third3_pos[i].append(third_set[i][j][2])
            if third_beta_comb_mask:
                third_beta_comb_mask_tmp = np.reshape(np.array(third_beta_comb_mask), (inter_func_num, num_inputs, embed_size))
                for i in range(inter_func_num):
                    self.third_xv_list[i] *= third_beta_comb_mask_tmp[i]
            if third_beta_prune:
                third1_2 = []
                third2_2 = []
                third3_2 = []
                for i in range(inter_func_num):
                    if len(self.third1_pos[i]) == 0:
                        continue
                    tmp_embedding_matrix = tf.transpose(self.third_xv_list[i], perm=[1, 0, 2])
                    third1_2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.third1_pos[i]))
                    third2_2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.third2_pos[i]))
                    third3_2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.third3_pos[i]))
                all_third1_2 = tf.reshape(tf.concat(third1_2, axis=1), shape=(-1, third_all_embed_size))
                all_third2_2 = tf.reshape(tf.concat(third2_2, axis=1), shape=(-1, third_all_embed_size))
                all_third3_2 = tf.reshape(tf.concat(third3_2, axis=1), shape=(-1, third_all_embed_size))
                all_third1_2 = tf.layers.batch_normalization(all_third1_2, axis=-1, training=self.training,
                                                             reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                             name='third1_prune_BN')
                all_third2_2 = tf.layers.batch_normalization(all_third2_2, axis=-1, training=self.training,
                                                             reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                             name='third2_prune_BN')
                all_third3_2 = tf.layers.batch_normalization(all_third3_2, axis=-1, training=self.training,
                                                             reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                             name='third3_prune_BN')
                all_third1_2 = tf.reshape(all_third1_2, shape=(-1, third_pair_num, embed_size))
                all_third2_2 = tf.reshape(all_third2_2, shape=(-1, third_pair_num, embed_size))
                all_third3_2 = tf.reshape(all_third3_2, shape=(-1, third_pair_num, embed_size))
                tmp_num = 0
                for i in range(inter_func_num):
                    if len(self.third1_pos[i]) > 0:
                        third1.append(all_third1_2[:, tmp_num: tmp_num + len(self.third1_pos[i])])
                        third2.append(all_third2_2[:, tmp_num: tmp_num + len(self.third2_pos[i])])
                        third3.append(all_third3_2[:, tmp_num: tmp_num + len(self.third3_pos[i])])
                        tmp_num += len(self.third1_pos[i])
                    else:
                        third1.append(None)
                        third2.append(None)
                        third3.append(None)
                with tf.variable_scope("third_beta_weight", reuse=tf.AUTO_REUSE):
                    self.third_beta_weight = tf.get_variable('third_beta_weights',
                                                             shape=[inter_func_num, num_inputs, embed_size],
                                                             initializer=tf.random_uniform_initializer(
                                                                 minval=weight_base - 0.001,
                                                                 maxval=weight_base + 0.001))
                    tf.add_to_collection("third_beta_edge_weights", self.third_beta_weight)
                    for i in range(inter_func_num):
                        if len(self.third1_pos[i]) > 0:
                            third1[i] *= tf.gather(self.third_beta_weight[i], self.third1_pos[i])
                            third2[i] *= tf.gather(self.third_beta_weight[i], self.third2_pos[i])
                            third3[i] *= tf.gather(self.third_beta_weight[i], self.third3_pos[i])
            else:
                for i in range(inter_func_num):
                    tmp_embedding_matrix = tf.transpose(self.third_xv_list[i], perm=[1, 0, 2])  # field size, batch, k
                    third1.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.third1_pos[i]))
                    third2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.third2_pos[i]))
                    third3.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.third3_pos[i]))

            all_level_3_matrix_list = []
            for f_index in range(inter_func_num):
                if len(self.third1_pos[f_index]) == 0:
                    continue
                if f_index == 0:
                    level_3_inner_product = tf.multiply(tf.multiply(third1[f_index], third2[f_index]), third3[f_index])  # batch, interaction, embed
                    pnn_input_vector_list.append(
                        tf.reshape(level_3_inner_product, (-1, len(self.third1_pos[f_index]) * embed_size)))
                    level_3_inner_product = tf.reduce_sum(level_3_inner_product, axis=-1)
                    all_level_3_matrix_list.append(level_3_inner_product)
                elif f_index == 1:
                    kernel_3_vec = get_variable('tnormal', name='kernel_3_vec',
                                              shape=[1, len(self.third1_pos[f_index]), embed_size])
                    level_3_vec_kernel = third1[f_index] * third2[f_index] * third3[f_index] * kernel_3_vec
                    pnn_input_vector_list.append(
                        tf.reshape(level_3_vec_kernel, (-1, len(self.third1_pos[f_index]) * embed_size)))
                    level_3_vec_kernel = tf.reduce_sum(level_3_vec_kernel, axis=-1)
                    all_level_3_matrix_list.append(level_3_vec_kernel)
                elif f_index == 2:
                    kernel_3_num = get_variable('tnormal', name='kernel_3_num', shape=[1, len(self.third1_pos[f_index]), 1])
                    level_3_num_kernel = third1[f_index] * third2[f_index] * third3[f_index] * kernel_3_num
                    pnn_input_vector_list.append(
                        tf.reshape(level_3_num_kernel, (-1, len(self.third1_pos[f_index]) * embed_size)))
                    level_3_num_kernel = tf.reduce_sum(level_3_num_kernel, axis=-1)
                    all_level_3_matrix_list.append(level_3_num_kernel)
                elif f_index == 3:
                    outer_3_w1 = get_variable(init, name='outer_3_w1', shape=[embed_size, 1])
                    outer_3_w2 = get_variable(init, name='outer_3_w2', shape=[embed_size, 1])
                    outer_3_w3 = get_variable(init, name='outer_3_w3', shape=[embed_size, 1])
                    level_3_outer1 = tf.squeeze(tf.tensordot(tf.expand_dims(third1[f_index], 2), outer_3_w1, axes=1),
                                                [2, 3])
                    level_3_outer2 = tf.squeeze(tf.tensordot(tf.expand_dims(third2[f_index], 2), outer_3_w2, axes=1),
                                                [2, 3])
                    level_3_outer3 = tf.squeeze(tf.tensordot(tf.expand_dims(third3[f_index], 2), outer_3_w3, axes=1),
                                                [2, 3])
                    level_3_outer = tf.multiply(tf.multiply(level_3_outer1, level_3_outer2), level_3_outer3)
                    all_level_3_matrix_list.append(level_3_outer)
                    pnn_input_vector_list.append(level_3_outer)

            all_level_3_matrix = tf.concat(all_level_3_matrix_list, axis=-1)
            if third_prune:
                all_level_3_matrix = tf.layers.batch_normalization(all_level_3_matrix, axis=-1,
                                                                   training=self.training,
                                                                   reuse=tf.AUTO_REUSE, scale=False,
                                                                   center=False, name='third_prune_BN')
                with tf.variable_scope("third_edge_weight", reuse=tf.AUTO_REUSE):
                    self.third_edge_weights = tf.get_variable('third_weights', shape=[third_pair_num],
                                                        initializer=tf.random_uniform_initializer(
                                                            minval=weight_base - 0.001,
                                                            maxval=weight_base + 0.001))
                    third_normed_wts = tf.identity(self.third_edge_weights, name="third_normed_wts")
                    tf.add_to_collection("third_structure", self.third_edge_weights)
                    tf.add_to_collection("third_edge_weights", self.third_edge_weights)
                    third_mask = tf.identity(third_normed_wts, name="third_unpruned_mask")
                    third_mask = tf.expand_dims(third_mask, axis=0)
                all_level_3_matrix *= third_mask

        print("fourth order")
        if fourth_order:
            fourth_pair_num = 0
            for i in range(inter_func_num):
                fourth_pair_num += len(fourth_set[i])
            print("all fourth interaction num:", fourth_pair_num)
            fourth_all_embed_size = fourth_pair_num * embed_size
            fourth1 = []
            fourth2 = []
            fourth3 = []
            fourth4 = []
            self.fourth1_pos = [[] for i in range(inter_func_num)]
            self.fourth2_pos = [[] for i in range(inter_func_num)]
            self.fourth3_pos = [[] for i in range(inter_func_num)]
            self.fourth4_pos = [[] for i in range(inter_func_num)]
            for i in range(inter_func_num):
                for j in range(len(fourth_set[i])):
                    self.fourth1_pos[i].append(fourth_set[i][j][0])
                    self.fourth2_pos[i].append(fourth_set[i][j][1])
                    self.fourth3_pos[i].append(fourth_set[i][j][2])
                    self.fourth4_pos[i].append(fourth_set[i][j][3])
            if fourth_beta_comb_mask:
                fourth_beta_comb_mask_tmp = np.reshape(np.array(fourth_beta_comb_mask), (inter_func_num, num_inputs, embed_size))
                for i in range(inter_func_num):
                    self.fourth_xv_list[i] *= fourth_beta_comb_mask_tmp[i]
            if fourth_beta_prune:
                fourth1_2 = []
                fourth2_2 = []
                fourth3_2 = []
                fourth4_2 = []
                for i in range(inter_func_num):
                    if len(self.fourth1_pos[i]) == 0:
                        continue
                    tmp_embedding_matrix = tf.transpose(self.fourth_xv_list[i], perm=[1, 0, 2])
                    fourth1_2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth1_pos[i]))
                    fourth2_2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth2_pos[i]))
                    fourth3_2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth3_pos[i]))
                    fourth4_2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth4_pos[i]))
                all_fourth1_2 = tf.reshape(tf.concat(fourth1_2, axis=1), shape=(-1, fourth_all_embed_size))
                all_fourth2_2 = tf.reshape(tf.concat(fourth2_2, axis=1), shape=(-1, fourth_all_embed_size))
                all_fourth3_2 = tf.reshape(tf.concat(fourth3_2, axis=1), shape=(-1, fourth_all_embed_size))
                all_fourth4_2 = tf.reshape(tf.concat(fourth4_2, axis=1), shape=(-1, fourth_all_embed_size))
                all_fourth1_2 = tf.layers.batch_normalization(all_fourth1_2, axis=-1, training=self.training,
                                                         reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                         name='fourth1_beta_BN')
                all_fourth2_2 = tf.layers.batch_normalization(all_fourth2_2, axis=-1, training=self.training,
                                                              reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                              name='fourth2_beta_BN')
                all_fourth3_2 = tf.layers.batch_normalization(all_fourth3_2, axis=-1, training=self.training,
                                                              reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                              name='fourth3_beta_BN')
                all_fourth4_2 = tf.layers.batch_normalization(all_fourth4_2, axis=-1, training=self.training,
                                                              reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                              name='fourth4_beta_BN')
                all_fourth1_2 = tf.reshape(all_fourth1_2, shape=(-1, fourth_pair_num, embed_size))
                all_fourth2_2 = tf.reshape(all_fourth2_2, shape=(-1, fourth_pair_num, embed_size))
                all_fourth3_2 = tf.reshape(all_fourth3_2, shape=(-1, fourth_pair_num, embed_size))
                all_fourth4_2 = tf.reshape(all_fourth4_2, shape=(-1, fourth_pair_num, embed_size))
                tmp_num = 0
                for i in range(inter_func_num):
                    if len(self.fourth1_pos[i]) > 0:
                        fourth1.append(all_fourth1_2[:, tmp_num: tmp_num + len(self.fourth1_pos[i])])
                        fourth2.append(all_fourth2_2[:, tmp_num: tmp_num + len(self.fourth2_pos[i])])
                        fourth3.append(all_fourth3_2[:, tmp_num: tmp_num + len(self.fourth3_pos[i])])
                        fourth4.append(all_fourth4_2[:, tmp_num: tmp_num + len(self.fourth4_pos[i])])
                        tmp_num += len(self.fourth1_pos[i])
                    else:
                        fourth1.append(None)
                        fourth2.append(None)
                        fourth3.append(None)
                        fourth4.append(None)
                with tf.variable_scope("fourth_beta_weight", reuse=tf.AUTO_REUSE):
                    self.fourth_beta_weight = tf.get_variable('fourth_beta_weights', shape=[inter_func_num, num_inputs, embed_size],
                                                    initializer=tf.random_uniform_initializer(
                                                    minval=weight_base - 0.001,
                                                    maxval=weight_base + 0.001))
                    tf.add_to_collection("fourth_beta_edge_weights", self.fourth_beta_weight)
                    for i in range(inter_func_num):
                        if len(self.fourth1_pos[i]) > 0:
                            fourth1[i] *= tf.gather(self.fourth_beta_weight[i], self.fourth1_pos[i])
                            fourth2[i] *= tf.gather(self.fourth_beta_weight[i], self.fourth2_pos[i])
                            fourth3[i] *= tf.gather(self.fourth_beta_weight[i], self.fourth3_pos[i])
                            fourth4[i] *= tf.gather(self.fourth_beta_weight[i], self.fourth4_pos[i])
            else:
                for i in range(inter_func_num):
                    tmp_embedding_matrix = tf.transpose(self.fourth_xv_list[i], perm=[1, 0, 2])  # field size, batch, k
                    fourth1.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth1_pos[i]))
                    fourth2.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth2_pos[i]))
                    fourth3.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth3_pos[i]))
                    fourth4.append(get_gather_transpose_embedding(tmp_embedding_matrix, self.fourth4_pos[i]))

            all_level_4_matrix_list = []
            for f_index in range(inter_func_num):
                print("shape:", len(self.fourth1_pos[f_index]))
                if len(self.fourth1_pos[f_index]) == 0:
                    continue
                if f_index == 0:
                    level_4_inner_product = tf.multiply(tf.multiply(tf.multiply(fourth1[f_index], fourth2[f_index]), fourth3[f_index]), fourth4[f_index])  # batch, interaction, embed
                    pnn_input_vector_list.append(
                        tf.reshape(level_4_inner_product, (-1, len(self.fourth1_pos[f_index]) * embed_size)))
                    level_4_inner_product = tf.reduce_sum(level_4_inner_product, axis=-1)
                    all_level_4_matrix_list.append(level_4_inner_product)
                elif f_index == 1:
                    kernel_4_vec = get_variable('tnormal', name='kernel_4_vec',
                                              shape=[1, len(self.fourth1_pos[f_index]), embed_size])
                    level_4_vec_kernel = fourth1[f_index] * fourth2[f_index] * fourth3[f_index] * fourth4[f_index] * kernel_4_vec
                    pnn_input_vector_list.append(
                        tf.reshape(level_4_vec_kernel, (-1, len(self.fourth1_pos[f_index]) * embed_size)))
                    level_4_vec_kernel = tf.reduce_sum(level_4_vec_kernel, axis=-1)
                    all_level_4_matrix_list.append(level_4_vec_kernel)
                elif f_index == 2:
                    kernel_4_num = get_variable('tnormal', name='kernel_4_num', shape=[1, len(self.fourth1_pos[f_index]), 1])
                    level_4_num_kernel = fourth1[f_index] * fourth2[f_index] * fourth3[f_index] * fourth4[f_index] * kernel_4_num
                    pnn_input_vector_list.append(
                        tf.reshape(level_4_num_kernel, (-1, len(self.fourth1_pos[f_index]) * embed_size)))
                    level_4_num_kernel = tf.reduce_sum(level_4_num_kernel, axis=-1)
                    all_level_4_matrix_list.append(level_4_num_kernel)
                elif f_index == 3:
                    outer_4_w1 = get_variable(init, name='outer_4_w1', shape=[embed_size, 1])
                    outer_4_w2 = get_variable(init, name='outer_4_w2', shape=[embed_size, 1])
                    outer_4_w3 = get_variable(init, name='outer_4_w3', shape=[embed_size, 1])
                    outer_4_w4 = get_variable(init, name='outer_4_w4', shape=[embed_size, 1])
                    level_4_outer1 = tf.squeeze(tf.tensordot(tf.expand_dims(fourth1[f_index], 2), outer_4_w1, axes=1),
                                                [2, 3])
                    level_4_outer2 = tf.squeeze(tf.tensordot(tf.expand_dims(fourth2[f_index], 2), outer_4_w2, axes=1),
                                                [2, 3])
                    level_4_outer3 = tf.squeeze(tf.tensordot(tf.expand_dims(fourth3[f_index], 2), outer_4_w3, axes=1),
                                                [2, 3])
                    level_4_outer4 = tf.squeeze(tf.tensordot(tf.expand_dims(fourth4[f_index], 2), outer_4_w4, axes=1),
                                                [2, 3])
                    level_4_outer = tf.multiply(tf.multiply(tf.multiply(level_4_outer1, level_4_outer2), level_4_outer3), level_4_outer4)
                    all_level_4_matrix_list.append(level_4_outer)
                    pnn_input_vector_list.append(level_4_outer)
            all_level_4_matrix = tf.concat(all_level_4_matrix_list, axis=-1)
            if fourth_prune:
                all_level_4_matrix = tf.layers.batch_normalization(all_level_4_matrix, axis=-1, training=self.training,
                                                                   reuse=tf.AUTO_REUSE, scale=False, epsilon=1e-10,
                                                                   center=False, name='fourth_prune_BN')
                with tf.variable_scope("fourth_edge_weight", reuse=tf.AUTO_REUSE):
                    self.fourth_edge_weights = tf.get_variable('fourth_weights', shape=[fourth_pair_num],
                                                        initializer=tf.random_uniform_initializer(
                                                            minval=weight_base - 0.001,
                                                            maxval=weight_base + 0.001))
                    fourth_normed_wts = tf.identity(self.fourth_edge_weights, name="fourth_normed_wts")
                    tf.add_to_collection("fourth_structure", self.fourth_edge_weights)
                    tf.add_to_collection("fourth_edge_weights", self.fourth_edge_weights)
                    fourth_mask = tf.identity(fourth_normed_wts, name="fourth_unpruned_mask")
                    fourth_mask = tf.expand_dims(fourth_mask, axis=0)
                all_level_4_matrix *= fourth_mask

        if retrain_stage:
            print("add MLP")
            depth = 5
            width = 700
            ls = [width] * depth
            ls.append(1)
            la = ['relu'] * depth
            la.append(None)
            lk = [1.] * depth
            lk.append(1.)
            pnn_input_vector_final = tf.concat(pnn_input_vector_list, axis=-1)
            h, self.layer_kernels, _ = bin_mlp(init, ls, la, lk, pnn_input_vector_final,
                                               pnn_input_vector_final.get_shape().as_list()[1],
                                               layer_norm=False, batch_norm=True, training=self.training)
            h = tf.squeeze(h)
        output_w1 = get_variable(init, name='output_w1', shape=[1])
        output_w2 = get_variable(init, name='output_w2', shape=[1])
        output_w3 = get_variable(init, name='output_w3', shape=[1])
        output_w4 = get_variable(init, name='output_w4', shape=[1])
        output_w5 = get_variable(init, name='output_w5', shape=[1])

        final_output = [b, output_w1 * l, output_w2 * fm_out]
        if third_order:
            third_fm_out = tf.reduce_sum(all_level_3_matrix, axis=-1)
            final_output.append(output_w3 * third_fm_out)
        if fourth_order:
            fourth_fm_out = tf.reduce_sum(all_level_4_matrix, axis=-1)
            final_output.append(output_w4 * fourth_fm_out)
        if retrain_stage:
            final_output.append(output_w5 * h)
        self.logits, self.outputs = output(final_output)

    def analyse_structure(self, sess, print_full_weight=False, epoch=None):
        import numpy as np
        if self.prune:
            wts, mask = sess.run(["edge_weight/normed_wts:0", "edge_weight/unpruned_mask:0"])
            if print_full_weight:
                outline = ""
                for j in range(wts.shape[0]):
                    outline += str(wts[j]) + ","
                outline += "\n"
                print("log avg auc all second weights for(epoch:%s)" % (epoch), outline)
            print("wts", wts[:10])
            print("mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("second alpha masked edge_num", sum(zeros_))
        if self.beta_prune:
            beta_mask = sess.run(self.beta_weight)
            beta_mask = np.reshape(beta_mask, (-1))
            print(beta_mask.shape)
            if print_full_weight:
                outline = ""
                for j in range(beta_mask.shape[0]):
                    outline += str(beta_mask[j]) + ","
                outline += "\n"
                print("log avg auc all second beta weights for(epoch:%s)" % (epoch), outline)
            print("beta_mask", beta_mask[:10])
            zeros_ = np.zeros_like(beta_mask, dtype=np.float32)
            zeros_[beta_mask == 0] = 1
            print("second beta masked edge_num", sum(zeros_))
        if self.third_prune:
            wts, mask = sess.run(["third_edge_weight/third_normed_wts:0", "third_edge_weight/third_unpruned_mask:0"])
            if print_full_weight:
                outline = ""
                for j in range(wts.shape[0]):
                    outline += str(wts[j]) + ","
                outline += "\n"
                print("log avg auc all third weights for(epoch:%s)" % (epoch), outline)
            print("wts", wts[:10])
            print("mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("third alpha masked third edge_num", sum(zeros_))
        if self.third_beta_prune:
            third_beta_mask = sess.run(self.third_beta_weight)
            third_beta_mask = np.reshape(third_beta_mask, (-1))
            print(third_beta_mask.shape)
            if print_full_weight:
                outline = ""
                for j in range(third_beta_mask.shape[0]):
                    outline += str(third_beta_mask[j]) + ","
                outline += "\n"
                print("log avg auc all third beta weights for(epoch:%s)" % (epoch), outline)
            print("third_beta_mask", third_beta_mask[:10])
            zeros_ = np.zeros_like(third_beta_mask, dtype=np.float32)
            zeros_[third_beta_mask == 0] = 1
            print("third beta masked third edge_num", sum(zeros_))
        if self.fourth_prune:
            wts, mask = sess.run(["fourth_edge_weight/fourth_normed_wts:0", "fourth_edge_weight/fourth_unpruned_mask:0"])
            if print_full_weight:
                outline = ""
                for j in range(wts.shape[0]):
                    outline += str(wts[j]) + ","
                outline += "\n"
                print("log avg auc all fourth weights for(epoch:%s)" % (epoch), outline)
            print("wts", wts[:10])
            print("mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("fourth alpha masked fourth edge_num", sum(zeros_))
        if self.fourth_beta_prune:
            fourth_beta_mask = sess.run(self.fourth_beta_weight)
            fourth_beta_mask = np.reshape(fourth_beta_mask, (-1))
            print(fourth_beta_mask.shape)
            if print_full_weight:
                outline = ""
                for j in range(fourth_beta_mask.shape[0]):
                    outline += str(fourth_beta_mask[j]) + ","
                outline += "\n"
                print("log avg auc all fourth beta weights for(epoch:%s)" % (epoch), outline)
            print("fourth_beta_mask", fourth_beta_mask[:10])
            zeros_ = np.zeros_like(fourth_beta_mask, dtype=np.float32)
            zeros_[fourth_beta_mask == 0] = 1
            print("fourth beta masked fourth edge_num", sum(zeros_))

    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                self.l2_loss = get_l2_loss([self.l2_w, self.l2_v],
                                           [self.xw, self.xv_list])
                self.l2_loss = None
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                if self.retrain_stage:
                    all_variable = [v for v in tf.trainable_variables()]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable)
                else:
                    weight_var = []
                    if self.selection_parameter["second_interaction"]:
                        weight_var += list(set(tf.get_collection("edge_weights")))
                    if self.selection_parameter["third_interaction"]:
                        weight_var += list(set(tf.get_collection("third_edge_weights")))
                    if self.selection_parameter["fourth_interaction"]:
                        weight_var += list(set(tf.get_collection("fourth_edge_weights")))
                    if self.selection_parameter["second_dimension"]:
                        weight_var += list(set(tf.get_collection("beta_edge_weights")))
                    if self.selection_parameter["third_dimension"]:
                        weight_var += list(set(tf.get_collection("third_beta_edge_weights")))
                    if self.selection_parameter["fourth_dimension"]:
                        weight_var += list(set(tf.get_collection("fourth_beta_edge_weights")))
                    weight_var = list(set(weight_var))
                    print(weight_var)
                    all_variable = [v for v in tf.trainable_variables()]
                    other_var = [i for i in all_variable if i not in weight_var]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=other_var)
                    self.optimizer2 = optimizer2.minimize(loss=_loss_, var_list=weight_var)
