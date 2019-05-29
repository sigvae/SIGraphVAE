#!/usr/bin/env python
# coding: utf-8

# In[17]:


# import libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
#%matplotlib inline

import numpy as np
import os
import sys
import seaborn as sns
import scipy.spatial.distance
from matplotlib import pyplot as plt
import pandas as pd 
import scipy.stats as stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from graphCNN import *
#import pygsp
import scipy.sparse as sp
from input_data import load_data
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


# In[19]:


# citation load data

# adj, features = load_data('pubmed')

adj = np.loadtxt('data/ns_adj.txt')
adj = sp.csr_matrix(adj)

features = np.loadtxt('data/ns_z_mean_128.txt')
features = sp.lil_matrix(features)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

adj_label = adj_train + sp.eye(adj_train.shape[0])

adj_norm = preprocess_graph(adj)
adj_norm_dense = scipy.sparse.coo_matrix((adj_norm[1], (adj_norm[0][:,0],adj_norm[0][:,1])), shape=adj_norm[2]).toarray()


# Some preprocessing
num_nodes = adj.shape[0]
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]
features_dense = scipy.sparse.coo_matrix((features[1], (features[0][:,0],features[0][:,1])), shape=features[2]).toarray()

train_xs = features_dense


# In[20]:


# garaph cnn function

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs
    
class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class GraphConvolutionK(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionK, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.output_dim = output_dim

    def _call(self, inputs):
        K = inputs.shape[1].value
#         outputs = tf.zeros([inputs.shape[0].value, K, self.output_dim])
        for i in range(K):
            x = tf.squeeze(inputs[:,i,:])
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            if i == 0:
                outputs = tf.expand_dims(self.act(x), axis=1)
            else:
                outputs = tf.concat([outputs, tf.expand_dims(self.act(x), axis=1)], axis=1)
                
        return outputs

class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
#         x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

class SparseDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., **kwargs):
        super(SparseDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        with tf.variable_scope(self.name + '_rk'):
            initial = tf.random_uniform([16], minval=-6,
                                        maxval=0, dtype=tf.float32)
            self.rk = tf.sigmoid(tf.Variable(initial, name='rk'))
#             self.rk = 2*tf.ones([16])

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(tf.diag(self.rk), x)
        x = tf.matmul(inputs, x)
        outputs = 1 - tf.exp(- tf.exp(x))
        return outputs, self.rk

class GCNNModel(Layer):
    """Stack of graph convolutional layers."""
    def __init__(self, num_layers, output_dims, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GCNNModel, self).__init__(**kwargs)
        
        self.output_dims = output_dims
        self.num_layers = num_layers
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = GraphConvolution(input_dim = x.shape[1].value, 
                     output_dim = self.output_dims[i], 
                     adj = self.adj,
                     act = self.act, 
                     dropout = self.dropout, 
                     logging = False)(x)
        return x

class GCNNModelK(Layer):
    """Stack of graph convolutional layers."""
    def __init__(self, num_layers, output_dims, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GCNNModelK, self).__init__(**kwargs)
        
        self.output_dims = output_dims
        self.num_layers = num_layers
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = GraphConvolutionK(input_dim = x.shape[2].value,
                                  output_dim = self.output_dims[i],
                                  adj = self.adj,
                                  act = self.act,
                                  dropout = self.dropout,
                                  logging = False)(x)
        return x


# In[31]:


# distribution functions

Exponential=tf.contrib.distributions.Exponential(rate=1.0)
Normal=tf.contrib.distributions.Normal(loc=0., scale=1.)
Mvn=tf.contrib.distributions.MultivariateNormalDiag
Bernoulli = tf.contrib.distributions.Bernoulli
plt.ioff()

sys.path.append(os.getcwd())

def sample_psi(x, adjacency_sparse, noise_dim, K, z_dim, reuse=False): 
    
    with tf.variable_scope("hyper_psi") as scope:
        if reuse:
            scope.reuse_variables()
        
        x_0 = tf.expand_dims(x, axis=1)
        x_1 = tf.tile(x_0, [1,K,1])   #N*K*784
        
        B3 = Bernoulli(0.5)
        e3 = tf.cast(B3.sample([tf.shape(x_1)[0], K, noise_dim[0]]),tf.float32)
        input_ = tf.concat([e3, x_1],axis=2)
        h3 = GCNNModelK(num_layers=1
                        ,output_dims=[32]
                        ,adj=adjacency_sparse
                        ,dropout = 0.)(input_)
        
        

        mu = GraphConvolutionK(input_dim = 32
                               ,output_dim = z_dim
                               ,adj = adjacency_sparse
                               ,act = lambda x: x
                               ,dropout = 0.
                               ,logging = False)(h3)

    return mu

def sample_logv(x, adjacency_sparse, noise_dim, z_dim, reuse=False): 
    with tf.variable_scope("hyper_sigma") as scope:
        if reuse:
            scope.reuse_variables()
        
#         net1 = GraphConvolutionSparse(input_dim=x.shape[1].value
#                                      ,output_dim=256
#                                      ,adj=adjacency_sparse
#                                      ,dropout=0.
#                                      ,act=tf.nn.relu
#                                      ,features_nonzero=features_nonzero
#                                      ,logging=False)(x)
        
        net1 = GCNNModel(num_layers=1
                         ,output_dims=[32]
                         ,adj=adjacency_sparse
                         ,act=tf.nn.relu
                         ,dropout=0.)(x)
        
        z_logv = GraphConvolution(input_dim = 32
                                  ,output_dim = z_dim
                                  ,adj = adjacency_sparse
#                                   ,act = tf.nn.relu
                                  ,act = lambda x: x
                                  ,dropout = 0.
                                  ,logging = False)(net1)
    
    return z_logv

def sample_n(psi, sigma):
    eps = tf.random_normal(shape=tf.shape(psi))
    z=psi+eps*sigma
    return z

def decoder(z, h_dim, reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        
        K = z.shape[1].value
        for i in range(K):
            input_ = tf.squeeze(z[:,i,:])
            logits_x = InnerProductDecoder(input_dim = h_dim
                                      ,act = lambda x: x
                                      ,logging = False)(input_)                          
            if i == 0:
                outputs = tf.expand_dims(logits_x, axis=2)
            else:
                outputs = tf.concat([outputs, tf.expand_dims(logits_x, axis=2)], axis=2)
        return outputs


# In[32]:


# ROC calculation functions

def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
    #    feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(psi_iw_vec, {x: train_xs, WU: warm_up})

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

      # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_roc_score_sp(edges_pos, edges_neg, emb=None):
    if emb is None:
    #    feed_dict.update({placeholders['dropout']: 0})
        [emb, rk] = sess.run([psi_iw_vec, rks], {x: train_xs, WU: warm_up})

    # Predict on test set of edges
    tmp = np.dot(np.diag(rk), emb.T)
    adj_rec = 1 - np.exp(- np.exp(np.dot(emb, tmp)))
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


# In[33]:


# model hyperparameters

noise_dim = [32]
z_dim = 16
x_dim = adj_norm_dense.shape[0]
eps = 1e-10
lr = 0.0005
training_epochs = 1500
display_step = 50
cost_val = []
acc_val = []
val_roc_score = []
tst_roc_score = []


pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


# In[34]:


# building model

tf.reset_default_graph() 

# creating sparse adjacency matrix
adjacency_dense = tf.convert_to_tensor(adj_norm_dense.astype(np.float32))
adjacency_orig_dense = tf.convert_to_tensor(adj_label.toarray().astype(np.float32))
zero = tf.constant(0, dtype=tf.float32)
where = tf.not_equal(adjacency_dense, zero)
indices = tf.where(where)
values = tf.gather_nd(adjacency_dense, indices)
adjacency_sparse = tf.SparseTensor(indices, values, adjacency_dense.shape)


x = tf.placeholder(tf.float32,[x_dim, train_xs.shape[1]])
merge = tf.constant(1)
WU = tf.placeholder(tf.float32, shape=())
K = tf.constant(2000)
J = tf.constant(150)

z_logv = sample_logv(x, adjacency_sparse, noise_dim, z_dim)
z_logv_iw = tf.tile(tf.expand_dims(z_logv, axis=1),[1,K,1])
sigma_iw1 = tf.exp(z_logv_iw/2)
sigma_iw2 = tf.tile(tf.expand_dims(sigma_iw1,axis=2),[1,1,J+1,1])

psi_iw = sample_psi(x, adjacency_sparse, noise_dim, K, z_dim)
psi_iw_vec = tf.reduce_mean(psi_iw, axis=1)
z_sample_iw = sample_n(psi_iw, sigma_iw1)

z_sample_iw1 = tf.expand_dims(z_sample_iw,axis=2)
z_sample_iw2 = tf.tile(z_sample_iw1,[1,1,J+1,1])

psi_iw_star = sample_psi(x, adjacency_sparse, noise_dim, J, z_dim, reuse=True)
psi_iw_star0 = tf.expand_dims(psi_iw_star, axis=1)
psi_iw_star1 = tf.tile(psi_iw_star0,[1,K,1,1])
psi_iw_star2 = tf.concat([psi_iw_star1, tf.expand_dims(psi_iw,axis=2)],2)


ker = tf.exp(-0.5*tf.reduce_sum(tf.square(z_sample_iw2 - psi_iw_star2)/tf.square(sigma_iw2 + eps),3))

log_H_iw_vec = tf.log(tf.reduce_mean(ker, axis=2) + eps)-0.5*tf.reduce_sum(z_logv_iw, 2)
log_H_iw = tf.reduce_mean(log_H_iw_vec, axis=0)

log_prior_iw_vec = -0.5*tf.reduce_sum(tf.square(z_sample_iw), 2)
log_prior_iw = tf.reduce_mean(log_prior_iw_vec, axis=0)


x_iw = tf.tile(tf.expand_dims(x, axis=1),[1,K,1])
logits_x_iw = decoder(z_sample_iw, x_dim)
reconstruct_iw = logits_x_iw

adj_orig_tile = tf.expand_dims(adjacency_orig_dense, -1)
adj_orig_tile = tf.tile(adj_orig_tile, multiples=[1,1,K])

log_lik_iw = -1 * norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstruct_iw
                                                                                ,targets=adj_orig_tile
                                                                                ,pos_weight=pos_weight)
                                       ,axis=[0,1])

loss_iw0 = -tf.reduce_logsumexp(log_lik_iw+(log_prior_iw-log_H_iw)*WU/num_nodes) + tf.log(tf.cast(K, tf.float32))
loss_iw = loss_iw0


# In[35]:


# optimization

var_all = tf.trainable_variables()
g_step = tf.Variable(0, name='g_step', trainable=False)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_iw, var_list=var_all, global_step=g_step)

init_op=tf.global_variables_initializer()


# In[36]:


# traininig

dat_train=[]
dat_test=[]

sess=tf.InteractiveSession()
sess.run(init_op)
saver = tf.train.Saver()

print("This is SIG-VAE-IP test")


warm_up = 0

for epoch in range(training_epochs):
    warm_up = np.min([epoch/300,1])
    
    _, cost = sess.run([train_op, loss_iw], {x: train_xs, WU: warm_up})
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % epoch, "cost_train=", "{:.9f}".format(cost))
        
    if epoch>300:
        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
        val_roc_score.append(roc_curr)
        
        roc_currt, ap_currt = get_roc_score(test_edges, test_edges_false)
        tst_roc_score.append(roc_currt)
        
        print("Epoch:", '%04d' % epoch, "val_ap=", "{:.5f}".format(ap_curr))
        print("Epoch:", '%04d' % epoch, "val_roc=", "{:.5f}".format(val_roc_score[-1]))
        print("Epoch:", '%04d' % epoch, "test_ap=", "{:.5f}".format(ap_currt))
        print("Epoch:", '%04d' % epoch, "test_roc=", "{:.5f}".format(tst_roc_score[-1]))
        print('--------------------------------')

