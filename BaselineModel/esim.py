###############
#20180615
#implementation of decomposable attention on cnli
################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import inspect
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from ops_cudnn_rnn import cudnn_lstm


class MyModel(object):
  """The ESIM model."""

  def __init__(self, is_training, config):

    batch_size = config.batch_size
    self.config = config
    self.is_training = is_training
    self.global_step = tf.Variable(0, trainable=False)
   
    self.add_placeholder() 
    self.add_embedding() 
    self.input_encoding()
    self.attend()
    self.compare() 
    self.aggregate() 

    self.compute_accuracy()
    self.compute_loss()   

    if not is_training:
        return
    self.optimization()

  def add_placeholder(self):
    '''
    add_placeholder for inputs
    '''
    self.x = tf.placeholder(tf.int32, [self.config.batch_size, self.config.xmaxlen])
    self.y = tf.placeholder(tf.int32, [self.config.batch_size, self.config.ymaxlen])

    self.x_mask = tf.placeholder(tf.int32, [self.config.batch_size, self.config.xmaxlen])
    self.y_mask = tf.placeholder(tf.int32, [self.config.batch_size, self.config.ymaxlen])
    self.x_mask = tf.cast(self.x_mask,tf.float32)
    self.y_mask = tf.cast(self.y_mask,tf.float32)

    self.x_len = tf.placeholder(tf.int32, [self.config.batch_size,])
    self.y_len = tf.placeholder(tf.int32, [self.config.batch_size,])
    self.x_len = tf.cast(self.x_len,tf.float32)
    self.y_len = tf.cast(self.y_len,tf.float32)

    self.label = tf.placeholder(tf.int32, [self.config.batch_size,self.config.num_classes])
  

  def add_embedding(self):
    '''
    add pretrained embedding
    '''
    with tf.device("/cpu:0"):
      embedding_matrix=np.load(self.config.cnli_embedding_dir)
      embedding = tf.Variable(embedding_matrix,trainable=False, name="embedding")
      
      self.input_xemb = tf.nn.embedding_lookup(embedding, self.x)
      self.input_yemb = tf.nn.embedding_lookup(embedding, self.y)
    
      if self.is_training and self.config.keep_prob < 1:
        self.input_xemb = tf.nn.dropout(self.input_xemb, self.config.keep_prob)
        self.input_yemb = tf.nn.dropout(self.input_yemb, self.config.keep_prob)



  def input_encoding(self):
    '''
    encode the x and y with a two-layer fnn seperately
    '''
    with tf.variable_scope("encode_xy") as scope:
      self.x_output = cudnn_lstm(inputs=self.input_xemb,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    
      self.x_output=self.x_output*self.x_mask[:,:,None]

      scope.reuse_variables()
      self.y_output = cudnn_lstm(inputs=self.input_yemb,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    
      self.y_output=self.y_output*self.y_mask[:,:,None]

      if self.is_training and self.config.keep_prob < 1:
        self.x_output = tf.nn.dropout(self.x_output,self.config.keep_prob)  # its length must be x_length
        self.y_output = tf.nn.dropout(self.y_output, self.config.keep_prob)


  def attend(self):
      self.weighted_y, self.weighted_x =self.attention(x_sen= self.x_output,
                                                       y_sen= self.y_output,
                                                       x_len= self.config.xmaxlen,
                                                       y_len= self.config.ymaxlen)


  def compare(self):

    with tf.variable_scope("compare"):
      with tf.variable_scope("compare-xy") as scope:
        co_xy = tf.concat([self.x_output,self.weighted_y, self.x_output-self.weighted_y, self.x_output*self.weighted_y],axis=-1) 
        co_xy_dense = tf.layers.dense(inputs=co_xy,units=self.config.hidden_units, activation=tf.nn.relu,
                                      kernel_regularizer=l2_regularizer(self.config.l2_strength),  use_bias=True)

        v_co_xy = cudnn_lstm(inputs=co_xy_dense,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    
        self.v_co_xy=v_co_xy*self.x_mask[:,:,None]


        scope.reuse_variables()
        co_yx = tf.concat([self.y_output,self.weighted_x, self.y_output-self.weighted_x, self.y_output*self.weighted_x],axis=-1) 
        co_yx_dense = tf.layers.dense(inputs=co_yx,units=self.config.hidden_units, activation=tf.nn.relu,
                                      kernel_regularizer=l2_regularizer(self.config.l2_strength),  use_bias=True,reuse=tf.AUTO_REUSE)

        v_co_yx = cudnn_lstm(inputs=co_yx_dense,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    
        self.v_co_yx=v_co_yx*self.y_mask[:,:,None]

        if self.is_training and self.config.keep_prob < 1:
          self.v_co_xy = tf.nn.dropout(self.v_co_xy,self.config.keep_prob)  
          self.v_co_yx = tf.nn.dropout(self.v_co_yx,self.config.keep_prob)  


  def aggregate(self):
    '''
    1. sum pooling   2. fnn
    ''' 
    with tf.variable_scope("pooling"):

      v_xyave = tf.div(tf.reduce_sum(self.v_co_xy, 1), tf.expand_dims(self.x_len, -1)) #div true length
      v_yxave = tf.div(tf.reduce_sum(self.v_co_yx, 1), tf.expand_dims(self.y_len,  -1)) #div true length
      v_xymax = tf.reduce_max(self.v_co_xy,axis=1)  #(b,2h)    
      v_yxmax = tf.reduce_max(self.v_co_yx,axis=1)  #(b,2h)

      self.v = tf.concat([v_xyave, v_xymax, v_yxave, v_yxmax],axis=-1) 

    with tf.variable_scope("pred-layer"):
  
      dense1 = tf.layers.dense(inputs=self.v,
                             units=self.config.hidden_units, 
                             activation=tf.nn.tanh,
                             use_bias=True,
                             kernel_regularizer= l2_regularizer(self.config.l2_strength),
                             name="dense-pred-W")

      if self.is_training and self.config.keep_prob < 1:
        dense1 = tf.nn.dropout(dense1, self.config.keep_prob)

      W_pred = tf.get_variable("W_pred", shape=[self.config.hidden_units, self.config.num_classes],regularizer=l2_regularizer(self.config.l2_strength))

      self.pred = tf.nn.softmax(tf.matmul(dense1, W_pred), name="pred")

  def compute_accuracy(self):
    correct = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.label,1))
    self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

  def compute_loss(self):
    
    self.loss_term = -tf.reduce_sum(tf.cast(self.label,tf.float32) * tf.log(self.pred),name="loss_term")
    self.reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="reg_term")
    self.loss = tf.add(self.loss_term,self.reg_term,name="loss")


  def optimization(self):
  
    with tf.variable_scope("bp_layer"):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      self.config.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
      self.optim = optimizer.apply_gradients(
          zip(grads, tvars),
          global_step=self.global_step)



  def attention(self,x_sen,y_sen,x_len,y_len):
    '''
    function: use the dot-production of left_sen and right_sen to compute the attention weight matrix
    :param left_sen: a list of 2D tensor (x_len,hidden_units)
    :param right_sen: a list of 2D tensor (y_len,hidden_units)
    :return: (1) weighted_y: the weightd sum of y_sen, a 3D tensor with shape (b,x_len,2*h)
             (2) weghted_x:  the weighted sum of x_sen, a 3D tensor with shape (b,y_len,2*h)
    '''
    
    weight_matrix =tf.matmul(x_sen, tf.transpose(y_sen,perm=[0,2,1])) #(b,x_len,h) x (b,h,y_len)->(b,x_len,y_len)

    weight_matrix_y =tf.exp(weight_matrix - tf.reduce_max(weight_matrix,axis=2,keep_dims=True))  #(b,x_len,y_len)
    weight_matrix_x =tf.exp(tf.transpose((weight_matrix - tf.reduce_max(weight_matrix,axis=1,keep_dims=True)),perm=[0,2,1]))  #(b,y_len,x_len)

    weight_matrix_y=weight_matrix_y*self.y_mask[:,None,:]#(b,x_len,y_len)*(b,1,y_len)
    weight_matrix_x=weight_matrix_x*self.x_mask[:,None,:]#(b,y_len,x_len)*(b,1,x_len)
    
    alpha=weight_matrix_y/(tf.reduce_sum(weight_matrix_y,2,keep_dims=True)+1e-8)#(b,x_len,y_len)
    beta=weight_matrix_x/(tf.reduce_sum(weight_matrix_x,2,keep_dims=True)+1e-8)#(b,y_len,x_len)

    #(b,1,y_len,2*h)*(b,x_len,y_len,1)*=>(b,x_len,y_len,2*h) =>(b,x_len,2*h)
    weighted_y =tf.reduce_sum(tf.expand_dims(y_sen,1) *tf.expand_dims(alpha,-1),2)

    #(b,1,x_len,2*h)*(b,y_len,x_len,1) =>(b,y_len,x_len,2*h) =>(b,y_len,2*h)
    weighted_x =tf.reduce_sum(tf.expand_dims(x_sen,1) * tf.expand_dims(beta,-1),2)

    return weighted_y,weighted_x


  def two_layer_dense(self,inp,out_dim,scope,regularizer=None):
    with tf.variable_scope(scope):
      dense1 = tf.layers.dense(inputs=inp,
                             units=out_dim, 
                             activation=tf.nn.relu,
                             kernel_regularizer= regularizer,
                             use_bias=True)

      dense2 = tf.layers.dense(inputs=dense1,
                             units=out_dim, 
                             activation=tf.nn.relu,
                             kernel_regularizer= regularizer,
                             use_bias=True)
      return dense2

