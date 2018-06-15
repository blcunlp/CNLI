from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import inspect
import logging
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from tensorflow.python.ops import variable_scope

from myutils import *
import data_reader as reader
from decomposable_att import MyModel 
from config import SmallConfig

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path","model_saved",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_float('learning_rate', 0.0004, 'Initial learning rate.')  
flags.DEFINE_float('keep_prob', 0.8, 'keep_prob for dropout.')  
flags.DEFINE_float('l2_strength', 0.0002, 'l2 rate for l2 loss.') 
flags.DEFINE_integer('batch_size', 32,'batch_size ') 
flags.DEFINE_bool('direction', "forward", 'forward or  reverse')

FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def fill_placeholder(data, model,config):
  batch_x,batch_y,batch_label,batch_x_mask,batch_y_mask, batch_x_len,batch_y_len= data.next_batch(config.batch_size)
  feed_dict = {model.x:batch_x , 
                model.y:batch_y,
                model.label:batch_label,
                model.x_mask:batch_x_mask,
                model.y_mask:batch_y_mask, 
                model.x_len :batch_x_len,
                model.y_len :batch_y_len,
                }

  return feed_dict

def run_epoch(session, data,model,config, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  losses = 0.0
  iters = 0
  acc_total=0.0
  fetches = {
      "acc":model.acc,
      "loss": model.loss,
      "global_step":model.global_step,
      "pred": model.pred,
      "label": model.label,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  
  start_time = time.time()
  epoch_size = data.get_epoch_size(config.batch_size)
  for step in range(epoch_size):
    feed_dict = fill_placeholder(data,model,config)
    
    vals = session.run(fetches, feed_dict)
    acc = vals["acc"]
    loss = vals["loss"]
    global_step=vals["global_step"]

    
    pred = vals["pred"]
    label = vals["label"]

    losses += loss
    iters= iters+1
    acc_total += acc
    #if verbose and step %10 == 0:
    #  print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters
  return acc_average,loss_average,global_step,pred,label


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  config = get_config()
  config.learning_rate = FLAGS.learning_rate
  config.keep_prob = FLAGS.keep_prob
  config.l2_strength = FLAGS.l2_strength
  config.batch_size = FLAGS.batch_size

  eval_config= copy.deepcopy(config)
  eval_config.batch_size=1
  print("config",vars(config))
  print("eval_config",vars(eval_config))

  Train,Dev,vocab = reader.file2seqid(config)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = MyModel(is_training=True, config=config)
    
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = MyModel(is_training=False,config=eval_config)

    
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      print ("model params",np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))
      t0=time.time()
      best_dev_acc = 0.0
      best_val_epoch = 0 
      last_change_epoch = 0


      for i in range(config.MAXITER):
        start_time=time.time()
        train_acc,train_loss,train_global_step,train_pred,train_label= run_epoch(session,data=Train, model=m,config=config, eval_op=m.optim, verbose=True)
        print("Epoch: %d train_acc: %.3f train_loss %.4f train_global_step:%s" % (i ,train_acc,train_loss,train_global_step))

        dev_acc,dev_loss,_,dev_pred,dev_label= run_epoch(session,data=Dev,model=mvalid,config=eval_config)
        print("Epoch: %d dev_acc: %.3f dev_loss %.4f" % (i , dev_acc,dev_loss))


        sys.stdout.flush()
        if best_dev_acc <= dev_acc:
          best_dev_acc = dev_acc
          best_val_epoch = i
          if FLAGS.save_path:
            print("train_global_step:%s.  Saving %d model to %s." % (train_global_step,i,FLAGS.save_path))
            sv.saver.save(session,FLAGS.save_path+"/model", global_step=train_global_step)
            print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        
        end_time=time.time()
        print("################# all_training time: %s one_epoch time: %s ############### " % ((end_time-t0)//60, (end_time-start_time)//60))
        if i - best_val_epoch > config.early_stopping:
          print ("best_val_epoch:%d  best_val_accuracy:%.4f"%(best_val_epoch,best_dev_acc))
          logging.info("Normal Early stop")
          print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
          break        
        elif i == config.MAXITER-1:
          print ("best_val_epoch:%d  best_val_accuracy:%.4f"%(best_val_epoch,best_dev_acc))
          logging.info("Finishe Training")

      
if __name__ == "__main__":
  tf.app.run()
