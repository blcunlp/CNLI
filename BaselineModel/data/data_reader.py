import tensorflow as tf
import os
import json
from myutils import *
from collections import Counter

from six.moves import xrange
import numpy as np
_PAD="_PAD"
_UNK= "_UNK"
_GO= "_GO"
_EOS= "_EOS"
_START_VOCAB=[_PAD,_UNK,_GO,_EOS]

PAD_ID=0
UNK_ID=1
GO_ID =2
EOS_ID =3

def filter_length(seq,maxlen):
  if len(seq)>maxlen:
    new_seq=seq[:maxlen]
  else:
    new_seq=seq
  return new_seq

def load_data(train,vocab,labels={'neutral':0,'entailment':1,'contradiction':2}):
    X,Y,Z=[],[],[]
    for p,h,l in train:
        p=map_to_idx(tokenize(p),vocab)+ [EOS_ID]
        h=[GO_ID]+map_to_idx(tokenize(h),vocab)+ [EOS_ID]
        p=filter_length(p,32)
        h=filter_length(h,30)
        if l in labels:        
            X+=[p]
            Y+=[h]
            Z+=[labels[l]]
    return X,Y,Z

def get_vocab(data):
    vocab=Counter()
    for ex in data:
        tokens=tokenize(ex[0])
        tokens+=tokenize(ex[1])
        vocab.update(tokens)
    vocab_sorted = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    lst = _START_VOCAB + [ x for x, y in vocab_sorted if y > 0]

    vocab_exist=os.path.isfile("./data/embedding/cnli_vocab.txt")

    #if not vocab_exist:
    print ("build cnli_vocab.txt")
    f =open("./data/embedding/cnli_vocab.txt","w+")
    for x,y in enumerate(lst):
      x_y = str(y) +"\t"+ str(x)+"\n"
      f.write(x_y)
    f.close()

    os.system('./data/embedding/run_embedding.sh') 
    vocab = dict([ (y,x) for x,y in enumerate(lst)])
    return vocab


class DataSet(object):
  def __init__(self,x,y,labels,x_len,y_len,X_mask,Y_mask):
    self._data_len=len(x)
    self._x =x
    self._y =y
    self._labels =labels
    self._x_len = x_len
    self._y_len = y_len
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = x.shape[0]
    self._x_mask=X_mask
    self._y_mask=Y_mask

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    batch_x, batch_x_mask, batch_x_len = self._x[start:end], self._x_mask[start:end], self._x_len[start:end]
    batch_y,batch_y_mask, batch_y_len = self._y[start:end], self._y_mask[start:end], self._y_len[start:end]
    batch_labels = self._labels[start:end]
    
    return batch_x,batch_y, batch_labels,batch_x_mask,batch_y_mask,batch_x_len,batch_y_len

  @property
  def get_x(self):
    return self._x
  
  @property
  def get_y(self):
    return self.y

  @property
  def labels(self):
    return self._labels

  @property
  def get_x_len(self):
    return self._x_len
  
  @property
  def get_y_len(self):
    return self._y_len

  @property
  def get_data_num(self):
    return self._data_len
  
  def get_epoch_size(self,batch_size):
    epoch_size = self._data_len //batch_size
    return epoch_size

def singlefile2seqid(data,vocab, config):
  X_data, Y_data,  Z_data = load_data(data, vocab)

  X_data_lengths=np.asarray([len(x) for x in X_data]).reshape(len(X_data))
  X_data_mask = np.asarray([np.ones(x) for x in X_data_lengths]).reshape(len(X_data_lengths))
  X_data_mask=pad_sequences(X_data_mask, maxlen=config.xmaxlen, value=vocab[_PAD], padding='post')
  X_data=pad_sequences(X_data, maxlen=config.xmaxlen, value=vocab[_PAD], padding='post')

  Y_data_lengths = np.asarray([len(x) for x in Y_data]).reshape(len(Y_data))
  Y_data_mask = np.asarray([np.ones(x) for x in Y_data_lengths]).reshape(len(Y_data_lengths))
  Y_data_mask = pad_sequences(Y_data_mask, maxlen=config.ymaxlen, value=vocab[_PAD], padding='post')
  Y_data = pad_sequences(Y_data, maxlen=config.ymaxlen, value=vocab[_PAD], padding='post')


  Z_data = to_categorical(Z_data, num_classes=config.num_classes)
  #X_data = np.asarray(X_data)
  dataset = DataSet(X_data,Y_data,Z_data,\
                    X_data_lengths,Y_data_lengths,
                    X_data_mask,Y_data_mask)

  return dataset

def file2seqid(config):

  xmaxlen = config.xmaxlen
  ymaxlen = config.ymaxlen
  train = [l.strip().split('\t') for l in open(config.train_file)]
  dev = [l.strip().split('\t') for l in open(config.dev_file)]
  vocab = get_vocab(train)

  Train = singlefile2seqid(train,vocab, config)
  Dev = singlefile2seqid(dev,vocab, config)
  return Train,Dev,vocab
 
  

 
if __name__=="__main__":

    train=[l.strip().split('\t') for l in open('train.txt')][:20000]
    dev=[l.strip().split('\t') for l in open('dev.txt')]
    test=[l.strip().split('\t') for l in open('test.txt')]
    labels={'neutral':0,'entailment':1,'contradiction':2}

    vocab=get_vocab(train)
    #X_train,Y_train,Z_train=load_data(train,vocab)
    X_dev,Y_dev,Z_dev=load_data(dev,vocab)
    #print (len(X_train),X_train[0])
    print (len(X_dev),X_dev[0])
    print (len(Y_dev),Y_dev[0])
    print (len(Z_dev),Z_dev[0])
