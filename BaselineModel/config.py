class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.0003
  #learning_rate = 0.0004
  min_lr = 0.000005
  lr_decay = 0.9 
  max_epoch = 8  
  max_max_epoch = 4

  diff_var=1

  max_grad_norm = 5
  num_layers = 1
  xmaxlen=32
  ymaxlen=30
  num_classes=3
  hidden_units = 300
  embedding_size =300
  MAXITER=70
  keep_prob = 0.8
              
  batch_size = 32
  vocab_size = 40000
  l2_strength=0.0003


  early_stopping=10
 
  change_epoch = 5
  update_learning = 5
  train_file='./data/cnli/cnli_train_beta1_seg.txt'
  dev_file='./data/cnli/cnli_dev_seg.txt'

  cnli_embedding_dir= './data/embedding/cnli_embedding.npy'


