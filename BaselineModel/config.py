class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.0003

  max_grad_norm = 5
  xmaxlen=32
  ymaxlen=30
  num_classes=3
  hidden_units = 300
  embedding_size =300
  MAXITER=70
  keep_prob = 0.8
              
  batch_size = 32
  l2_strength=0.0003

  early_stopping=5
 
  train_file='./data/cnli/cnli_train_1.0_seg.txt'
  dev_file='./data/cnli/cnli_dev_1.0_seg.txt'

  cnli_embedding_dir= './data/embedding/cnli_embedding.npy'

