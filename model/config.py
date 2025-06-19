import torch 

batch_size = 64 
block_size = 256
n_embd = 384
n_head = 6 
n_layer = 6
dropout = 0.2
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
eval_iters = 200