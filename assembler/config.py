output_size = 16
input_size = 3
hidden_size = 16

batch_size = 20
comp_size = 100

data_path = '/home/laekov/spest/data/threadblock_break'

cuda = True

lr = 1e-3
weight_decay = 1e-6

train_iters = 1000000
validate_every = 1000

mave_weight = .999

load_path = 'ckpt/2.pt'

max_data_len = 16
