output_size = 128
input_size = 1
hidden_size = 256

batch_size = 20
comp_size = 100

data_path = '/home/laekov/spest/data/threadblock_break'

cuda = True

lr = 1e-4
weight_decay = 1e-6

train_iters = 1000000
validate_every = 1000

mave_weight = .999

load_path = 'ckpt/2.pt'
