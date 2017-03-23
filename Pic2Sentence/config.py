class Config(object):
    num_alex_net=160
    batch_size = 160
    epoch = 20000
    initial_learning_rate = 0.1
    keep_prob = 0.5
    lstm_max_time = 20
    pic_size = [150, 150, 3]
    vocab_size = 2401
    output_embedding_size = 300
    #alex_net's parameters
    windows_kernel = [7, 5, 3, 3, 5]  #len(windows_kernel) = num_convolution_layers, in n convolution layer,
                         #kernel_h = kernel_w = windows_kernel[n-1]
    windows_pool = [2, 2, 2, 2, 2]
    strides_kernel = [2, 1, 1, 1, 1]
    strides_pool = [2 , 2, 2, 2, 2]
    c_os = [96, 256, 384, 384, 256] #output channels
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    num_fc_units = [256,4096, 4096, 1024]
class AlexConfig(object):
    num_alex_net = 100
    batch_size = num_alex_net
    windows_kernel = []  #len(windows_kernel) = num_convolution_layers, in n convolution layer,
                         #kernel_h = kernel_w = windows_kernel[n-1]
    windows_pool = []
    strides_kernel = []
    strides_pool = []
    c_os = [] #output channels
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    num_fc_units = []
    pass

class LSTMConfig(object):
    lstm_max_time = 20
    batch_size = 1
    num_unit = Config.output_embedding_size
    pass