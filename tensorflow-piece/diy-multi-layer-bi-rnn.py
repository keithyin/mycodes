"""
tensorflow 1.1.0版本
"""
def stacked_bidirectional_rnn(RNN, num_units, num_layers, inputs, seq_lengths, batch_size):
    """
    multi layer bidirectional rnn
    :param RNN: RNN class, e.g. LSTMCell
    :param num_units: int, hidden unit of RNN cell
    :param num_layers: int, the number of layers
    :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
    :param seq_lengths: list, sequence length, a list of sequence lengths, the length of the list is batch_size
    :param batch_size: int 
    :return: the output of last layer bidirectional rnn with concatenating
    这里用到几个tf的特性
    1. tf.variable_scope(None, default_name="bidirectional-rnn")使用default_name
    的话,tf会自动处理命名冲突
    """
    _inputs = inputs
    if len(_inputs.get_shape().as_list()) != 3:
    	raise ValueError("the inputs must be 3-dimentional")
    
    for _ in range(num_layers):
        #为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
        #恶心的很.如果不在这加的话,会报错的.
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = RNN(num_units)
            rnn_cell_bw = RNN(num_units)
            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                              initial_state_fw, initial_state_bw, dtype=tf.float32)
            _inputs = tf.concat(output, 2)
    return _inputs
