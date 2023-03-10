import tensorflow as tf

class L1Conv2D(tf.keras.layers.Layer):
  def __init__(self, kernel_size, strides=[1,1], padding='SAME', dilated=1):
    super(L1Conv2D, self).__init__()
    self.kernel_h, self.kernel_w, self.kernel_in_ch, self.kernel_out_ch = kernel_size
    if type(dilated) is int:
      self.dilated = [dilated, dilated]
    else:
      self.dilated = dilated
    if type(strides) is int:
      self.strides = [strides, strides]
    else:
      self.strides = strides
    self.padding = padding

  def build(self, input_shape):
    if self.padding == 'SAME':
        pad_h, pad_w = (self.kernel_h-1)//2, (self.kernel_w-1)//2
    elif self.padding == 'VALID':
        pad_h, pad_w = (0,0)
    else:
        raise NotImplementedError("padding %s not implement" % self.padding)
    out_h, out_w = (input_shape[1]-self.kernel_h+2*pad_h)//self.strides[0] + 1, (input_shape[2]-self.kernel_w+2*pad_w)//self.strides[1] + 1
    self.input_reshape = tf.keras.layers.Reshape([out_h, out_w, self.kernel_h*self.kernel_w*self.kernel_in_ch, 1])
    self.kernel = self.add_weight("kernel",
                    shape=[self.kernel_h, self.kernel_w, self.kernel_in_ch, self.kernel_out_ch])
    self.subtract_layer = tf.keras.layers.Subtract()

  def call(self, inputs):
    def custom_subgraph_gradient(forward_pass_graph, backward_pass_graph):
      '''
      A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
      a op that behave as f(x) in forward mode,
      but as g(x) in the backward mode.
      '''
      return backward_pass_graph + tf.stop_gradient(forward_pass_graph - backward_pass_graph)

    # im2col
    inputs_patches = tf.image.extract_patches(inputs,
                          sizes=[1, self.kernel_h, self.kernel_w, 1],
                          strides=[1, self.strides[0], self.strides[1], 1],
                          rates=[1, self.dilated[0], self.dilated[1], 1],   # dilated conv
                          padding=self.padding)
    inputs_patches = self.input_reshape(inputs_patches)

    kernel_reshape = tf.reshape(self.kernel, [1, 1, 1, self.kernel_h*self.kernel_w*self.kernel_in_ch, self.kernel_out_ch])

    differ = self.subtract_layer([inputs_patches, kernel_reshape])
    forward = -tf.abs(differ)   # L1 norm
    backward = -differ**2       # L2 norm
    l1_fd_l2_bd = custom_subgraph_gradient(forward, backward)
    result = tf.reduce_sum(l1_fd_l2_bd, 3)
    return result

class L1Dense(tf.keras.layers.Layer):
  def __init__(self, out_dims):
    super(L1Dense, self).__init__()
    self.out_dims = out_dims

  def build(self, input_shape):
    self.in_dims = input_shape[-1]
    self.kernel = self.add_weight("kernel",
                    shape=[self.in_dims, self.out_dims])
    self.subtract_layer = tf.keras.layers.Subtract()
    self.input_reshape = tf.keras.layers.Reshape([self.in_dims, 1])

  def call(self, inputs):
    def custom_subgraph_gradient(forward_pass_graph, backward_pass_graph):
      '''
      A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
      a op that behave as f(x) in forward mode,
      but as g(x) in the backward mode.
      '''
      return backward_pass_graph + tf.stop_gradient(forward_pass_graph - backward_pass_graph)

    inputs_reshape = self.input_reshape(inputs)
    kernel_reshape = tf.reshape(self.kernel, [1, self.in_dims, self.out_dims])

    differ = self.subtract_layer([inputs_reshape, kernel_reshape])
    forward = -tf.abs(differ)   # L1 norm
    backward = -differ**2       # L2 norm
    l1_fd_l2_bd = custom_subgraph_gradient(forward, backward)
    result = tf.reduce_sum(l1_fd_l2_bd, 1)
    return result
