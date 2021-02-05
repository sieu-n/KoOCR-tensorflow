import tensorflow as tf

class GlobalWeightedAveragePooling(tf.keras.layers.Layer):
  #Implementation of GlobalWeightedAveragePooling
  def __init__(self,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)):
    #self.num_outputs = num_outputs
    self.kernel_initializer =kernel_initializer
    super(GlobalWeightedAveragePooling, self).__init__()
  def build(self, input_shape):
    #input_shape=(w,h,c)
    self.kernel = self.add_weight("kernel",shape=input_shape[1:],initializer=self.kernel_initializer)

  def call(self, input):
    com=tf.math.multiply(input, self.kernel)
    return tf.reduce_mean(com,axis=[1,2])

class GlobalWeightedOutputAveragePooling(tf.keras.layers.Layer):
  def __init__(self):
    #self.num_outputs = num_outputs
    super(GlobalWeightedOutputAveragePooling, self).__init__()
  def build(self, input_shape):
    #input_shape=(w,h,c)
    self.kernel = self.add_weight("kernel",shape=(input_shape[-1],))

  def call(self, input):
    out=tf.keras.layers.GlobalAveragePooling2D()(input)
    return tf.math.multiply(input, self.kernel)