import tensorflow as tf
class GlobalWeightedAveragePooling(tf.keras.layers.Layer):
  def __init__(self):
    self.num_outputs = num_outputs

  def build(self, input_shape):
    #input_shape=(w,h,c)
    self.kernel = self.add_weight("kernel",shape=input_shape)

  def call(self, input):
    return tf.math.multiply(input, self.kernel)

