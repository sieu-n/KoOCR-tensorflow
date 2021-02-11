from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

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

class GradCAMMultiOutput:
  #CAM Implementaion referenced from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.chosung,self.jungsung,self.jongsung = classIdx
		self.layerName = layerName
		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()

  def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
  
  def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output,
				self.model.output])
    # record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)
