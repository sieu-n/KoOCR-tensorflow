from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import utils.korean_manager as korean_manager
class GlobalWeightedAveragePooling(tf.keras.layers.Layer):
	#Implementation of GlobalWeightedAveragePooling
	def __init__(self,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)):
		#self.num_outputs = num_outputs
		self.kernel_initializer =kernel_initializer
		super(GlobalWeightedAveragePooling, self).__init__()
    
	def build(self, input_shape):
		#input_shape=(w,h,c)
		self.kernel = self.add_weight("kernel",shape=input_shape[1:],initializer=self.kernel_initializer)

	def get_config(self):
		config = super().get_config().copy()
		config.update({'kernel': self.kernel})
		return config
    
	def call(self, input):
		com=tf.math.multiply(input, self.kernel)
		return tf.math.reduce_sum(com,axis=[1,2])

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

class MultiOutputGradCAM:
  #CAM Implementaion referenced from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
	def __init__(self, model, character):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.character_Idx= korean_manager.korean_split(character)
		# if the layer name is None, attempt to automatically find
		# the target output layer
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
		gradModel = Model(inputs=[self.model.inputs],\
		outputs=[self.model.get_layer(self.layerName).output]+[self.model.output])

		heatmap_list=[]
		for component in range(3):
			# record operations for automatic differentiation
			with tf.GradientTape() as tape:
				# cast the image tensor to a float-32 data type, pass the
				# image through the gradient model, and grab the loss
				# associated with the specific class index
				inputs = tf.cast(image, tf.float32)
				(convOutputs, predictions) = gradModel(inputs)
				loss=predictions[component][:, self.character_Idx[component]]
			# use automatic differentiation to compute the gradients
			grads = tape.gradient(loss, convOutputs)
			# compute the guided gradients
			castConvOutputs = tf.cast(convOutputs > 0, "float32")
			castGrads = tf.cast(grads > 0, "float32")
			guidedGrads = castConvOutputs * castGrads * grads
			# the convolution and guided gradients have a batch dimension
			# (which we don't need) so let's grab the volume itself and
			# discard the batch
			convOutputs = convOutputs[0]
			guidedGrads = guidedGrads[0]
			# compute the average of the gradient values, and using them
			# as weights, compute the ponderation of the filters with
			# respect to the weights
			weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
			cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
			# grab the spatial dimensions of the input image and resize
			# the output class activation map to match the input image
			# dimensions
			(w, h) = (image.shape[2], image.shape[1])
			heatmap = cv2.resize(cam.numpy(), (w, h))
			# normalize the heatmap such that all values lie in the range
			# [0, 1], scale the resulting values to the range [0, 255],
			# and then convert to an unsigned 8-bit integer
			numer = heatmap - np.min(heatmap)
			denom = (heatmap.max() - heatmap.min()) + eps
			heatmap = numer / denom
			heatmap = (heatmap * 255).astype("uint8")
      
			heatmap_list.append(heatmap)
		return heatmap_list