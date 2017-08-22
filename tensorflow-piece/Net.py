"""
the abstract class tensorflow Net
"""

class Net(object):
	def __init__(self, data_shape, data_dtype, label_shape, label_dtype):
		self.data_dtype = data_dtype
		self.data_shape = data_shape
		self.label_dtype = label_dtype
		self.label_shape = label_shape
		self.all_variables_have_been_loaded = False
		self.train_op = None
		self.inputs = None
		self.labels = None

		raise ValueError("__init__ not implenmented")

	def feedforward(self, inputs=None):
		"""
		if using the feed_dict method to feed the data ,  inputs=None, 
		else just pass the input tensor to inputs
		this method return the logits
		"""
		if inputs is None:
			self.inputs = tf.placeholder(dtype=self.data_dtype, shape=self.data_shape, name='inputs')
		else:
			self.inputs = inputs
		raise ValueError("feedforward not implenmented")

	
	def get_loss(self, logits, labels=None):
		"""
		if using the feed_dict method to feed the label ,  labels=None, 
		else just pass the input tensor to inputs
		this method return the loss
		"""
		if labels is None:
			self.labels = tf.placeholder(dtype=self.label_dtype, shape=self.label_shape, name='labels')
		else:
			self.labels = labels
		raise ValueError("get_loss not implenmented")

	def set_train_op(self, loss, regulizer=True):
		"""
		fill the self.train_op value
		this method return nothing
		"""
		raise ValueError("get_train_op not implenmented")

	def model_ready_check():
		"""check the model whether ready to use for traning or inference"""
		raise ValueError("model_ready_check not implented")


	def train_one_step(self, inputs=None, labels=None):
		raise ValueError("train_one_step not implented")		

	def train_n_iteration(self, inputs=None, labels=None):
		raise ValueError("train_n_iteration not implented")


	def inference(self, inputs):
		raise ValueError("inference not implented")		

	@staticmethod
	def save_ckpt(self, path):
		raise ValueError("save_ckpt not implenmented")
		
	@staticmethod
	def restore_ckpt(self, path):
		raise ValueError("restore_ckpt not implemented")

