"""
the abstract class tensorflow Net
"""

class Net(object):
	def __init__(self):
		raise ValueError("__init__ not implenmented")

	def feedforward(self, inputs):
		raise ValueError("feedforward not implenmented")

	
	def get_loss(self, logits, labels):
		raise ValueError("get_loss not implenmented")

	def get_train_op(self, loss):
		raise ValueError("get_train_op not implenmented")

	def save_ckpt(self, path):
		raise ValueError("save_ckpt not implenmented")

	def restore_ckpt(self, path):
		raise ValueError("restore_ckpt not implemented")

