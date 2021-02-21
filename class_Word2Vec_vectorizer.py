import numpy as np

class Word2Vec_vectorizer:

	''' Transform text to a vector of a word embedding model.
	    It returns the average of the word embedding vectors.
	'''
	def __init__(self, model):
		self.word_vectors = model

	def fit(self, data, y = None):
		pass

	def transform(self, data, y = None):
		# dimensionality of word_vectors
		v = self.word_vectors.get_vector('queen')
		self.D = v.shape[0]

		X = np.zeros((len(data), self.D))
		n = 0
		emptycount = 0
		for judgement_text in data:
			tokens = judgement_text.split() # tokenise 
			vecs = []
			m = 0
			for word in tokens:
				try:
					vec = self.word_vectors.get_vector(word) #replace word by a vector
					vecs.append(vec)
					m += 1
				except KeyError:
					pass
			if len(vecs) > 0:
				vecs = np.array(vecs)
				X[n] = vecs.mean(axis = 0) # averaging all the vectors in data 
			else:
				emptycount += 1
			n += 1
		#print(" Number of samples with no words found: %s / %s" % (emptycount, len(data)))

		return X

	def fit_transform(self, data, y = None ):
		self.fit(data)

		return self.transform(data)

