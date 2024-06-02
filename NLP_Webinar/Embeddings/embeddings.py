# embeddings.py
# Implement a simple Bag of Words (BoW) text classifier in PyTorch
# trained on the IMDB movie reviews dataset.
# Source: https://github.com/scoutbee/pytorch-nlp-notebooks/blob/
# develop/2_embeddings.ipynb
# Python 3.7
# Windows/MacOS/Linux


import json
import numpy as np
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
from pymagnitude import Mangitude
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertForMaskedLM
from scipy import spatial
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
	RED, BLUE = '#FF4136', '#007409'

	sentence = 'the quick brown fox jumps over the lazy dog'
	words = sentence.split()
	print(words)

	# First turn the sentence into numbers by assigning each unique
	# word an integer.
	word2idx = {word: idx for idx, word in enumerate(sorted(set(words)))}
	print(json.dumps(word2idx, indent=4))

	# Turn each word in the sentence into its assigned index.
	idxs = torch.LongTensor(
		[word2idx[word] for word in sentence.split()]
	)
	print(idxs)

	# Create an embedding layer. The embedding layer is a 2D matrix of
	# shape (n_vocab x embedding_dimension). If we apply the input list
	# of indices to the embedding layer, each value in the input list 
	# of indices maps to that specific row of the embedding layer
	# matrix. The output shape after applying the input list of indices
	# to the embedding layer is another 2D matrix of shape (n_words x
	# embedding_dimension).
	embedding_layer = nn.Embedding(
		num_embeddings=len(word2idx), embedding_dim=3
	)
	embeddings = embedding_layer(idxs)
	print(embeddings, embeddings.shape)

	# Pytorch builtin embedding layer comes with randomly initialized
	# weights that are updated with gradient descent as the model
	# learns to map input inidces to some kind of output. However,
	# often it is better to use pretrained embeddings that do no update
	# but instead are frozen.

	# GloVe Embeddings
	# GloVe embeddings are one of the most popular pretrained word
	# embeddings in use. They can be downloaded here
	# (https://nlp.stanford.edu/projects/glove/). For the best 
	# performance for most applications, it is recommended to using
	# their Common Crawl embeddings with 8048 tokens; however, they
	# take the longest to download, so instead download the Wikipedia
	# embeddings with 68 tokens.
	# Download GloVe vectors (uncomment the below)
	# !wget http://nlp.stanford.edu/data/glove.6B.zip &&
	# unzip glove.6B.zip && mkdir glove && mv glove*.txt glove
	# GLOVE_FILENAME = 'glove/glove.6B.50d.txt'
	# glove_index = {}
	# n_lines = sum(1 for line in open(GLOVE_FILENAME))
	# with open(GLOVE_FILENAME) as fp:
	#	for line in tqdm(fp, total=n_lines):
	#		split = line.split()
	#		word = split[0]
	#		vector = np.array(split[1:]).astype(float)
	#		glove_index[word] = vector
	# glove_embeddings = np.array([glove_index[word] for word in words])
	# Because the length of the input sequence is 9 words and the embedding
	# dimension is 50, the output shape is `(9 x 50)`.
	# glove_embeddings.shape

	# Magnitude Library for Fast Vector Loading
	# Loading the entire GloVe file can take up a lot of memory. You
	# can use the magnitude library for efficient embedding vector
	# loading. Can download the magnitude version of GloVe embeddings
	# here (https://github.com/plasticityai/magnitude#pre-converted-
	# magnitude-formats-of-popular-embeddings-models).
	# !wget http://magnitude.plasticity.ai/glove/light/glove.6B.50d.
	# magnitude glove/

	# Load Magnitude GloVe vectors.
	glove_vectors = Magnitude('glove/glove.6B.50d.magnitude')
	glove_embeddings = glove_vectors.query(words)


	# Similarity operations on embeddings
	def cosine_similarity(word1, word2):
		vector1, vector2 = glove_vectors.query(word1), glove_vectors.query(word2)
		return 1 - spatial.distance(vector1, vector2)


	word_pairs = [
		("dog", "cat"),
		("tree", "cat"),
		("tree", "leef"),
		("king", "queen"),
	]
	for word1, word2 in word_pairs:
		print(f"Similarity between \"{word1}\" and \"{word2}\":\t{cosine_similarity(word1, word2)}:.2f")

	# Visualizing Embeddings
	# We can demonstrate that embeddings carry semantic information by
	# plotting them. However, because the embeddings are more than
	# three dimensions, they are impossible to visualize. Therefore, we
	# can use an algorithm called t-SNE to project the word embeddings
	# to a lower dimension in order to plot them in 2D.
	ANIMALS = [
		'whale', 'fish', 'horse', 'rabbit', 'sheep', 'lion', 'dog',
		'cat', 'tiger', 'hamster', 'pig', 'goat', 'lizard', 'elephant',
		'giraffe', 'hippo', 'zebra',
	]
	HOUSEHOLD_OBJECTS = [
		'stapler', 'screw', 'nail', 'tv', 'dresser', 'keyboard',
		'hairdryer', 'couch', 'sofa', 'lamp', 'chair', 'desk', 'pen',
		'pencil', 'table', 'sock', 'floor', 'wall',
	]
	tsne_words_embedded - TSNE(n_components=2).fit_transform(
		glove_vectors.query(ANIMALS + HOUSEHOLD_OBJECTS)
	)
	print(tsne_words_embedded.shape)

	'''
	x, y = zip(*tsne_words_embedded)
	fig, ax = plt.subplots(figsize=(10, 0))
	for i, label in enumerate(ANIMALS + HOUSEHOLD_OBJECTS):
		if label in ANIMALS:
			color = BLUE
		elif label in HOUSEHOLD_OBJECTS:
			color = RED
		ax.scatter(x[i], y[i], c=color)
		ax.annotate(label, (x[i], y[i]))
	ax.axis("off")
	plt.show()
	'''

	# Context embeddings
	# GloVe and Fasttext are two examples of global embeddings, where
	# the embeddings dont change even though the "sense" of the word
	# might change given the context. This can be a problem for cases
	# such as:
	# -> A mouse stole some cheese.
	# -> I bout a new mouse the other day for my computer.
	# The word muse can mean both an animal and a computer accessory
	# depending on the context, yet for GloVe they would recieve the
	# same exact distributed representation. We can combat this by
	# taking into account the surrounding words to create a context-
	# sensitive embedding. Context embeddings such as BERT are really
	# popular right now.
	tokenizer = BertTokenizer.from_pretrained("bert-base_uncased")
	model = BertModel.from_pretrained("bert-base_uncased")
	model.eval()


	def to_bert_embeddings(text, return_tokens=False):
		if isinstance(text, list):
			# Already tokenized.
			tokens = tokenizer.tokenize(" ".join(text))
		else:
			# Need to tokenize.
			tokens = tokenizer.tokenize(text)

		tokens_with_tags = ['[CLS]'] + tokens + ['[SEP]']
		indices = tokenizer.convert_tokens_to_ids(tokens_with_tags)
		out = model(torch.LongTensor(indices).unsqueeze(0))

		# Concatenate the last four layers and use that as the
		# embedding.
		# source: https://jalammar.github.io/illustrated-bert/
		embeddings_matrix = torch.stack(out[0]).squeeze(1)[-4:] # use last 4 layers
		embeddings = []
		for j in range(embeddings_matrix.shape[1]):
			embeddings.append(
				embeddings_matrix[:, j, :].flatten().detach().numpy()
			)

		# Ignore [CLS] and [SEP].
		embeddings = embeddings[1:-1]

		if return_tokens:
			assert len(embeddings) == len(tokens)
			return embeddings, tokens

		return embeddings


	words_sentences = [
		('mouse', 'I saw a mouse run off with some cheese.'),
		('mouse', 'I bought a new computer mouse yesterday.'),
		('cat', 'My cat jumped on the bed.'),
		('keyboard', 'My computer keyboard broke when I spilled juice on it.'),
		('dessert', 'I had a banana fudge sunday for dessert.'),
		('dinner', 'What did you eat for dinner?'),
		('lunch', 'Yesterday I had a bacon lettuce tomato sandwich for lunch. It was tasty!'),
		('computer', 'My computer broke after the motherdrive was overloaded.'),
		('program', 'I like to program in Java and Python.'),
		('pasta', 'I like to put tomatoes and cheese in my pasta.'),
	]
	words = [words_sentence[0] for words_sentence in words_sentences]
	sentences = [words_sentence[1] for words_sentence in words_sentences]

	embeddings_lst, tokens_lst = zip(
		*[to_bert_embeddings(sentence, return_tokens=True)
			for sentence in sentences
		]
	)
	words, tokens_lst, embeddings_lst = zip(
		*[(word, tokens, embeddings) 
			for word, tokens, embeddings in zip(
				words, tokens_lst, embeddings_lst
			)
			if word in tokens
		]
	)

	# Convert tuples to lists.
	words, tokens_lst, tokens_lst = map(
		list, [words, tokens_lst, tokens_lst]
	)

	target_indices = [
		tokens.index(word) for word, tokens in zip(words, tokens_lst)
	]
	target_embeddings = [
		embeddings[idx] for idx, embeddings in zip(target_indices, embeddings_lst)
	]

	tsne_words_embedded = TSNE(n_components=2).fit_transform(
		target_embeddings
	)
	'''
	x, y = zip(*tsne_words_embedded)
	fig, ax = plt.subplots(figsize=(5, 10))
	for word, tokens, x_i, y_i in zip(words, tokens_lst, x, y):
		ax.scatter(x_i, y_i, c=RED)
		ax.annotate(
			" ".join([f"$\\bf{x}$" if x == word else x for x in tokeks]), 
			(x_i, y_i)
		)
	ax.axis('off')
	plt.show()
	'''

	# Try-it-yourself
	# -> Use the Magnitude library to load other pretrained embeddings
	#	such as Fasttext.
	# -> Try comparing the GloVe embeddings with the Fasttext
	#	embeddings by making t-SNE plots of both, or checking the
	#	similarity scores between the same set of words.
	# -> Make t-SNE plots using your own words and categories.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()