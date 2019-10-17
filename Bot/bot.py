import pandas as pd
from .model import Model
from QuestionAnswer.question import Question
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Bot:
	# Constructor
	def __init__(self, path, classifier):
		self.__path = path
		self.__question = None
		self.__question_model = None
		self.__classifier = classifier


	# Properties
	@property
	def classifier(self):
		return self.__classifier

	# Functions/Methods
	def train_question_model(self):
		'''
			Aim: Run a training question model
			Steps:
				1. Read question dataset
				2. Call Model object, passing data and a wanted classifier
				3. Start training the model

		'''
		path_question = self.__path + 'question.csv'
		self.__question = pd.read_csv(path_question)
		self.__question_model = Model(self.__question, self.__classifier)

		print('Model is being trained!')
		self.__question_model.train()
		print('Model has been trained successfully!')

	def process_question(self, user_input):
		'''
			Aim: Predict user input and find an best question based on predicted tag
			Steps:
				1. Add user input to a question model
				2. Predict tag based  on input
				3. Filter question dataframe based on predicted tag
				4. Find the best match by using Cosing Similiarity algorithms
				5. Find the answer based on best-match index and a predicted tag
		'''
		self.__question_model.add_user_input(user_input)
		tag = self.__question_model.predict()

		if tag not in self.__question_model.vocab_summary:
			print(tag)
		else:
			## Filter question based on tag
			filtered_question = self.__question[self.__question.Summary == tag]

			## Find the best match
			max_similar = 0
			max_ind = 0
			for question in filtered_question.Question:
				similar = self.__get_cosine(user_input, question)[0][1]

				if max_similar < similar:
					max_similar = similar
					max_ind = filtered_question[filtered_question.Question == question].index[0]

			## Find choices with corresponding index
			self.__find_answer(max_ind, tag)


	def __find_answer(self, index, tag):
		'''
			Aim: Find the answer for a question with a given index and tag
			Steps:
				1. Create question object to load a question
				2. Load question
				3. Print out a title of question, its tag and  an answer
		'''
		question = Question(index, tag)
		question.load_question(self.__path)

		print("Question: {}".format(question.quest))
		print("Summary: {}".format(question.summary))
		print("Answer: {}".format(question.answer))

	def __get_cosine(self, *args):
		'''
			Aim: Get a cosine similiarity between 2 question
			Steps:
				1. Convert two texts to count vectors
				2. Compute a cosine similiarity matrices
		'''
		vectors = [t for t in self.__get_vectors(*args)]
		return cosine_similarity(vectors)

	def __get_vectors(self, *args):
		'''
			Aim: Compute a count vector of a given list of text
			Steps:
				1. Store each text into a list of text
				2. Transform the list to count vector
		'''
		text = [t for t in args]
		vectorizer = CountVectorizer(text)
		return vectorizer.fit_transform(text).toarray()
	
	