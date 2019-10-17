import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class Model:
	# Constructor
	def __init__(self, data, classifier):
		self.__data 		 = data
		self.__user_input	 = []
		self.__corpus 		 = []
		self.__summary 		 = []
		self.__vocab		 = dict()
		self.__vocab_summary = dict()
		self.__cv 			 = CountVectorizer(max_features = 15000)
		self.__classifier	 = classifier	

	# Properties
	@property
	def data(self):
		return self.__data

	@property
	def user_input(self):
		return self.__user_input

	@property
	def corpus(self):
		return self.__corpus
	
	@property
	def summary(self):
		return self.__summary
	

	@property
	def vocab(self):
		return self.__vocab

	@property
	def vocab_summary(self):
		'''
			A vocabulary dicionary of all tags (unique)
			{"Place":  1, "Time": 2, "Building": 3, "Number": 4, "Station": 5, "Other": 6}
		'''
		return self.__vocab_summary

	@property
	def classifier(self):
		return self.__classifier
	
	#Function/Methods
	def train(self):
		'''
			Aim: Train the prediction model based on the given classifier
			Steps:
				1. Processing data
				2. Train model
					2.1. Convert all corpus to count vector (corpus is a list of question after processing) as X-axis
						['ta buildings', 'tb building', 'ta tb building'] --> [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
					
					2.2. Labeling each row as y-axis
						[1, 1, 2, 3, 3, ...]

					2.3. Split data to train and test set
					2.4. Fit train data to the given classifier
		'''
		self.__process_data()

		## Model
		X = self.__cv.fit_transform(self.__corpus).toarray()
		y = np.array([s[1] for s in self.__summary])
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
		self.__classifier.fit(X_train, y_train)

	def  add_user_input(self, user_input):
		'''
			Aim: Add user input into a list and process it
		'''
		self.__user_input.append(user_input)
		self.__process_user_input()	
		
	def predict(self):
		'''
			Aim: Using a trained model to predict the label(tag) of user input
			Steps:
				1. Check whether there is an existing user input of not
				2. If there is no user input after processing
					2.1. Return an approriate statement
				3. If there is an user input after processing
					3.1. Combine an existing corpus and user input into a local corpus
					3.2. Convert all corpust o count vector as X_pred
					3.3. Using a trained model to predict a local corpus and take the last one as y_pred
						 ['ta buildings'] ->  [1] | ['studentHQ']  -> [1]

					3.4. Based on y_pred, generating an appropriate tag from a vocabulary dictionary of tags
						 [1] -> "Place"  | [2] -> "Time"

					3.5. Clean user input
		'''
		result = ""
		if not self.__user_input:
			result = "Sorry! I don't understand your question."
		else:
			corpus = self.__corpus + self.__user_input

			X_pred = self.__cv.fit_transform(corpus).toarray()
			y_pred = self.__classifier.predict(X_pred[-1].reshape(1, -1))

			for vs in self.__vocab_summary.items():
				if vs[1] == y_pred[0]:
					result = vs[0]

			self.__user_input.pop()
		return result

	def __process_data(self):
		'''
			Aim: A wrapper of processing questions and summary tags for training the prediction model
		'''
		self.__process_question()
		self.__process_summary()

	def __process_question(self):
		'''
			Aim: Process all questions in a  dataset for training the prediction model
			Steps:
				1. Clean text in each question
					"What are TA-TB-TC-TD buildings known for?" -> ['ta', 'tb', 'tc', 'td', 'buildings', 'known']
					"How many floors does the library have?" 	-> ['many', 'floors', 'library']
					"Which is the quite floor in the library?"	-> ['quite', 'floor', 'library']
				
				2. Add each word of a question uniquely to a vocabulary dictionary of question
				3. Convert a list type to a string type
					['ta', 'tb', 'tc', 'td', 'buildings', 'known'] -> 'ta tb tc td buildings known'
					['many', 'floors', 'library']  				   -> 'many floors library'

				4. Add a question to a list of corpus
		'''
		for question in self.__data.Question:
			question = self.__clean_text(question)

			## Add each word uniquely to a vocabulary dictionary of question
			for word in question:
				if word not in self.__vocab:
					self.__vocab[word] = len(self.__vocab)+1

			question = ' '.join(question)
			self.__corpus.append(question)

	def __process_summary(self):
		'''
			Aim: Process all tags in a  dataset for training the prediction model
			Steps:
				1. Check whether word is already in a vocabulary dictionary of summary or not
				2. If not, add it to the dictionary with key = [word] and value = [unique id]
					["PLace",  "Place", "Time"]	-> {"Place": 1, "Time": 2}

				3. Add a list of word and unique to a list of summary
					["Place", "Place", "Time"]  ->  [["PLace", 1], ["Place",  1], ["Time", 2]]
		'''
		for word in self.__data.Summary:
			if word not in self.__vocab_summary:
				self.__vocab_summary[word] = len(self.__vocab_summary)+1
			self.__summary.append([word, self.__vocab_summary[word]])

	def __process_user_input(self):
		'''
			Aim: Process user input into an approriate input for predicting
			Steps:
				1. Clean text in user input
				2. Check whether user input contains an unapproriate word (not in a vocabulary dictionary of question) after cleaned
				3. If yes, remove an input, break the loop and toggle removed tag (True|False)
				4. If the input is not removed, covert it into string type
		'''
		removed = False
		for sentence in self.__user_input:
			clean_sentence = self.__clean_text(sentence)
			for word in clean_sentence:
				if word not in self.__vocab:
					self.__user_input.remove(sentence)
					removed = True
					break
			
			if not removed:
				self.__user_input = [' '.join(clean_sentence)]

	def __clean_text(self, data):
		'''
			Aim:  Clean text for training and predicting purposes
			Steps:
				1. Take an input and make sure it contains only word and number (using Regex)
					"What are TA-TB-TC-TD buildings known for?" -> "What are TA TB TC TD buildings known for"

				2. Lower all characters
					"What are TA TB TC TD buildings known for" 	-> "what are ta tb tc td buildings known for"
				
				3. Tokenize a sentence (it differs from split because it can take something like [n't == not])
					"what are ta tb tc td buildings known for" 	-> ["what", "are", "ta", "tb", "td", "buildings", "known", "for"]

				4. Import a list of all stop words
				5. Clean tokens with stop_words
					["what", "are", "ta", "tb", "td", "buildings", "known", "for"] -> ['ta', 'tb', 'tc', 'td', 'buildings', 'known']
		'''
		data = re.sub('[^a-zA-Z0-9]', ' ', data)
		data = data.lower()
		tokens = nltk.word_tokenize(data)
		stop_words = nltk.corpus.stopwords.words('english')
		data = [word for word in tokens if word not in stop_words]

		return data
	
	
	