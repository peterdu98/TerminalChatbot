from .answer import Answer
import pandas as pd
import os

class Question:
	# Constructor
	def __init__(self, index, summary):
		self.__index 		= index
		self.__summary		= summary
		self.__quest 		= ""
		self.__answer 		= ""

	# Properties (decorator)
	@property
	def quest(self):
		return self.__quest
	
	@property
	def index(self):
		return self.__index

	@property
	def answer(self):
		return self.__answer
	
	@property
	def summary(self):
		return self.__summary
	

	# Methods/Functions
	def load_question(self, path):
		## Load question and then find choices for that question
		file_name = path + 'question.csv'

		if os.path.exists(file_name):
			data = pd.read_csv(file_name)
			if (self.__index <= len(data)-1 and self.__index >= 0):
				self.__quest = data[data.index == self.__index].values[0][0]
				
				self.__find_answer(path)
		else:
			print("The question file doesn't exist")

	def __find_answer(self, path):
		## Find choices for a particular question based on summary
		choice = Answer(self.__index, path)
		choice.load_answer()
		self.__answer = choice.answer
	
