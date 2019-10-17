import pandas as pd
import os

class Answer:
	# Constructor
	def __init__(self, index, path):
		self.__index = index
		self.__answer = ""
		self.__file_name = path + "answer.csv"

	# Properties 
	@property
	def answer(self):
		return self.__answer

	@property
	def _file_name(self):
		return self.__file_name


	# Methods/Functions
	def load_answer(self):
		## Load file and then update _answer based on the given index
		if os.path.exists(self.__file_name):
			data = pd.read_csv(self.__file_name)
			if (self.__index <= len(data)-1 and self.__index >= 0):
				self.__answer = data[data.index == self.__index].values[0][0]
			else:
				print("There is no answer for this question")
		else:
			print("The answer file doesn't exist")



	
	