from Bot.bot import Bot
from sklearn.naive_bayes import GaussianNB

def main():
	path = './data/'
	classifier = GaussianNB()
	bot = Bot(path, classifier)
	bot.train_question_model()

	user_input = input()
	while user_input != 'exit':
		bot.process_question(user_input)

		user_input = input()




if __name__ == "__main__":
	main()