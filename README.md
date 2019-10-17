# TerminalChatbot

**Description**: TerminalChat is a chatbot application using natural language process (NLP) and a classification model 
to predict a question and generate an approriate answer.

**Dataframes Summary:**
1. Question
   1. ID - unique identity of each question
   2. Question - An original text of a question
   3. Summary - A tag to classify a question
   
2. Answer
   1. ID - unique indeity of each answer (corresponding to a particular question)
   2. Answer - An original text of a question
   
**Dependencies:** Pandas, Numpy, sklearn, nltk, re.

**How does the process of finding an answer from a question work?**:
1. Add user input to a question model
2. Predict tag based  on input
3. Filter question dataframe based on predicted tag
4. Find the best match by using Cosing Similiarity algorithms
5. Find the answer based on best-match index and a predicted tag

**How is data trained and fitted into the classification model?**
1. Clean and process text
2. Convert all corpus to count vector (corpus is a list of question after processing) as X-axis
3. Labeling each row as y-axis
4. Split data to train and test set
5. Fit train data to the given classifier

**How does the model predict random inputs?**
1. Check whether an input is valid to predict or not.
    1. IF invalid, it returns an approriate statement
    2. IF valid, it moves to step 2
2. Combine an existing corpus and user input into a "local" corpus
3. Convert all corpust o count vector as X_pred
4. Using a trained model to predict a "local" corpus and take the last one as y_pred
5. Based on y_pred, generating an appropriate tag from a vocabulary dictionary of tags
6. Clean user input


