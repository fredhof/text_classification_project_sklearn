# Simple text classification project in Scikit-learn

[Source data set, examples and exercises, from scikit-learn.org](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#exercise-3-cli-text-classification-utility)

## Exercise 1: Language identification

Divides text into a vector of strings of lenght 1 to 3, then uses single-layer perceptron to build a model of language identification.

Data set for training and testing: The website https://en.wikipedia.org/wiki/Wikipedia in various languages. 

Requires that the data set is downloaded. Can be done by changing directory to data/languages and running `fetch_data.py`.

Run as `python3 exercise_01_language_train_model.py data/languages/paragraphs`.

## Exercise 2: Polarity of movie reviews
Computes the polarity of movie reviews using Linear Support Vector Classification.
First it divides the reviews into a vector of words to perform classification, before doing a grid search with 5-fold cross-validation to determine the best configuration of the given parameters, also printing the top 3 configurations.

Data set for training and testing: `Polarity data` from http://www.cs.cornell.edu/people/pabo/movie-review-data/

Requires that the data set is downloaded. Can be done by changing directory to data/movie_reviews and running `fetch_data.py`.

Run as `python3 exercise_02_sentiment.py data/movie_reviews/txt_tokens`.

## Exercise 3: CLI text classification utility
Uses the models trained in exercise 1 and 2 to predict the language of some text, and then predicts the polarity of the text if it is in English.

Included pre-trained model files: `ex1.joblib, ex2.joblib`.

Run as `python3 exercise_03_CLI_utility.py "TEXT TO CLASSIFY HERE"`.

### Module requirements:
Scikit-learn + dependencies and joblib.