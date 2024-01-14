"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess whether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Skeleton code: Olivier Grisel <olivier.grisel@ensta.org>
# Author: Fredrik Hoftun
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = sys.argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent, see below and grid search

    # e.g. min_df = 5: needs 5 occurances of word
    # e.g. max_df = 0.9: cuts out 10% of the most frequent words
    clf = Pipeline([
        ('vect', TfidfVectorizer(analyzer="word")),
        ('clf', CalibratedClassifierCV(estimator=LinearSVC())) 
        ]) # applyu CCCV to use .predict_proba() method for confidence levels
    

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'vect__min_df': [2, 3, 4],
            'vect__max_df': [0.9, 0.95, 0.99]
        }
    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    gs_clf.fit(docs_train, y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    # printing best 3
    for idx,i in enumerate(gs_clf.cv_results_['rank_test_score']):
        if i<4:
            print(i, gs_clf.cv_results_['params'][idx],
            gs_clf.cv_results_['mean_test_score'][idx], 
            gs_clf.cv_results_['std_test_score'][idx])

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = gs_clf.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    # For task 3
    from joblib import dump
    dump(gs_clf, "ex2.joblib")