"""
Combines Excercise 1 and 2 
"""
# Author: Fredrik Hoftun

# Country codes from languages/fetch_data.py
pages = {
    'ar': 'http://ar.wikipedia.org/wiki/%D9%88%D9%8A%D9%83%D9%8A%D8%A8%D9%8A%D8%AF%D9%8A%D8%A7',   # noqa: E501
    'de': 'http://de.wikipedia.org/wiki/Wikipedia',
    'en': 'https://en.wikipedia.org/wiki/Wikipedia',
    'es': 'http://es.wikipedia.org/wiki/Wikipedia',
    'fr': 'http://fr.wikipedia.org/wiki/Wikip%C3%A9dia',
    'it': 'http://it.wikipedia.org/wiki/Wikipedia',
    'ja': 'http://ja.wikipedia.org/wiki/Wikipedia',
    'nl': 'http://nl.wikipedia.org/wiki/Wikipedia',
    'pl': 'http://pl.wikipedia.org/wiki/Wikipedia',
    'pt': 'http://pt.wikipedia.org/wiki/Wikip%C3%A9dia',
    'ru': 'http://ru.wikipedia.org/wiki/%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F',  # noqa: E501
#    u'zh': u'http://zh.wikipedia.org/wiki/Wikipedia',
}


# Note that this is a poor predictor of language,
# unless you use many of the words given in the articles above.
# Accuracy should increase with longer texts.
# The polarity predictor is trained on movie reviews.


import sys
from joblib import load

ex1_cfs = load("ex1.joblib")
ex2_cfs = load("ex2.joblib")

text = [sys.argv[1]]


# converts the country category e.g. [1] into integer from the pages.keys() [1] -> "ar"
# list(pages.keys()) creates a new list thats easily iterable
language = list(pages.keys())[
    int(ex1_cfs.predict(text))]
print(f"The predicted language is: {language}")

if language == "en":
    if int(ex2_cfs.predict(text)) == 1:
        print("Predicted polarity: Positive")
    else:
        print("Predicted polarity: Negative")
        
    print(f"Prediction confidence fraction [Negative, Positive]: {ex2_cfs.predict_proba(text)}")
