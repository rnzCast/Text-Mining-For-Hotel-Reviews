from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.probability import FreqDist
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from textblob import TextBlob, Word
from sklearn.pipeline import Pipeline
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re


class CleaningDF:

    def __init__(self, p_df):
        self.p_df = p_df

    def get_info(self):
        return self.p_df.info()

    # cristina. deletes unnecessary features
    def drop_columns(self):
        dropcols = ['id', 'dateadded', 'dateupdated', 'address', 'categories', 'primarycategories', 'keys', 'latitude',
                    'longitude', 'postalcode', 'reviews_date', 'reviews_dateseen', 'reviews_sourceurls',
                    'reviews_usercity', 'reviews_userprovince', 'reviews_username', 'sourceurls', 'websites', 'location']

        self.p_df = self.p_df.drop(dropcols, axis=1)
        return self.p_df


    def missing_val(self):
        self.p_df.isnull().values.any()                                              # cristina
        self.p_df["reviews_text"].isna().sum()                                       # Kevin
        self.p_df["reviews_title"].notnull().isna().sum()                            # Kevin
        self.p_df['reviews_text'] = self.p_df['reviews_text'].str.replace('\d+', '') #cristina. delete numbers
        self.p_df['reviews_text'] = self.p_df['reviews_text'].dropna().reset_index(
            drop=True)  # delete NaN and reindex   #cristina
        self.p_df['reviews_text'] = self.p_df['reviews_text'].astype(str)  # Cristina. To assure all are strings.
        return self.p_df


class PreprocessReview:

    def __init__(self, pr_df):
        self.pr_df = pr_df


    # cristina. Remove Most frequent words
    def common_words(self, wfilter, n_words):

        self.filter = wfilter
        self.n_words = n_words
        all_words = ' '.join([text for text in wfilter])
        all_words = all_words.split()

        #get word frequency
        fdist = FreqDist(all_words)
        words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())}) #converts to df

        # selecting top #terms most frequent words and plot
        d = words_df.nlargest(columns="count", n=self.n_words)
        # plt.figure(figsize=(20, 5))
        # ax = sns.barplot(data=d, x="word", y="count")
        # ax.set(ylabel='Count')
        # plt.show()
        return d


    #1. Kevin. convert to lower case, remove punctuation
    def clean_split_text(self):  # convert to lower case, remove punctuation, tokenize
        self.pr_df["reviews_text"] = self.pr_df['reviews_text'].apply(
            lambda x: " ".join(x.lower() for x in str(x).split()))  # lower case
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].str.replace('[^\w\s]', "")  # puntuation
        self.pr_df['reviews_text_token'] = self.pr_df.apply(lambda row: nltk.word_tokenize(row['reviews_text']),
                                                            axis=1)  # tokenization #cristina
        return self.pr_df


    # renzo. Remove stop words and reassign to the same column
    def remove_stop_w(self):
        stop_words = stopwords.words('english') #cristina.
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
        #Cristina. Get ride of numbers
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].str.replace('\d+', '')
        self.pr_df['reviews_text'] = self.pr_df['reviews_text'].dropna().reset_index(drop=True) # Cristina: delete NaN and reindex

        return self.pr_df


    # renzo. Count the lest frequent words
    def count_rare_word(self):
        freq = pd.Series(" ".join(self.pr_df['reviews_text']).split()).value_counts()[-15:]
        # freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
        return freq

    # renzo . Remove rare words
    def remove_rare_words(self):
        freq = pd.Series(" ".join(self.pr_df['reviews_text']).split()).value_counts()[-15:]
        freq = list(freq.index)
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        self.pr_df['reviews_text'] = self.pr_df['reviews_text'].dropna().reset_index(drop=True) # Renzo: delete NaN and reindex

        return self.pr_df


    # cristina. tokenization - separates words
    def tokenization(self):
        self.pr_df['reviews_text_token'] = self.pr_df.apply(lambda row: nltk.word_tokenize(row['reviews_text']), axis=1)
        return self.pr_df


    #Renzo. converts the word into its root word
    def lematization(self):
        self.pr_df["reviews_text_lematized"] = self.pr_df["reviews_text"]
        self.pr_df["reviews_text_lematized"] = self.pr_df["reviews_text_lematized"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        return self.pr_df


    def stemming(self):
        st = PorterStemmer()
        self.pr_df["reviews_text_lematized"] = self.pr_df["reviews_text_lematized"].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
        return self.pr_df


    #Renzo. Spelling correction
    def spelling_correction(self):
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(lambda x: str(TextBlob(x).correct()))
        return self.pr_df

    # Kevin
    def cont_neg_feel(self):
        # Number of Words (the negative sentiments contain
        # a lesser amount of words than the positive ones.)
        self.pr_df["wordcount_reviews.text"] = self.pr_df["reviews_text"].apply(lambda x: len(str(x).split(" ")))
        self.pr_df["wordcount_reviews.title"] = self.pr_df["reviews_title"].apply(lambda x: len(str(x).split(" ")))

    # Kevin
    def count_chr(self):
        # Number of characters (includes spaces)
        self.pr_df["charcount_reviews.text"] = self.pr_df["reviews_text"].str.len()
        self.pr_df["charcount_reviews.title"] = self.pr_df["reviews_title"].str.len()

    # Kevin
    def rating_negative_Moderate_positive(self):
        #Kevin. scale convert rating into negative(x >= 0,x < 3),moderate(x >= 3 and x <= 4), positive(x<4 and x<5)
        self.pr_df["New_reviews_rating_NMP"] = self.pr_df["reviews_rating"].apply(
            lambda x: 1 if x >= 0 and x < 3 else (2 if x >= 3 and x <=4 else 3))
        return self.pr_df

    def rating_Negative_Positive(self):
        #Kevin. scale convert rating into negative(x >= 0,x =< 3), positive (x > 3 and x<=5)
        self.pr_df["New_reviews_rating_NP"] = self.pr_df["reviews_rating"].apply(
            lambda x: 1 if x >= 0 and x <= 3 else 2)
        return self.pr_df

    # def avg_word(self, reviews): # Average Word Length
    # ords = str(reviews).split()
    # return (sum(len(word) for word in words) / len(words))
    # self.pr_df["avgword_reviews.text"] = self.pr_df["reviews.text"].apply(lambda x: avg_word(x))
    # self.pr_df["avgword_reviews.title"] = self.pr_df["reviews.title"].apply(lambda x: avg_word(x))



class Predictors:
    def __init__(self, f_df):
        self.f_df = f_df

    #Cristina. Model and evaluation
    def naivesb(self):

        X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'],
                                                            self.f_df['reviews_rating'].astype('int'),
                                                            test_size=0.25, random_state=85)
        count_vectorizer = CountVectorizer()  # converts to bags of words and it would remove stop words

        count_train = count_vectorizer.fit_transform(X_train.values)
        count_test = count_vectorizer.transform(X_test.values)

        nb_classifier = MultinomialNB()
        nb_classifier.fit(count_train, y_train)
        pred = nb_classifier.predict(count_test)
        print(metrics.confusion_matrix(y_test, pred, labels=[1, 2, 3, 4, 5]))

        counter = 0
        for review, category in zip(X_test, pred):
            print('%r => %s' % (category, review))
            if (counter == 5):
                break
            counter += 1
        print("Accuracy score Naives Bayes: " + str(metrics.accuracy_score(y_test, pred)))
        return metrics.accuracy_score(y_test, pred)

    #Kevin. SVM with rbf, including the compare with scale
    def svm_apply(self):
        X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'], self.f_df['reviews_rating'].astype('int'), test_size=0.25, random_state=85)
        count_vectorizer = CountVectorizer()
        count_train = count_vectorizer.fit_transform(X_train.values)
        count_test = count_vectorizer.transform(X_test.values)
        SVM = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
        SVM.fit(count_train, y_train)
        predictions_SVM = SVM.predict(count_test)
        return "Accuracy score SVM without scale: "+str(metrics.accuracy_score(y_test, predictions_SVM))

    def svm_apply_1(self):
        X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'], self.f_df["New_reviews_rating_NMP"].astype('int'), test_size=0.25, random_state=85)
        count_vectorizer = CountVectorizer()
        count_train = count_vectorizer.fit_transform(X_train.values)
        count_test = count_vectorizer.transform(X_test.values)
        SVM = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
        SVM.fit(count_train, y_train)
        predictions_SVM = SVM.predict(count_test)
        return "Accuracy score SVM with NMP Scale: "+str(metrics.accuracy_score(y_test, predictions_SVM))

    def svm_apply_2(self):
        X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'], self.f_df["New_reviews_rating_NP"].astype('int'), test_size=0.25, random_state=85)
        count_vectorizer = CountVectorizer()
        count_train = count_vectorizer.fit_transform(X_train.values)
        count_test = count_vectorizer.transform(X_test.values)
        SVM = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
        SVM.fit(count_train, y_train)
        predictions_SVM = SVM.predict(count_test)
        return "Accuracy score SVM with NP Scale:"+str(metrics.accuracy_score(y_test, predictions_SVM))

    #Renzo
    def linearsvc(self):
        self.f_df['reviews_rating'] = self.f_df['reviews_rating'].round()

        X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'], self.f_df['reviews_rating'].astype('int'), test_size = 0.25, random_state=53)

        pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=(1,3), stop_words=None, sublinear_tf=True)),
            ('chi', SelectKBest(chi2, k=8000)),
            ('clf', LinearSVC(C=1.0, penalty='l1',max_iter=3000, dual=False))
                            ])

        model = pipeline.fit(X_train, y_train)

        vectorizer = model.named_steps['vect']
        chi = model.named_steps['chi']
        clf = model.named_steps['clf']

        feature_names = vectorizer.get_feature_names()
        feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
        feature_names = np.asarray(feature_names)

        target_names = ['1', '2', '3', '4', '5']
        print("The top keywords are: ")

        for i, label in enumerate(target_names):
            top10 = np.argsort(clf.coef_[i])[-10:]
            print("%s: %s" %(label, " ".join(feature_names[top10])))

        print("Accuracy score: " + str(model.score(X_test, y_test)))
        print("the score of manual input is:", model.predict(["I love this palace, is amazing"]))

        return model.score(X_test, y_test)

def main():

    # 1. UPLOAD THE FILE
    df_file = pd.read_csv('datafiniti_hotel_reviews.csv')
    a = df_file.info()


    # 2. CLEANING PROCESS
    clean = CleaningDF(df_file)                           # Cristina. instance class CleaningDF()
    df = clean.drop_columns()                             # Cristina. Drop features that are not necessary for the analysis
    df = clean.missing_val()                              # Cristina - Kevin. Verify and clean missing values and converts to string reviews_text


    # 3. PREPROCESSING
    clean_text = PreprocessReview(df)                     # Cristina. instance class PreprocessReview()
    clean_text.common_words(df['reviews_text'],25)        # Cristina. Shows the frequency of stop words BEFORE removing
    df = clean_text.clean_split_text()                    # Cristina - Kevin. Converts to lower case, removes punctuation.
    df = clean_text.remove_stop_w()                       # Renzo. It removes stop words from reviews_text
    cwc = clean_text.count_rare_word()                    # Renzo. Count the rare words in the reviews. I tried with :10, then with :-20
    df = clean_text.remove_rare_words()                   # Renzo. This will clean the rare words from the reviews column
    clean_text.common_words(df['reviews_text'], 25)       # Cristina. Shows the frequency of stop words AFTER removing
    #df = clean_text.spelling_correction()                # Renzo. This will do the spelling correction. SLOW PROCESS
    df = clean_text.tokenization()                        # Cristina. Tokenization: Convert to strings
    df = clean_text.lematization()                        # Renzo. Converts the word into its root word
    df = clean_text.stemming()                            # Renzo. This will do the stemming process
    print(df[['reviews_text_lematized']])
    print(df[['reviews_text_token']])


    lsvc = Predictors(df)
    lsvc.linearsvc()

    predictor = Predictors(df)                          # Cristina. instance class Predictors()
    prediction_NB = predictor.naivesb()                 # Cristina. calls model naives bayes
    print(prediction_NB)
    prediction_rbf = predictor.svm_apply()              # Kevin. calls SVM without scale of rating
    print(prediction_rbf)

    # look for the improvement by doing the re-scale
    df = clean_text.rating_negative_Moderate_positive()  # Kevin. scaled the rating to negative_Moderate_positive
    df = clean_text.rating_Negative_Positive()          # Kevin. scaled the rating to negative_positive
    prediction_rbf_1 = predictor.svm_apply_1()          # Kevin. calls SVM, with negative_Moderate_positive rating with 0.4808 accurancy
    print(prediction_rbf_1)
    prediction_rbf_2 = predictor.svm_apply_2()          # Kevin. calls SVM, with negative_positive rating with 0.7224 accurancy
    print(prediction_rbf_2)

if __name__ == "__main__": # "Executed when invoked directly"
    main()


