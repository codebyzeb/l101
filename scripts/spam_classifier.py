import sys, re
import fasttext
import numpy as np
from numpy.random import default_rng
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from itertools import chain
from gensim.models import KeyedVectors
import pickle
from sklearn import svm

from sklearn import linear_model

def get_emails(filename, verbose=False):

    """
    Extracts the text from the message body field of each email, not including embedded text
    and returns an array of emails, each containing an array of tokens present in the email.

    Also performs pre-processing, removing punctuation and tokens that are too long.
    """

    emails = []

    # Regex used to remove punctuation and words longer than 15 characters
    pre_process_regex = re.compile("(\\w|&)\\w{1,14}$")

    # Porter stemmer used for stemming
    ps = PorterStemmer()
    
    in_body = False
    in_normal = False
    email = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        i = 0
        while line:
            
            # Find UnicodeDecoding Errors
            try:
                line = fp.readline()
            except UnicodeDecodeError:
                if verbose:
                    print("UnicodeDecodeError at line",i,"for",filename)
            i+=1

            # Only parse normal text message bodies
            if "<MESSAGE_BODY>" in line:
                in_body = True
                email = []
            if "</MESSAGE_BODY>" in line:
                in_body = False
                # Pre-processing occurs here
                email = [token.lower() for token in email if pre_process_regex.match(token)]
                if email != []:
                    emails.append(email)
            if "<TEXT_NORMAL>" in line:
                in_normal = True
            if "</TEXT_NORMAL>" in line:
                in_normal = False
            
            if in_body and in_normal and line[0] == "^": # dataset is already sentence-split
                sentence = line.split(" ") # dataset is already tokenized
                email.extend(sentence[1:-1]) # remove start-of-line character and end-of-line character

    emails = [" ".join(email) for email in emails] # return emails as space-separated string of tokens
    return emails

def get_data():
    """
    Prepare training and testing data by reading emails and
    shuffling the labels for both sets.
    """
    
    # Extract training email bodies from files
    train_gen = get_emails("train_GEN.ems")
    train_spam = get_emails("train_SPAM.ems")

    # Use 1/10th of the training data for validation
    valid_gen = train_gen[-int(len(train_gen)/10):]
    valid_spam = train_spam[-int(len(train_spam)/10):]
    print("Using {} genuine emails and {} spam emails from TRAINING set for validation".format(len(valid_gen), len(valid_spam)))

    # Validation set should never be part of training
    train_gen = train_gen[:-int(len(train_gen)/10)]
    train_spam = train_spam[:-int(len(train_spam)/10)]
    print("Using {} genuine emails and {} spam emails from TRAINING set for training".format(len(train_gen), len(train_spam)))

    # Extract testing email bodies from files
    test_gen = get_emails("adapt_GEN.ems")
    test_spam = get_emails("adapt_SPAM.ems")
    print("Using {} genuine emails and {} spam emails from TEST set for testing".format(len(test_gen), len(test_spam)))

    # Combine spam and ham into training and test sets
    train_size = len(train_gen) + len(train_spam)
    test_size = len(test_gen) + len(test_spam)
    valid_size = len(valid_gen) + len(valid_spam)
    X_train = np.array(train_gen + train_spam)
    X_test = np.array(test_gen + test_spam)
    X_valid = np.array(valid_gen + valid_spam)

    # Create labels for data
    y_train = np.array([0] * len(train_gen) + [1] * len(train_spam))
    y_test = np.array([0] * len(test_gen) + [1] * len(test_spam))
    y_valid = np.array([0] * len(valid_gen) + [1] * len(valid_spam))

    # Shuffle labels and data
    rng = default_rng(0)
    shuffle_train = rng.permutation(train_size)
    shuffle_test = rng.permutation(test_size)
    shuffle_valid = rng.permutation(valid_size)
    X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]
    X_valid, y_valid = X_valid[shuffle_valid], y_valid[shuffle_valid]
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid

def run_nb_classifier(X_train, y_train, X_test):
    """ Run a multinomial NB classifier on email data """

    # Override the tokenization of CountVectorizer to ensure the vocabulary is the same
    # as the other model
    vocabulary = np.unique(list(chain(*[sent.split(' ') for sent in X_train])))
    cv = CountVectorizer(vocabulary=vocabulary, tokenizer=lambda x: x.split(' '), )

    # Get emails as bag-of-word frequency vectors
    training_data = cv.fit_transform(X_train)
    print(len(cv.vocabulary_))
    testing_data = cv.transform(X_test)

    # Classify using Multinomial Naive Bayes
    y_predictions= []
    for i in range(10):
        clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        clf.fit(training_data, y_train)
        y_predictions.append(clf.predict(testing_data))
        
    return y_predictions

def report_stats(y_test, y_predictions):

    print("GEN PRECISION: ", np.mean([precision_score(y_test, y_predict, pos_label=0) for y_predict in y_predictions]))
    print("GEN RECALL ", np.mean([recall_score(y_test, y_predict, pos_label=0) for y_predict in y_predictions]))
    print("GEN F1 ", np.mean([f1_score(y_test, y_predict, pos_label=0) for y_predict in y_predictions]))
    print("SPAM PRECISION: ", np.mean([precision_score(y_test, y_predict, pos_label=1) for y_predict in y_predictions]))
    print("SPAM RECALL ", np.mean([recall_score(y_test, y_predict, pos_label=1) for y_predict in y_predictions]))
    print("SPAM F1 ", np.mean([f1_score(y_test, y_predict, pos_label=1) for y_predict in y_predictions]))

    print('Accuracy score:', np.mean([accuracy_score(y_test, y_predict) for y_predict in y_predictions]))
    print('Precision score:', np.mean([precision_score(y_test, y_predict) for y_predict in y_predictions]))
    print('Recall score:', np.mean([recall_score(y_test, y_predict) for y_predict in y_predictions]))
    print('F1 score:', np.mean([f1_score(y_test, y_predict) for y_predict in y_predictions]))

def prepare_data_fasttext(X_train, y_train, X_valid, y_valid):
    """ Write training data and validation data to files that fasttext can understand """

    with open("emails.train", 'w') as train_file:
        for i, email in enumerate(X_train):
            label = '__label__spam ' if y_train[i] == 1 else '__label__gen '
            train_file.write(label + email + '\n')

    with open("emails.valid", 'w') as valid_file:
        for i, email in enumerate(X_valid):
            label = '__label__spam ' if y_valid[i] == 1 else '__label__gen '
            valid_file.write(label + email + '\n')

def tune_fasttext(X_train, y_train, X_valid, y_valid):
    """ Train and run a fasttext classifier on email data """

    # Train supervised model using autotuning
    prepare_data_fasttext(X_train, y_train, X_valid, y_valid)
    best_f1 = 0
    best_parameter = 0
    for i in range(5, 10, 1):

        f1 = 0

        # Find average f1 over 20 runs
        for _ in range(20):
            model = fasttext.train_supervised(input="emails.train", epoch=8,verbose=0,ws=4,neg=6)
            
            # Get predictions for each label
            y_predict = []
            for email in X_valid:
                prediction, _ = model.predict(email)
                y_predict.append(1 if "__label__spam" in prediction else 0)

            f1 += f1_score(y_valid, y_predict)/20
            
        print("Parameter:",i,"-",f1)
        if f1 > best_f1:
            best_f1 = f1
            best_parameter = i

    print("Achieved F1 =",best_f1,"with parameter:",best_parameter)


def run_fasttext_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """ Train and run a fasttext classifier on email data """

    prepare_data_fasttext(X_train, y_train, X_valid, y_valid)

    y_predictions = []
    for i in range(10):
        # Train supervised model
        model = fasttext.train_supervised(input="emails.train", epoch=8, verbose=2, ws=4, neg=6)# dim=300, pretrainedVectors="wiki-news-300d-1M.vec")

        # Get predictions for each label
        y_predict = []
        for email in X_test:
            prediction, _ = model.predict(email)
            y_predict.append(1 if "__label__spam" in prediction else 0)
        y_predictions.append(y_predict)
        
    return y_predictions

def get_document_vector(sentence, vectors):
    """ Creates a document vector for a sentence by averaging word vectors """
    document_vector = np.zeros(300)
    num_valid_words = 0
    for word in sentence.split(' '):
        if vectors.__contains__(word):
            document_vector += vectors[word]
            num_valid_words += 1
    if num_valid_words > 0:
        document_vector /= num_valid_words
    # Maybe normalize?
    return document_vector

def prepare_vectors(X_train, y_train, X_valid, X_test):
    """ Create and save document vectors using pretrained word vectors """

    print("Loading pretrained vectors...",end='')
    vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
    print("Done")
    
    # Create a document vector for each sentence
    training_data = []
    validation_data = []
    testing_data = []
    
    empty_training_vector_positions = []
    for i, sentence in enumerate(X_train):
        vector = get_document_vector(sentence, vectors)
        if vector.any():
            empty_training_vector_positions.append(i)
        training_data.append(get_document_vector(sentence, vectors))
    # Remove empty vectors and labels from training data
    training_data = list(np.array(training_data)[empty_training_vector_positions])
    y_train = y_train[empty_training_vector_positions]

    for i, sentence in enumerate(X_valid):
        validation_data.append(get_document_vector(sentence, vectors))
        
    for sentence in X_test:
        testing_data.append(get_document_vector(sentence, vectors))

    # Save document vectors
    with open('email_vectors.train', 'wb') as train_file:
        pickle.dump(training_data, train_file)
    with open('email_vectors.trainlabels', 'wb') as trainlabel_file:
        pickle.dump(y_train, trainlabel_file)
    with open('email_vectors.valid', 'wb') as valid_file:
        pickle.dump(validation_data, valid_file)
    with open('email_vectors.test', 'wb') as test_file:
        pickle.dump(testing_data, test_file)

def run_pretrained_classifier(validation=True):
    """ Use pre-trained word vectors to run a classifier """

    # Read vectors
    training_data = []
    testing_data = []
    with open('email_vectors.train', 'rb') as train_file:
        training_data = pickle.load(train_file)
    with open('email_vectors.trainlabels', 'rb') as trainlabel_file:
        y_train = pickle.load(trainlabel_file)
    with open('email_vectors.valid' if validation else 'email_vectors.test', 'rb') as test_file:
        testing_data = pickle.load(test_file)

    print(sum([1 for v in testing_data if not v.any()]))
    

    y_predictions= []
    for i in range(10):
        clf = linear_model.SGDClassifier()
        clf.fit(training_data, y_train)
        y_predictions.append(clf.predict(testing_data))

    return y_predictions

X_train, y_train, X_test, y_test, X_valid, y_valid = get_data()
#prepare_vectors(X_train, y_train, X_valid, X_test)
#y_predictions = run_pretrained_classifier(False)

#tune_fasttext(X_train, y_train, X_valid, y_valid)
#y_predictions = run_fasttext_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test)
y_predictions = run_nb_classifier(X_train, y_train, X_test)
report_stats(y_test, y_predictions)
