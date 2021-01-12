import pickle, sys
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import friedmanchisquare

def load_data(foldername, validation=True):
    """
    Load training and testing vectors
    """
    
    test_filename = "validation" if validation else "testing"
    with open('vectors/{}/training.vectors'.format(foldername), 'rb') as f:
        X_train = pickle.load(f)
    with open('vectors/{}/{}.vectors'.format(foldername, test_filename), 'rb') as f:
        X_test = pickle.load(f)
    with open('preprocessed/training.labels', 'rb') as f:
        y_train = pickle.load(f)
    with open('preprocessed/{}.labels'.format(test_filename), 'rb') as f:
        y_test = pickle.load(f)
    return X_train, y_train, X_test, y_test

def run_classifier(X_train, y_train, X_test):
    """
    Run an SVM to classify the emails
    """

    clf = make_pipeline(StandardScaler(with_mean=False), svm.SVC(kernel="linear"))
    #clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def report_stats(y_test, y_predict):

    gen_rec = recall_score(y_test, y_predict, pos_label=0)
    spam_rec = recall_score(y_test, y_predict, pos_label=1)
    acc = accuracy_score(y_test, y_predict)

    print("GEN PRECISION: ", precision_score(y_test, y_predict, pos_label=0))
    print("GEN RECALL ", recall_score(y_test, y_predict, pos_label=0))
    print("GEN F1 ", f1_score(y_test, y_predict, pos_label=0))
    print("SPAM PRECISION: ", precision_score(y_test, y_predict, pos_label=1))
    print("SPAM RECALL ", recall_score(y_test, y_predict, pos_label=1))
    print("SPAM F1 ", f1_score(y_test, y_predict, pos_label=1))

    print('Accuracy score:', accuracy_score(y_test, y_predict))
    print('Macro F1 score:', f1_score(y_test, y_predict, average='macro'))

if len(sys.argv) == 3:
    foldername = sys.argv[1]
    validation = sys.argv[2].lower() == "validation"
    print("Testing SVM classifier using {} set for testing and {} for vector type".format(sys.argv[2], foldername))
    X_train, y_train, X_test, y_test = load_data(foldername, validation)
    y_predict = run_classifier(X_train, y_train, X_test)
    report_stats(y_test, y_predict)
if len(sys.argv) == 4:
    foldername_A = sys.argv[1]
    foldername_B = sys.argv[2]
    validation = sys.argv[3].lower() == "validation"
    X_train, y_train, X_test, _ = load_data(foldername_A, validation)
    y_predict_A = run_classifier(X_train, y_train, X_test)
    X_train, y_train, X_test, _ = load_data(foldername_B, validation)
    y_predict_B = run_classifier(X_train, y_train, X_test)
    print("Running Wilcoxon Signed-Rank Test comparing {} vectors to {} vectors on {} set".format(foldername_A, foldername_B, sys.argv[3]))
    stat, p = friedmanchisquare(y_predict_A, y_predict_B)
    print('stat=%.3f, p=%.3f' % (stat, p))  
