import re
import numpy as np
from numpy.random import default_rng
import pickle

def get_emails(filename):

    """
    Extracts the text from the message body field of each email, not including embedded text
    and returns an array of emails, each containing an array of tokens present in the email.
    """

    emails = []
    
    in_body = False
    in_normal = False
    email = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        i = 0
        while line:
            
            # Some emails have invalid characters, throw these out
            try:
                line = fp.readline()
            except UnicodeDecodeError:
                print("unicode error",filename)
                in_normal = False
                in_body = False
                email = []

            # Only parse normal text message bodies
            if "<MESSAGE_BODY>" in line:
                in_body = True
                email = []
            if "</MESSAGE_BODY>" in line:
                in_body = False
                emails.append(email)
            if "<TEXT_NORMAL>" in line:
                in_normal = True
            if "</TEXT_NORMAL>" in line:
                in_normal = False
            
            if in_body and in_normal and line[0] == "^": # dataset is already sentence-split
                sentence = line.split(" ") # dataset is already tokenized
                email.extend(sentence[1:-1]) # remove start-of-line character and end-of-line character

    return emails

def preprocess(emails):
    """
    Given an array of tokenized emails, perform preprocessing by lowercasing
    and only keeping tokens of length 2-15. Return emails as space-separated strings.
    """
    
    # Regex used to remove punctuation and words longer than 15 characters or dhorter than 2
    preprocess_regex = re.compile("(\\w|&)\\w{1,14}$")

    preprocessed_emails = []
    for email in emails:
        processed_email = [token.lower() for token in email if preprocess_regex.match(token)]
        if processed_email != []:
            preprocessed_emails.append(" ".join(processed_email))

    return preprocessed_emails

def get_data_from_emails():
    """
    Parse, preprocess and shuffle the GenSpam dataset to produce training, validation and testing data
    """

    # Extract training email bodies from files
    train_gen = preprocess(get_emails("genspam/train_GEN.ems"))
    train_spam = preprocess(get_emails("genspam/train_SPAM.ems"))

    # Extract testing email bodies from files
    test_gen = preprocess(get_emails("genspam/test_GEN.ems"))
    test_spam = preprocess(get_emails("genspam/test_SPAM.ems"))

    # Shuffle training data before creating validation set
    rng = default_rng(0)
    shf1 = rng.permutation(len(train_gen))
    shf2 = rng.permutation(len(train_spam))
    train_gen = list(np.array(train_gen)[shf1])
    train_spam = list(np.array(train_spam)[shf2])

    # Create a validation set the same size as the test set
    valid_gen = train_gen[-len(test_gen):]
    valid_spam = train_spam[-len(test_spam):]

    # Removing validations set from training set
    train_gen = train_gen[:-len(test_gen)]
    train_spam = train_spam[:-len(test_spam)]

    print("Using {} genuine emails and {} spam emails from TRAINING set for training".format(len(train_gen), len(train_spam)))
    print("Using {} genuine emails and {} spam emails from TRAINING set for validation".format(len(valid_gen), len(valid_spam)))
    print("Using {} genuine emails and {} spam emails from TEST set for testing".format(len(test_gen), len(test_spam)))

    # Combine spam and ham into training, testing and validation sets
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

    # Shuffle genuine and spam emails together
    rng = default_rng(0)
    shuffle_train = rng.permutation(train_size)
    shuffle_test = rng.permutation(test_size)
    shuffle_valid = rng.permutation(valid_size)
    X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]
    X_valid, y_valid = X_valid[shuffle_valid], y_valid[shuffle_valid]
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid

def save_X_y(X, y, filename):
    # Save data and labels to files

    with open("preprocessed/{}.data".format(filename), 'w') as f:
        f.writelines([line+"\n" for line in X])
    with open("preprocessed/{}.labels".format(filename), 'wb') as f:
        pickle.dump(y, f)

# Save training, validation and testing data to files  
X_train, y_train, X_test, y_test, X_valid, y_valid = get_data_from_emails()
save_X_y(X_train, y_train, "training")
save_X_y(X_test, y_test, "testing")
save_X_y(X_valid, y_valid, "validation")

