import pickle, sys, os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors

def get_training_vocab(X_train):
    """
    Gets the vocabulary from the tokens in the training data
    """
    
    vocab = []
    for email in X_train:
        for token in email.split(" "):
            if not token in vocab:
                vocab.append(token)
    
    return np.unique(vocab)

def get_intersection_vocab(vocab, vectors):
    """
    Returns a vocabulary calculated as the intersection of the
    given vocabulary and the vocabulary provided by the pretrained
    word vectors
    """
    inter_vocab = []
    for v in vocab:
        if vectors.__contains__(v):
            inter_vocab.append(v)
    return np.unique(inter_vocab)

def save_sparse_vectors(X_train, X_valid, X_test, vocab, foldername):
    """
    Creates sparse count-based vectors using the provided vocabulary
    and saves the vectors for further classification
    """

    # Override tokenization step of countvectorizer
    cv = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split(' '))

    training_vectors = cv.fit_transform(X_train)
    validation_vectors = cv.transform(X_valid)
    testing_vectors = cv.transform(X_test)

    if not os.path.exists("vectors/"+foldername):
        os.makedirs("vectors/"+foldername)
    with open("vectors/{}/training.vectors".format(foldername), "wb") as f:
        pickle.dump(training_vectors, f)
    with open("vectors/{}/validation.vectors".format(foldername), "wb") as f:
        pickle.dump(validation_vectors, f)
    with open("vectors/{}/testing.vectors".format(foldername), "wb") as f:
        pickle.dump(testing_vectors, f)

    print("Saved sparse vectors of length {} to {}.".format(len(cv.vocabulary_), foldername))


def get_document_vectors(X, vectors):
    """
    Create document vectors by summing pretrained word vectors
    """
    
    document_vectors = []
    vector_size = len(vectors["car"])
    num_tokens = 0
    for document in X:
        document_vector = np.zeros(vector_size)
        for token in document.split(' '):
            if vectors.__contains__(token):
                vec = vectors[token]
                document_vector += vec
                num_tokens += 1
        document_vectors.append(document_vector)
    return np.array(document_vectors)
  
def save_dense_vectors(X_train, X_valid, X_test, vectors, foldername):
    """
    Creates dense vectors by summing pretrained word vectors for each token
    in a document and saves the vectors for further classification
    """

    training_vectors = get_document_vectors(X_train, vectors)
    validation_vectors = get_document_vectors(X_valid, vectors)
    testing_vectors = get_document_vectors(X_test, vectors)

    if not os.path.exists("vectors/"+foldername):
        os.makedirs("vectors/"+foldername)
    with open("vectors/{}/training.vectors".format(foldername), "wb") as f:
        pickle.dump(training_vectors, f)
    with open("vectors/{}/validation.vectors".format(foldername), "wb") as f:
        pickle.dump(validation_vectors, f)
    with open("vectors/{}/testing.vectors".format(foldername), "wb") as f:
        pickle.dump(testing_vectors, f)

    print("Saved dense vectors of length {} to {}.".format(training_vectors.shape[1], foldername))

# Load preprocessed data
with open('preprocessed/training.data', 'r') as f:
    X_train = np.array([line[:-1] for line in f.readlines()])
with open('preprocessed/validation.data', 'r') as f:
    X_valid = np.array([line[:-1] for line in f.readlines()])
with open('preprocessed/testing.data', 'r') as f:
    X_test = np.array([line[:-1] for line in f.readlines()])

# Create sparse vectors with full vocabulary or sparse and dense vectors
# with a limited vocabulary determined by the vocabulary of the pretrained vector set
full_vocab = get_training_vocab(X_train)

if len(sys.argv) < 2:
    save_sparse_vectors(X_train, X_valid, X_test, full_vocab, "sparse-full")
    
else:  
    # Load word vectors
    wordvec_file = sys.argv[1]
    foldername = sys.argv[2]
    vectors = KeyedVectors.load_word2vec_format(wordvec_file, binary=("bin" in wordvec_file))
    print("Loaded vectors...OK")

    # Limit vocabulary size for dense vector calculation
    limited_vocab = get_intersection_vocab(full_vocab, vectors)

    save_sparse_vectors(X_train, X_valid, X_test, limited_vocab, "sparse-"+foldername)
    save_dense_vectors(X_train, X_valid, X_test, vectors, "dense-"+foldername)
