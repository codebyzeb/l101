# L101 Machine Learning for NLP Project

This repository contains the report and scripts produced for my project as part of the L101 course on Machine Learning for NLP, offered by Part III of the Computer Science Tripos at Cambridge University.

## Report

The report can be found at `report.pdf`.

## Scripts

All scripts can be found in `scripts/`. The `prepare_data.py` script is used for the data preparation and pre-processing step, described in Section 3.1 and 3.2 of the report. To run it, place the [GenSpam corpus](http://www.benmedlock.co.uk/genspam.html) in the directory and run:

	cd data
	python ../scripts/prepare_data.py

The `prepare_vectors.py` script is used to produce sparse and dense vectors for the classifier, as described in Section 3.3. of the report. To produce dense vectors, place a set of pre-trained word vectors in the `data/` repository, such as the [Google News word2vec vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) and run:

	cd data
	python ../scripts/prepare_vectors.py WORDVECFILE FOLDERNAME

This will create sparse and dense vectors using the vocabulary provided by the pre-trained word vectors placed at `data/WORDVECFILE` and place them in the `data/vectors/sparse-FOLDERNAME` and `data/vectors/dense-FOLDERNAME` directories, respectively. To create sparse vectors with the full vocabulary available (the `sparse-full` vectors) just run the script without any arguments:

	cd data
	python ../scripts/prepare_vectors.py

The `classify_emails.py` script is used to run the SVM classifier, as described in Section 3.4 of the report. Run the script as follows:

	cd data
	python ../scripts/classify_emails.py FOLDERNAME [validation|testing]

Where `FOLDERNAME` is the name of the folder containing the vectors produced by the previous step. The classifier is run on either the "validation" or "testing" set, according to the second argument.
