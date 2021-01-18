# train-a-part-of-speech-tagger-using-the-Hidden-Markov-HMM-Model-and-the-Viterbi-decoding-algorithm

The minimum requirement for this assignment is to implement and train a part-of-speech tagger using the Hidden Markov (HMM) Model and the Viterbi decoding algorithm, as described in class and in Chapter 8 of the text (SLP3).

We provide you with a copy of a labeled corpus used for training and testing POS taggers. This data is made available to you in the form of a zip file, POS_data.zip. This zip file contains:
POS_train.pos: words and tags for training corpus
POS_dev.word: words for development corpus
POS_dev.pos: words and tags for development corpus
POS_test.word: words for test corpus
scorer.py: the scorer file provided to evaluate the output files (need Python 3)

Files with a “word” extension have the document text, one word per line; for those with a “pos” extension, each line consists of a word, a tab character, and a part-of-speech. In both formats, an empty line separates sentences.
You should train your tagger on the training corpus and evaluate its performance on the development corpus. We provide a simple Python scoring program for this purpose (we will use this scorer to score your output on the test set as well). Please read the scorer file to know how to run it (“python scorer.py keyFile responseFile”). Having a development corpus allows you to evaluate alternatives in designing the tagger: for example, how to treat unknown words. When you are satisfied with your tagger you should run it on the test data and submit the result. You have the option of training on the union of the training and development corpora before making this final run. Note that we do not provide the key (.pos file) for test data; you should not train or do development using the test data.
The minimal requirement involves (1) estimating probabilities that are necessary for a HMM model, and (2) implement Viterbi decoding. For unknown words, a baseline strategy is to use a uniform probability these words (i.e., if you see an unknown word in test data, you assume the same probability for every POS tag in the emission probabilities). If you are more ambitious, you may want to make additional enhancements to improve performance through better treatment of unknown words or through trigram models, as discussed in section 5.8 of the text (i.e., SLP2). Some extra credit will be given to assignments that incorporate such improvements.
