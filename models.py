# models.py

from sentiment_data import *
from utils import *

from nltk.corpus import stopwords
from collections import Counter
import numpy as np

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    
    
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence, add_to_indexer = False):
        features = Counter()
        words =  [w.lower() for w in sentence if w.isalpha()]
        for word in words:
            if(add_to_indexer):
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
            if idx != -1:
                features[idx] += 1
        return features

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
    def get_indexer(self):
        return self.indexer
    def extract_features(self, sentence, add_to_indexer = False):
        features = Counter()
        words =  [w.lower() for w in sentence if w.isalpha()]

        padded = ["<s>"] + words + ["</s>"]
        for i in range(len(padded)-1):
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(padded[i] + "|" + padded[i+1])
            else:
                idx = self.indexer.index_of(padded[i] + "|" + padded[i+1])
            if idx != -1:
                features[idx] += 1
        return features
    
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer =indexer
        self.min_freq = 3
        self.min_freq_bigram = 2
        self.stopwords = set(stopwords.words('english'))
        self.word_counts = Counter()


    def get_indexer(self):
        return self.indexer
    
        
    def extract_features(self, sentence, add_to_indexer = False):
        features = Counter()
        
        words = [w.lower() for w in sentence if w.isalpha() and (w.lower() not in self.stopwords)] ##preprocess
        
        #UNIGRAM
        for word in words:
            self.word_counts[word] += 1
            if self.word_counts[word]<self.min_freq:
                continue
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else: 
                idx = self.indexer.index_of(word)
            if idx != -1:
                features[idx] += 1
            
        #BIGRAM
        padded =["<s>"] + words + ["</s>"] 
        for i in range(len(padded)-1):
            bigram =padded[i] + "_" + padded[i+1] 
            self.word_counts[bigram] +=1
            if self.word_counts[bigram]<self.min_freq_bigram:
                continue
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
            if idx != -1:
                features[idx] += 1
        return features
            
class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor, weights):
        self.weights = weights 
        self.feature_extractor = feature_extractor

    def score(self, features):
        return sum(self.weights[f] * v for f, v in features.items())

    def predict(self, sentence):
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        return 1 if self.score(features) > 0 else 0

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor, weights):
        self.weights = weights 
        self.feature_extractor = feature_extractor
        self.loglikelihood = 0
    
    def score(self, features):
        return sum(self.weights[f] * v for f, v in features.items())
    def predict(self, sentence):
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        return 1 if self.score(features)> 0 else 0




def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    import numpy as np
    
    weights = Counter()
    
    def score(features):
        return sum(weights[f] * v for f, v in features.items())
 
    for epoch in range(1, 31):
        stepsize = 1
        np.random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
            pred = 1 if score(features) > 0 else 0
            if ex.label!=pred:
                for f, value in features.items():
                    if ex.label ==1:
                        weights[f] += value*stepsize
                    else: 
                        weights[f] -= value*stepsize
                    
    indexer = feat_extractor.get_indexer()
    word_weights = [(indexer.get_object(i), w) for i, w in weights.items()]
    word_weights.sort(key=lambda x: x[1])

    lowest = [w for w,i in word_weights[:10]]
    highest =[w for w,i in word_weights[-10:]]
    print(f"lowest: {lowest}\nhighest: {highest}")
    return PerceptronClassifier(feat_extractor, weights)

        


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    import numpy as np
    
    weights = Counter()
    def score(features):
        return sum(weights[f] * v for f, v in features.items())
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    epochs = 20
    ll = 0
    
    for epoch in range(1, epochs+1):
        stepsize =.08/epoch
        np.random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
            p = sigmoid(score(features))
            for f, value in features.items():
                weights[f] += (ex.label - p) * value * stepsize
            if epoch==epochs:
                y = ex.label
                p = np.clip(p, 1e-12, 1-1e-12)
                ll+= y*np.log(p)+(1-y)*np.log(1-p)
                     
    print(f"ll: {ll}")            
    return LogisticRegressionClassifier(feat_extractor, weights)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model