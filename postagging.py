import nltk
from nltk.corpus import indian
from nltk.tag import tnt
from nltk import word_tokenize, pos_tag
import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import precision_recall_fscore_support as score

tagged_sentences = [sentences for sentences in indian.tagged_sents('marathi.pos')]
nltk.help.upenn_tagset()
print("Number of Tagged Sentences ", len(tagged_sentences))
tagged_words = [tup for sent in tagged_sentences for tup in sent]
print("Total Number of Tagged words", len(tagged_words))
vocab = set([word for word, tag in tagged_words])
print("Vocabulary of the Corpus", len(vocab))
tags = set([tag for word, tag in tagged_words])
print("Number of Tags in the Corpus ", len(tags))
print(tags)


def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit()
    }

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

cutoff = int(0.80 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y

X, y = transform_to_dataset(training_sentences)

clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)), ('classifier', DecisionTreeClassifier(criterion='entropy'))])

clf.fit(X[:], y[:])

print('Training completed')

X_test, y_test = transform_to_dataset(test_sentences)

print("Accuracy:", clf.score(X_test, y_test) * 100)

target_names = ['', 'QO', 'UNK', '"CC', 'SYM', 'NEG', 'QF', 'JJ', 'RDP', 'UT', 'PRP', 'NN', 'INTF', 'INJ', 'NNP', 'NST', 'VM', 'DEM', 'WQ', 'PSP', 'RP', 'SYMC', 'NNPC', 'QC', 'VAUX', 'NNC', 'RB', 'CC']

#scores = cross_val_score(clf, X_test, y_test, cv=5, scoring='f1_macro')

#print(scores)

def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    k = 0
    for word in sentence:
        print(word + " " + tags[k])
        k = k + 1
    return zip(sentence, tags)

pos_tag(word_tokenize('वाक्य हे शब्दांनी बनेलेले असते. त्या शब्दाच्या एकुण आठ जाती आहेत.'))