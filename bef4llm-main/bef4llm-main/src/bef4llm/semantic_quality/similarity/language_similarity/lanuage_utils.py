import enum
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem.snowball import GermanStemmer
import nltk

#nltk.download('wordnet')
#nltk.download('omw-1.4')


class Language(enum.Enum):
    """
    Enum for languages
    """
    ENGLISH = "english"
    GERMAN = "german"


def tokenize(sentence):
    """
    Tokenizes a sentence

    Parameters
    ----------
    sentence : str
        sentence to tokenize

    Returns
    -------
    tokens : list of str
        resulting tokens
    """
    token_words = sentence.split()
    return token_words

def remove_special_chars(tokenized_sentence):
    """
    Function removes special characters from the given, tokenized string.

    Parameters
    ----------
    tokenized_sentence: list
        List of tokenized string that represents label of the node of the process model.

    Returns
    -------
    result: list
        List of strings without special characters.
    """
    result = []
    for item in tokenized_sentence:
        string = str(re.sub('[^A-Za-z0-9]+', ' ', item).strip())
        if string == '':
            continue
        result.append(string)

    return result
def remove_stopwords(word_tokens, language):
    """
    Removes all stopwords of a list of tokens

    Parameters
    ----------
    word_tokens : list of str
        given tokens
    language : language
        language to use

    Returns
    -------
    filtered_tokens : list of str
        tokens without stopwords
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words(language.value))

    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]

    return filtered_sentence

def stem_tokens(word_tokens, language, stemmer=None):
    """
    Stemms all tokens

    Parameters
    ----------
    word_tokens : list of str ord
        tokens to stem
    language : language
        language to use
    stemmer : stemmer, optional
        stemmer to use

    Returns
    -------
    stemmed_tokens : list of str
        resulting tokens
    """

    stemmed_word_tokens = []

    for word_token in word_tokens:
        stemmed_word_tokens.append(stem_word(word_token, language, stemmer))

    return stemmed_word_tokens

def stem_word(word, language, stemmer=None):
    """
    Stems a word

    Parameters
    ----------
    word : str
        word to stem
    language : language
        language to use
    stemmer : stemmer, optional
        stemmer to use

    Returns
    -------
    stemmed: str
        stemmed word
    """
    if stemmer is None:

        if language is language.ENGLISH:
            stemmer = PorterStemmer()
        else:
            stemmer = GermanStemmer()

    stemmed = stemmer.stem(word)

    return stemmed


def check_synonym(word1, word2):
    """
    Function verifies whether two strings are synonyms or not.

    Parameters
    ----------
    word1 : str
        This word.
    word2 : str
        Other word, to be compared.

    Returns
    -------
    result : bool
        Returns True if words are synonyms, False otherwise.
    """
    try:
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('corpora/wordnet_ic.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('wordnet_ic')

    for syn in wordnet.synsets(word1):
        for lem in syn.lemmas():
            if lem.name() == word2:
                return True

    return False



