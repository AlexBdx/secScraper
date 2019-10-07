from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib
import re
import string
import math
from tqdm import tqdm
import nltk


# tokenize = lambda string_of_text: string_of_text.lower().split(" ")

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude




def diff_jaccard(str1, str2):
    """
    Calculates the Jaccard similarity between two strings.

    :param str1: First string.
    :param str2: Second string.
    :return: float in the [0, 1] interval
    """
    assert type(str1) == list and type(str2) == list
    assert type(str1[0]) == str and type(str2[0]) == str
    a = set(str1) 
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def diff_sk_cosine_tf(str1, str2, stop_words):
    # Direct via sklearn
    # sklearn_tf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True, tokenizer=tokenize)
    sklearn_tf = TfidfVectorizer(norm='l2', stop_words=stop_words, use_idf=False)
    sklearn_representation = sklearn_tf.fit_transform([str1, str2])
    x, y = sklearn_representation.toarray()
    cosine_similarity = linear_kernel(x.reshape(1, -1), y.reshape(1, -1))  # Only works with norm='l2'
    return cosine_similarity[0][0]

def diff_sk_cosine_tf_idf(str1, str2, stop_words):
    # sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    sklearn_tfidf = TfidfVectorizer(norm='l2', stop_words=stop_words, use_idf=True)
    sklearn_representation = sklearn_tfidf.fit_transform([str1, str2])
    x, y = sklearn_representation.toarray()
    cosine_similarity = linear_kernel(x.reshape(1, -1), y.reshape(1, -1))  # Only works with norm='l2'
    return cosine_similarity[0][0]

def diff_cosine_tf(str1, str2):
    """
    Calculates the Cosine TF similarity between two strings.

    :param str1: First string.
    :param str2: Second string.
    :return: float in the [0, 1] interval
    """
    vect = TfidfVectorizer(use_idf=False)  # Per paper
    tf = vect.fit_transform([str1, str2])
    tf_similarity = tf * tf.T
    return float(tf_similarity[0, 1])

def diff_cosine_tf_idf(str1, str2):
    """
    Calculates the Cosine TF-IDF similarity between two strings.

    :param str1: First string.
    :param str2: Second string.
    :return: float in the [0, 1] interval
    """
    vect = TfidfVectorizer(use_idf=True)  # Activate TF-IDF
    tfidf = vect.fit_transform([str1, str2])
    tfidf_similarity = tfidf * tfidf.T
    return float(tfidf_similarity[0, 1])


def diff_minEdit(str1, str2):
    """
    Calculates the minEdit similarity between two strings. This is word based.
    WARNING: VERY SLOW BEYOND ~10,000 CHAR TO COMPARE.

    :param str1: First string.
    :param str2: Second string.
    :return: float in the [0, 1] interval
    """
    f = difflib.SequenceMatcher(None, a=str1, b=str2)
    count_words_str1 = len(re.compile(r'\w+').findall(str1))
    count_words_str2 = len(re.compile(r'\w+').findall(str2))
    transformations = f.get_opcodes()  # Impossible to compute for larger texts
    transformations = [t for t in transformations if t[0] != 'equal']
    similarity = 1-len(transformations)/(count_words_str1+count_words_str2)
    similarity = abs(similarity)
    similarity = min(1, abs(similarity))  # Prevent it from being negative
    # similarity = f.ratio()  # That could be another option
    return similarity


def diff_gfg_editDistDP(str1, str2):
    # WARNING: O(m x n) complexity in RAM & time (if enough RAM...)
    # str1 = tokenize(str1)
    # str2 = tokenize(str2)
    assert type(str1) == list and type(str2) == list
    assert type(str1[0]) == str and type(str2[0]) == str
    
    m = len(str1)
    n = len(str2)
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
  
    # Fill d[][] in bottom up manner 
    for i in range(m+1): 
        for j in range(n+1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    # return dp[m][n]
    return 1 - dp[m][n]/(m+n)


def diff_simple(str1, str2):
    """
    Calculates the simple difference similarity between two strings. This is character based.
    WARNING: VERY SLOW BEYOND ~10,000 CHAR TO COMPARE.

    :param str1: First string.
    :param str2: Second string.
    :return: float in the [0, 1] interval
    """
    d = difflib.Differ()
    comparison = list(d.compare(str1, str2))
    comparison = [change for change in comparison if change[0] != ' ']
    similarity = 1-len(comparison)/(len(str1) + len(str2))
    return similarity

def diff_edit_distance(str1, str2):
    # str1 = tokenize(str1)
    # str2 = tokenize(str2)
    return nltk.edit_distance(str1, str2)/(len(str1)+len(str2))


def composite_index(data):
    """
    Create a composite index based on the sentiment analysis based on Loughran and McDonald's
    dictionary and script.

    :param data: String to analyse.
    :return: List of values. See unused variable OUTPUT_FIELDS in the source, there is a lot.
    """
    OUTPUT_FIELDS = ['file type', 'file size', 'number of words', '% positive', '% negative',
                 '% uncertainty', '% litigious', '% modal-weak', '% modal moderate',
                 '% modal strong', '% constraining', '# of alphabetic', '# of digits',
                 '# of numbers', 'avg # of syllables per word', 'average word length', 'vocabulary']

    # Sign will be of positive + negative proportion. Averaged by number of words.
    if data[2] + data[3] + data[4]:
        result = (data[3]-data[4])/data[2]
    else:  # Avoid the case when the text is too short and a div per zero error is thrown
        result = 0
    
    return result


def sing_sentiment(text, lm_dictionary):
    """
    Run the Loughran and McDonald's sentiment analysis on a string.

    :param text: String to analyze.
    :param lm_dictionary: Sentiment dictionary
    :return: Quite a few fields.
    """

    text_len = len(text)
    text = re.sub('(May|MAY)', ' ', text)  # drop all May month references ## lol
    text = text.upper()  # for this parse caps aren't informative so shift
    output_data = _get_data(text, lm_dictionary)
    output_data[0] = type(text)
    output_data[1] = text_len
    result = composite_index(output_data)

    return result


def _get_data(text, lm_dictionary):
    """
    Internal function to load the data and process it - comes from Loughran and McDonald's work with light
    modifications to incorporate it in my script.

    :param text: string to analyze
    :param lm_dictionary: Sentiment dictionary
    :return:
    """
    vdictionary = {}
    _odata = [0] * 17
    total_syllables = 0
    word_length = 0

    tokens = re.findall(r'\w+', text)  # Note that \w+ splits hyphenated words
    for token in tokens:
        if not token.isdigit() and len(token) > 1 and token in lm_dictionary:
            _odata[2] += 1  # word count
            word_length += len(token)
            if token not in vdictionary:
                vdictionary[token] = 1
            if lm_dictionary[token].positive: _odata[3] += 1
            if lm_dictionary[token].negative: _odata[4] += 1
            if lm_dictionary[token].uncertainty: _odata[5] += 1
            if lm_dictionary[token].litigious: _odata[6] += 1
            if lm_dictionary[token].weak_modal: _odata[7] += 1
            if lm_dictionary[token].moderate_modal: _odata[8] += 1
            if lm_dictionary[token].strong_modal: _odata[9] += 1
            if lm_dictionary[token].constraining: _odata[10] += 1
            total_syllables += lm_dictionary[token].syllables

    _odata[11] = len(re.findall('[A-Z]', text))
    _odata[12] = len(re.findall('[0-9]', text))
    # drop punctuation within numbers for number count
    text = re.sub(r'(?!=[0-9])(\.|,)(?=[0-9])', '', text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    _odata[13] = len(re.findall(r'\b[-+\(]?[$€£]?[-+(]?\d+\)?\b', text))
    _odata[14] = total_syllables / _odata[2]
    _odata[15] = word_length / _odata[2]
    _odata[16] = len(vdictionary)

    # Convert counts to %
    for i in range(3, 10 + 1):
        _odata[i] = (_odata[i] / _odata[2]) * 100
    # Vocabulary

    return _odata
