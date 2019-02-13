# import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as stop
from wordcloud import WordCloud


def load_stopwords(files=None):
    'Load files of stopwords and the basic english set from nltk'
    if files is not None:
        stopwords = set()
        for file in files:
            stopwords = stopwords.union(set(w.rstrip() for w in open(file)))
        stopwords = stopwords.union(set(stop.words('english')))
    else:
        stopwords = set(stop.words('english'))

    return stopwords


def my_tokenizer(s):
    '''
    Break input text in to a list of words, lemmatize them, and remove those
    that will not be useful. Short words, stopwords, etc.
    '''
    s = s.lower()  # downcase
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words
    # put words into base form (remove plural, 'ing', 'ed', etc)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


# maybe add something in here to remove lines by CCB
def process(lines):
    word_index_map = {}
    freqs = {}
    current_index = 0
    all_tokens = []
    all_lines = []
    index_word_map = []
    error_count = 0
    for line in lines:
        try:
            # this will throw exception if bad characters
            line = line.encode('ascii', 'ignore').decode('utf-8')
            all_lines.append(line)
            tokens = my_tokenizer(line)
            all_tokens.append(tokens)
            for token in tokens:
                if token not in word_index_map:
                    word_index_map[token] = current_index
                    current_index += 1
                    index_word_map.append(token)
                    freqs[token] = 1
                else:
                    freqs[token] += 1
        except Exception as e:
            print(e)
            print(line)
            error_count += 1

    return all_lines, all_tokens, word_index_map, index_word_map, freqs


def value(elem):
    return elem


def main():
    path = '..\\large_files\\Transcripts\\'
    interviews = {
        'chiros': ['CCP1TRUE', 'CCP2', 'CCP3'],
        'chiroClients': ['CC1'],
        'instructors': ['INSTRUCTOR1', 'INSTRUCTOR2'],
        'yogis': ['SKJ%s' % (i) for i in range(1, 10)],
    }

    all_raw = []
    for subject in interviews['yogis']:
        all_raw += [line.rstrip() for line in open(path + subject + '.txt')]

    (all_lines, all_tokens, word_index_map,
        index_word_map, freqs) = process(all_raw)

    sorted_freqs = []
    for w in sorted(freqs, key=freqs.get, reverse=True):
        if freqs[w] > 10:
            sorted_freqs.append([w, freqs[w]])
    print(sorted_freqs)

    wordcloud = WordCloud(
        stopwords=stopwords, width=800, height=400, colormap='jet',
        min_font_size=8
    ).fit_words(freqs)
    plt.imshow(wordcloud)
    plt.show()


if __name__ == '__main__':
    nltk.download(['punkt', 'wordnet', 'stopwords'])

    wordnet_lemmatizer = WordNetLemmatizer()

    stopwords = load_stopwords(files=['LP_stopwords.txt', 'my_stopwords.txt'])
    stopwords = stopwords.union({
        'ccb', 'n\'t', '\'ve', '\'re', 'pause', 'wa', 'mentioned', 'yeah',
        })
    main()
