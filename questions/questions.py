import nltk
import sys

import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = os.listdir(directory)
    files_dict = {
        filename: str() for filename in files
    }

    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            files_dict[file] = str(f.read())

    return files_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.tokenize.wordpunct_tokenize(document)
    stopwords = nltk.corpus.stopwords.words("english")
    punctuation = string.punctuation

    words_to_remove = set()

    for i in range(len(words)):
        words[i] = words[i].lower()

        if words[i] in stopwords or words[i] in punctuation or not words[i].isalpha:
            words_to_remove.add(words[i])
            continue

    for word in words_to_remove:
        words.remove(word)

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = list()
    for document in documents:
        for word in documents[document]:
            words.append(word)

    idfs = {
        word: math.log(len(documents) / NumDocumentsContaining(word, documents)) for word in words
    }

    return idfs


def NumDocumentsContaining(word, docs):
    """
    Returns the number of documents that contain the word.
    """
    numDocs = 0
    for doc in docs:
        if word in docs[doc]:
            numDocs += 1

    return numDocs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    top_filenames = {
        file_name: 0 for file_name in list(files.keys())
    }

    for file in files:
        tfidf_values_for_query = set()

        for word in query:
            if word not in files[file]:
                continue

            tfidf_values_for_query.add(tf(word, files[file]) * idfs[word])

        top_filenames[file] += sum(tfidf_values_for_query)

    # Sort the files from best to worst.
    top_filenames = list(dict(sorted(top_filenames.items(), key=lambda item: item[1], reverse=True)))

    return top_filenames[:n]


def tf(word, doc):
    """
    Returns the number of how frequently a specific word appears in a specific document.
    """
    term_frequency = 0
    for w in doc:
        if word == w:
            term_frequency += 1

    return term_frequency


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_sents_dict = {
        sent: 0 for sent in sentences
    }
    for sent in sentences:
        qwords_idf = set()

        for word in query:
            if word not in sentences[sent]:
                continue

            qwords_idf.add(idfs[word])

        top_sents_dict[sent] += sum(qwords_idf)

    top_sents_dict = dict(sorted(top_sents_dict.items(), key=lambda item: item[1], reverse=True))
    top_sents_list = list(top_sents_dict)

    tsd_values = list(top_sents_dict.values())

    # Choose the best sentence if two sentences have the same matching measure value.
    for i in range(len(tsd_values)):
        for j in range(i+1, len(tsd_values)):
            if tsd_values[i] != tsd_values[j]:
                continue

            if QTermDensity(query, sentences[top_sents_list[i]]) < QTermDensity(query, sentences[top_sents_list[j]]):
                top_sents_list[i], top_sents_list[j] = top_sents_list[j], top_sents_list[i]

    return top_sents_list[:n]


def QTermDensity(query, sentence_words):
    """
    Returns the query word density in the sentence.
    """
    qwords_in_sentence = 0
    for word in query:
        if word in sentence_words:
            qwords_in_sentence += 1

    return qwords_in_sentence / len(sentence_words)


if __name__ == "__main__":
    main()
