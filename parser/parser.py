import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | VP NP | S NP | S Conj S | S PP
NP -> N | Det N | Det Adj NP | Adj NP | NP PP
VP -> V | V Adv | Adv V | V PP Adv
PP -> P NP

"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.tokenize.word_tokenize(sentence)
    words_to_remove = set()

    for i in range(len(words)):
        if not words[i].isalpha():
            words_to_remove.add(words[i])
            continue

        words[i] = words[i].lower()

    for word in words_to_remove:
        words.remove(word)

    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    noun_phrases = list()

    np_candidates = get_NP(tree, list())

    for np in np_candidates:
        if check_np(np):
            noun_phrases.append(np)

    # Remove duplicates.
    filtered_npchunks = list()
    for np in noun_phrases:
        if np not in filtered_npchunks:
            filtered_npchunks.append(np)

    return filtered_npchunks


def get_NP(tree, NPs):
    """
    Returns every singls NP in all of the subtrees to the deep.
    """
    subtrees = list(tree.subtrees())
    if len(subtrees) == 1 and subtrees[0] == tree:
        return NPs

    for subtree in subtrees[1:]:
        if subtree.label() == 'NP':
            NPs.append(subtree)
        get_NP(subtree, NPs)
    return NPs


def check_np(tree):
    """
    Check that NPs don't contain other NP in their subtrees.
    """
    subtrees = list(tree.subtrees())
    if len(subtrees) == 1 and subtrees[0] == tree:
        return True

    for subtree in subtrees[1:]:
        if subtree.label() == 'NP':
            return False

    for subtree in subtrees[1:]:
        return check_np(subtree)


if __name__ == "__main__":
    main()
