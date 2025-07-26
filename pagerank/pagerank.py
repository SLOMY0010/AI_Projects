from copy import copy, deepcopy
import os
import random
import re
import sys
from tkinter import N

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pages = corpus.keys()
    page_links = corpus[page]
    p_distribution = dict()

    for p in pages:
        if len(page_links) == 0:
            p_distribution[p] = (1 - damping_factor) / len(pages) + damping_factor / len(pages)
        elif p in page_links:
            p_distribution[p] = (1 - damping_factor) / len(pages) + damping_factor / len(page_links)
        else:
            p_distribution[p] = (1 - damping_factor) / len(pages)
    
    return p_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    ranks = dict()
    visits = dict() # this dictionary is made to keep track of how many times a page is visited

    for page in corpus:
        visits[page] = 0
        ranks[page] = 0

    # start with a random page
    sample = random.choice(list(corpus))
    visits[sample] += 1
    
    for i in range(1, n):

        trans_model = transition_model(corpus, sample, damping_factor)

        # storing every possible page to visit next and storing the probability of visiting
        next_pages = list(trans_model.keys())
        probabilities = list(trans_model.values())

        # choose a random page from the possible pages to visit based on the probability distribution
        sample = random.choices(next_pages, weights=probabilities, k=1)[0]
        visits[sample] += 1

    # calculating the rank of each page
    for page in corpus:
        ranks[page] = visits[page] / n

    return ranks
            

    


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    ranks = dict()
    N = len(corpus.keys())

    for page in corpus:
        ranks[page] = 1 / N

    while True:
        # ...
        passed = 0

        # applying the equation:
        for page in corpus:

            # the 1st part of the equation
            part1 = (1 - damping_factor) / N
            sigma = 0

            # counting sigma
            for p in corpus:

                # if there are links in p that link to page, we count sigma, else we skip
                # and sigma will remain unchanged therefor the 2nd part of the equation will be 0
                if page in corpus[p]:
                    numLinks = len(corpus[p])
                    sigma += ranks[p] / numLinks

            # the 2nd part of the equation
            part2 = damping_factor * sigma

            new_rank = part1 + part2

            # ... this is to keep track of the number of pages that passed the convergence test
            # if all of them did, that means our job is done
            if abs(ranks[page] - new_rank) < 0.0001:
                passed += 1

            ranks[page] = new_rank

        if passed == N:
            break

    return ranks

if __name__ == "__main__":
    main()