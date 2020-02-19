import numpy as np
import text_functions as tf
import nltk

nltk.download("wordnet")

# @author: The first version of this code is the courtesy of Vadim Selyanik

threshold = 15000  # Frequency threshold in the corpus ??
lemmatizer = nltk.WordNetLemmatizer()  # create an instance of lemmatizer
ones_number = (
    2  # number of nonzero elements in randomly generated high-dimensional vectors
)
window_size = 2  # number of neighboring words to consider both back and forth. In other words number of words before/after current word
test_name = "new_toefl.txt"  # file with TOEFL dataset
data_file_name = "lemmatized.text"  # file with the text corpus


def count_words():
    # Count how many times each word appears in the corpus
    text_file = open(data_file_name, "r")
    res = {}
    for line in text_file:
        if line != "\n":
            words = line.split()
            for word in words:
                res[word] = 1 if res.get(word) is None else res[word] + 1
    text_file.close()
    return res


amount_dictionary = count_words()

word_space = {}  # embedings


def create_vector_dict(amount_dict, dimension):
    # Create a dictionary with the assigned random high-dimensional vectors
    text_file = open(data_file_name, "r")
    res = {}
    for line in text_file:  # read line in the file
        words = line.split()  # extract words from the line
        for word in words:  # for each word
            if res.get(word) is None:  # If the word was not yed added to the vocabulary
                res[word] = (
                    tf.get_random_word_vector(dimension, ones_number)
                    if amount_dict[word] < threshold
                    else np.zeros(dimension)
                )

    text_file.close()
    return res


def build_word_embedding(dictionary):
    # Processing the text to build the embeddings
    text_file = open(data_file_name, "r")
    lines = [[], [], [], []]  # neighboring lines
    for i in range(2, 4):
        line = "\n"
        while line == "\n":
            line = text_file.readline()
        lines[i] = line.split()

    line = text_file.readline()
    while line != "":
        if line != "\n":
            lines.append(line.split())
            words = lines[2]
            length = len(words)
            for i in range(length):
                if not (word_space.get(words[i]) is None):
                    k = 1
                    while (i - k >= 0) and (
                        k <= window_size
                    ):  # process left neighbors of the focus word
                        word_space[words[i]] = np.add(
                            word_space[words[i]], np.roll(dictionary[words[i - k]], -1)
                        )
                        k += 1
                    # Handle different situations if there was not enough neighbors on the left in the current line
                    if k <= window_size and (len(lines[1]) > 0):
                        if len(lines[1]) < 2:
                            word_space[words[i]] = np.add(
                                word_space[words[i]],
                                np.roll(dictionary[lines[1][0]], -1),
                            )
                            if k == 1:  # if on the first word
                                word_space[words[i]] = np.add(
                                    word_space[words[i]],
                                    np.roll(
                                        dictionary[lines[0][len(lines[0]) - 1]], -1
                                    ),
                                )
                        else:
                            word_space[words[i]] = np.add(
                                word_space[words[i]],
                                np.roll(dictionary[lines[1][len(lines[1]) - 1]], -1),
                            )
                            if k == 1:
                                word_space[words[i]] = np.add(
                                    word_space[words[i]],
                                    np.roll(
                                        dictionary[lines[1][len(lines[1]) - 2]], -1
                                    ),
                                )

                    k = 1
                    while (
                        i + k < length and k <= window_size
                    ):  # process right neighbors of the focus word
                        word_space[words[i]] = np.add(
                            word_space[words[i]], np.roll(dictionary[words[i + k]], 1)
                        )
                        k += 1
                    if k <= window_size:
                        if len(lines[3]) < 2:
                            word_space[words[i]] = np.add(
                                word_space[words[i]],
                                np.roll(dictionary[lines[3][0]], 1),
                            )
                            if k == 1:
                                word_space[words[i]] = np.add(
                                    word_space[words[i]],
                                    np.roll(dictionary[lines[4][0]], 1),
                                )
                        else:
                            word_space[words[i]] = np.add(
                                word_space[words[i]],
                                np.roll(dictionary[lines[3][0]], 1),
                            )
                            if k == 1:
                                word_space[words[i]] = np.add(
                                    word_space[words[i]],
                                    np.roll(dictionary[lines[3][1]], 1),
                                )

            lines.pop(0)
        line = text_file.readline()


def toefl_test(number_of_tests, dimension):
    # Testing of the embeddings on TOEFL
    a = 0.0  # accuracy of the encodings
    i = 0
    zero_vector = np.zeros(dimension)
    text_file = open(test_name, "r")
    right_answers = 0.0  # variable for correct answers
    number_skipped_tests = 0.0  # some tests could be skipped if there are no corresponding words in the vocabulary extracted from the training corpus
    while i < number_of_tests:
        line = text_file.readline()  # read line in the file
        words = line.split()  # extract words from the line
        words = [
            lemmatizer.lemmatize(
                lemmatizer.lemmatize(lemmatizer.lemmatize(word, "v"), "n"), "a"
            )
            for word in words
        ]  # lemmatize words in the current test
        try:

            if not (
                amount_dictionary.get(words[0]) is None
            ):  # check if there word in the corpus for the query word
                k = 1
                while k < 5:
                    # if amount_dictionary.get(words[k]) is None:
                    #     word_space[words[k]] = np.random.randn(dimension)
                    if np.array_equal(
                        word_space[words[k]], zero_vector
                    ):  # if no representation was learnt assign a random vector
                        word_space[words[k]] = np.random.randn(dimension)
                    k += 1
                right_answers += tf.get_answer_mod(
                    [
                        word_space[words[0]],
                        word_space[words[1]],
                        word_space[words[2]],
                        word_space[words[3]],
                        word_space[words[4]],
                    ]
                )  # check if word is predicted right
        except KeyError:  # if there is no representation for the query vector than skip
            number_skipped_tests += 1
            print("skipped test: " + str(i) + "; Line: " + str(words))
        except IndexError:
            print(i)
            print(line)
            print(words)
            break
        i += 1
    text_file.close()
    a += 100 * right_answers / number_of_tests
    print(
        "Dimension: "
        + str(dimension)
        + "\t Percentage of correct answers: "
        + str(100 * right_answers / number_of_tests)
        + "%"
    )


def train_and_test(dimension, simulations):
    for i in range(simulations):
        print(f"Simulation {i}")
        dictionary = create_vector_dict(amount_dictionary, dimension)

        # Note that in order to save time we only create embeddings for the words needed in the TOEFL task

        # Find all unique words amongst TOEFL tasks and initialize their embeddings to zeros
        number_of_tests = 0
        text_file = open(test_name, "r")  # open TOEFL tasks
        for line in text_file:
            words = line.split()
            words = [
                lemmatizer.lemmatize(
                    lemmatizer.lemmatize(lemmatizer.lemmatize(word, "v"), "n"), "a"
                )
                for word in words
            ]  # lemmatize words in the current test
            word_space[words[0]] = np.zeros(dimension)
            word_space[words[1]] = np.zeros(dimension)
            word_space[words[2]] = np.zeros(dimension)
            word_space[words[3]] = np.zeros(dimension)
            word_space[words[4]] = np.zeros(dimension)
            number_of_tests += 1
        text_file.close()

        build_word_embedding(dictionary)

        toefl_test(number_of_tests, dimension)


train_and_test(dimension=1000, simulations=5)
train_and_test(dimension=4000, simulations=5)
train_and_test(dimension=10000, simulations=5)
