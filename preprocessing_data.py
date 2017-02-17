import re
import collections
import shutil
from tensorflow.python.platform import gfile

num_movie_scripts = 10
vocabulary_size = 10000
fraction_dev = 50
path_for_x_train = 'X_train.txt'
path_for_y_train = 'y_train.txt'
path_for_x_dev = 'X_dev.txt'
path_for_y_dev = 'y_dev.txt'


_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\":;)(])")
_DIGIT_RE = re.compile(br"\d")

#FROM DATA UTILS
# Build the dictionary with word-IDs from self-made dictionary and replace rare words with UNK token.
def build_dataset(words, vocabulary_size):
    count = [['_UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def create_vocabulary(dictionary, vocabulary_path):
    f = open(vocabulary_path, 'w')
    
    for key in dictionary:
        f.write(dictionary[key] + '\n')
    f.close()

def initialize_vocabulary(vocabulary_path):
  # finds vocabulary file
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def generate_encoded_files2(x_train_file, y_train_file, x_dev_file, y_dev_file, tokenized_sentences, dictionary):
    """Sentence A is in x_train, Sentence B in y_train"""
    encoded_holder = []
    unk_id = dictionary['_UNK']
    for sentence in tokenized_sentences:
        encoded_holder.append(encode_sentence(sentence, dictionary, unk_id))

    f1 = open(x_train_file, 'w')
    f2 = open(y_train_file, 'w')
    fraction = int(len(encoded_holder) / fraction_dev)
    if (len(encoded_holder) % 2 == 0):
        end = len(encoded_holder)
    else:
        end = len(encoded_holder)-1

    for i in xrange(0,fraction,2):
        f1.write(str(encoded_holder[i]) + '\n')
        f2.write(str(encoded_holder[i+1]) + '\n')

    f1.close()
    f2.close()

    d1 = open(x_dev_file, 'w')
    d2 = open(y_dev_file, 'w')

    for i in xrange(fraction, end, 2):
        d1.write(str(encoded_holder[i]) + '\n')
        d2.write(str(encoded_holder[i+1]) + '\n')    

    d1.close()
    d2.close()


def generate_encoded_files(x_train_file, y_train_file, x_dev_file, y_dev_file, tokenized_sentences, dictionary):
    """Sentence A is in x_train and y_train, Sentence B in X_train and y_train"""
    encoded_holder = []
    f1 = open(x_train_file, 'w')

    last_line = tokenized_sentences.pop()
    first_line = tokenized_sentences.pop(0)
    dev_counter = int(len(tokenized_sentences) - len(tokenized_sentences)/fraction_dev)

    unk_id = dictionary['_UNK']
    first_line_encoded = encode_sentence(first_line, dictionary, unk_id)
    f1.write(first_line_encoded + '\n')

    # Creates data for X_train
    for x in xrange(dev_counter):
        encoded_sentence = encode_sentence(tokenized_sentences[x], dictionary, unk_id)
        encoded_holder.append(encoded_sentence)
        f1.write(encoded_sentence + '\n') # Write sentence to file
    f1.close()

    d1 = open(x_dev_file, 'w')
    # Creates data for x_dev_file
    for x in xrange(dev_counter, len(tokenized_sentences)):
        encoded_sentence = encode_sentence(tokenized_sentences[x], dictionary, unk_id)
        encoded_holder.append(encoded_sentence)
        d1.write(encoded_sentence + '\n') # Write sentence to file

    d1.close()

    # Creates data for y_train
    f2 = open(y_train_file, 'w')

    for x in xrange(dev_counter + 1):
        f2.write(encoded_holder[x] + '\n') # Write sentence to file

    f2.close()

    # Creates data for y_dev
    d2 = open(y_dev_file, 'w')
    for x in xrange(dev_counter + 1, len(tokenized_sentences)):
        d2.write(encoded_holder[x] + '\n') # Write sentence to file

    last_line_encoded = encode_sentence(last_line, dictionary, unk_id)
    d2.write(last_line_encoded + '\n')
    d2.close()

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens"""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def encode_sentence(sentence, dictionary, unk_id):
    # Extract first word (and don't add any space)
    if not sentence:
        return ""
    first_word = sentence.pop(0)
    if first_word in dictionary:
        encoded_sentence = str(dictionary[first_word])
    else:
        encoded_sentence = str(unk_id)

    # Loop rest of the words (and add space in front)
    for word in sentence:
        if word in dictionary:
            encoded_word = dictionary[word]
        else:
            encoded_word = unk_id
        encoded_sentence += " " + str(encoded_word)
    return encoded_sentence


def sentence_to_token_ids(sentence, vocabulary):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  words = basic_tokenizer(sentence)
  return [vocabulary.get(w, UNK_ID) for w in words]


def read_data(num_movie_scripts):
    data_tokens = []
    # Append each line in file to the set
    for i in range(0, num_movie_scripts):
        path = 'data/'+str(i)+'raw.txt'
        print 'Reading ', path, '...'
        lines = [line.rstrip('\n') for line in open(path)]
        data_tokens_temp = []
        for line in lines:
            # Tokenize each sentence
            data_tokens_temp.extend(re.findall(r'\S+', line))
        data_tokens.extend(data_tokens_temp)

    return data_tokens


# Reads data and puts every sentence in a TWO DIMENSIONAL array as tokens
# data_tokens[0] = ['This', 'is', 'a', 'sentence']
def read_sentences(num_movie_scripts):
    data_tokens = []
    # Append each line in file to the set
    for i in range(0, num_movie_scripts):
        path = 'data/'+str(i)+'raw.txt'
        print 'Reading ', path, '...'
        lines = [line.rstrip('\n') for line in open(path)]
        data_tokens_temp = []
        for line in lines:
            # Tokenize each sentence
            data_tokens_temp.append(re.findall(r'\S+', line))
        data_tokens.extend(data_tokens_temp)
    return data_tokens



def make_files(num_movie_scripts, vocabulary_size, fraction_dev=50, path_for_x_train = 'X_train.txt', path_for_y_train = 'y_train.txt', path_for_x_dev = 'X_dev.txt', path_for_y_dev = 'y_dev.txt'):
    # Generate dictionary for dataset
    print '------------------------------------------------'
    print ' Generating dictionary based on ', str(num_movie_scripts), ' scripts'
    print '------------------------------------------------'

    tokenized_data = read_data(num_movie_scripts)
    data, count, dictionary, reverse_dictionary = build_dataset(tokenized_data, vocabulary_size)
    create_vocabulary(reverse_dictionary, 'vocabulary_for_movies.txt')


    # Generate an encoded file using the freated dictionary
    print '------------------------------------------------'
    print ' Creating encoded file using created dictionary'
    print ' (Saved in  ', path_for_x_train, ')'
    print '------------------------------------------------'
    tokenized_sentences = read_sentences(num_movie_scripts)
    generate_encoded_files(path_for_x_train, path_for_y_train, path_for_x_dev, path_for_y_dev, tokenized_sentences, dictionary)





#-----------------------Printing methods----------------------------
def print_dic(dic, counter):
    c = 0
    for x in dic:
        print x
        c += 1
        if(c == counter):
            break

def print_info(data, count, dictionary, reverse_dictionary):
    print '-------- data'
    print data
    print '-------- count'
    print count
    print '-------- dictionary'
    print_dic(dictionary, 5)
    print dictionary
    print '-------- reverse_dictionary'
    print_dic(reverse_dictionary, 5)
    print reverse_dictionary

