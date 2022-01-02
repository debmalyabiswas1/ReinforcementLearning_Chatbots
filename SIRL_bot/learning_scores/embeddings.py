import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf
import tensorflow_hub as hub

class embeddings(object):
    """ Compute sentence embeddings using either gloves (not advised) or universal sentence encoder (preferred) via tf-hub 
    Attributes:
    --------------------------------------------------------------
    embed_size: int

    Methods (public):
    -------------------------------------------------------------
    fit(texts): compute sentence embeddings 
    """
    
    def __init__(self, texts, extend_vocab = False, embed_par='gloves'):
        """ 
        Parameters:
        --------------------------------------------------
        texts: list of str 
        extend_vocab: boolean, useful only for gloves
        embed_par: str (either "gloves" or "tf")
        """
 
        self.STOP = set(stopwords.words('english'))
        self.embed_par = embed_par
        self.extend_vocab = extend_vocab

        if embed_par == 'gloves':
            self._glove_embeddings(texts)
        elif embed_par == 'tf':
            print('Loading TF hub embeddings') 
            self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
            
            
    def fit(self, texts):
        """ provides sentence embeddings """
        
        if self.embed_par == 'gloves':

            embeds = []
            for text in texts:
                words = self._tokenizer(text)
                embeds.append(self._BoW(words))

            return np.array(embeds)

        elif self.embed_par =='tf':

            return self._tf_embeddings(texts)

    def _glove_embeddings(self, texts):
        """ get words embeddings for words in vocabulary """
        
        words_to_index, index_to_words, word_to_vec_map = read_glove_vecs("deep_dialog/nlu/glove.6B.50d.txt")
        if self.extend_vocab:
            words_to_index, index_to_words, word_to_vec_map, new_vocab = increase_vocab(words_to_index, index_to_words, word_to_vec_map, texts)

        self.words_to_index = words_to_index
        self.index_to_words = index_to_words
        self.word_to_vec_map = word_to_vec_map

        self.embed_size = len(list(word_to_vec_map.values())[0])
        self.vocab_size = len(words_to_index)

    def _tokenizer(self, text, special_chars=['<br']):
           """ tokenize text and remove stopwords """
        
           #Step 1: remove special chars
           for spch in special_chars:
               text = text.replace(spch,'')

           # Step 2: tokenize text inot list of low case tokens
           tokenizer = RegexpTokenizer(r'\w+')
           tokens = tokenizer.tokenize(text.lower())

           # Step 3 : TODO : stemming

           # Step 4: remove stopwords
           words = np.array([w for w in tokens if w not in self.STOP]) #tokens without stopwords

           return words

    def _BoW(self, words, wei=None):

        """
        Provide Bag of Words  embeddings for text:
        - if wei == None : simple average of words embeddings
        - if wei is provided : weighted average

        words: list of words (tokens) in a text
        wei: list of weights, same size as words

        Return:
        embeds: text embeddings
        """

        if wei is None:
            wei = np.ones(self.vocab_size)

        avg = np.zeros(self.embed_size)

        sum_wei = 0

        if len(words) == 0:
            return avg

        for i, w in enumerate(words):

            # accumulate the word embeddings in avg if word is in vocabulary
            if w in self.word_to_vec_map:

                avg += wei[i]*self.word_to_vec_map[w] # weight by IDF
                #avg += word_to_vec_map[w] #None

            #don't take into account word if not in vocab
            else:
                continue

            sum_wei  = sum_wei +  wei[i]

        embeds = avg/float(sum_wei)

        return embeds


    def _tf_embeddings(self,texts):
        """ sentence embeddings from universal sentence encoder """
        
        print('Computing embeddings in Tensorflow...')
        N = np.shape(texts)[0]

        messages = tf.placeholder(dtype=tf.string, shape =[N])
        embeddings = self.embed(messages)
        
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sentence_embeddings = session.run(embeddings, feed_dict={messages: texts})

        
        self.embed_size = np.shape(sentence_embeddings)[1]
        
        return sentence_embeddings
    

def read_glove_vecs(glove_file):
    """ read gloves files with pre-trained word embeddings  """
    
    with open(glove_file, 'r', encoding="utf8") as f:

        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1

    return words_to_index, index_to_words, word_to_vec_map

def increase_vocab(words_to_index, index_to_words, word_to_vec_map, project):

    """ increase vocabulary by splitting acronyms. Useful only for gloves. The acronyms splitting is  corpus-specific and might be changed """

    size_vocab = len(words_to_index)

    STOP = set(stopwords.words('english'))

    new_vocab = []
    s = size_vocab
    for sentence in project:

        words = (sentence.lower()).split()

        sentence = sentence.replace('ffwd','fast forward')
        sentence = sentence.replace('FFWD','fast forward')
        sentence = sentence.replace("KPI", "key performance metric")
        sentence = sentence.replace("KPIs", "key performance metric")
        sentence = sentence.replace('omnichannel','omni channel')
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence.lower())
        words = [w for w in tokens if w not in STOP]


        for w in words:
            new_vocab.append(w)
            if w not in word_to_vec_map:

                words_to_index[w] = s
                index_to_words[s] = w
                ran = np.random.rand(50)
                ran = ran*2-1
                word_to_vec_map[w] = ran
                

    return words_to_index, index_to_words, word_to_vec_map, new_vocab
