import numpy as np
import tensorflow as tf
import math, pickle, random

from collections import Counter, deque
from itertools import compress
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing_data import load_data, time_format, custom_tokenizer, process_new_docs

class doc2vec(BaseEstimator, TransformerMixin):

    '''
    Initialize a model from an existing 'doc_dataset_pickle' created by 'build_doc_dataset' function
    in preprocessing_data.py

    'batch_size' is the number of feature-label set in each training batch. By default, batch_size = 128

    'window_size' is the number of consecutive words used in DM model to predict the next word

    'concat' defines whether we concat the vectors of words in each training input (DM-concat model)
    or simply average them (DM no-concat model)

    'doc_embedding_size' is the dimensionality of feature vector for document

    'word_embedding_size' is the dimensionality of feature vector for word

    'vocab_size' is the number of unique word in vocabulary

    'document_size' is the number of document used to train

    'n_neg_samples' is the number of negative sample used in sampling

    'learning_rate' is the learning rate used in optimization part. By default, learning_rate = 0.025

    'doc_dataset.pickle' is the name of docs data set file (pickle) in the same directory. By default,
    doc_dataset_pickle = 'docs_dataset.pickle'

    'loss_type' defines the type of loss, either 'sampled_softmax_loss for softmax loss or
    'nce_loss' for 'noise contrastive estimation' loss. By default, 'loss_type' = 'nce_loss'

    'optimize' defines the type of optimizer, either 'Adagrad' for AdagradOptimizer or 'GradDes'
    for GradientDescentOptimizer. By default, 'optimize' = 'Adagrad'

    'model' defines the model of Doc2vec, either 'pv-dm' for Paragraph Vector - Distributed Memory
    or 'pv-dbow' for Paragraph Vector - Distributed Bags of Word. By default, 'model' = 'pv-dm'

    'n_skip' is the number of context words chosen with each predicted word (n_skip <= 2 * skip_window). By
    default, 'n_skip' = None

    'skip_window' is the maximum distance between context words chosen and their predicted word

    'log' is the name of log file to record the training. This file is saved in the same directory as
    doc2vec.py. By default, 'log' = 'doc2vec_log.pickle'

    'path' is the saving directory of the model. By default, path = '/doc2vec_model/'
    '''


    def __init__(self, batch_size=128, window_size=8, concat=True, doc_embedding_size=128, word_embedding_size=128,
                 vocab_size=30000, document_size=50000, n_neg_samples=64,
                 learning_rate=0.025, doc_dataset_pickle='docs_dataset.pickle',
                 loss_type='sampled_softmax_loss',
                 optimize='Adagrad',
                 model='pv-dm', n_skip=None,
                 skip_window=None,
                 log='doc2vec_log.pickle',
                 path='/doc2vec_model/'):

        print('Creating new doc2vec instance')

        # Initialize params
        self.batch_size = batch_size
        self.window_size = window_size
        self.concat = concat
        self.doc_embedding_size = doc_embedding_size
        self.word_embedding_size = word_embedding_size
        self.vocab_size = vocab_size
        self.document_size = document_size
        self.n_neg_samples = n_neg_samples
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.optimize = optimize
        self.model = model
        self.n_skip = n_skip
        self.skip_window = skip_window
        self.data_index = 0
        self.log = log
        self.path = path

        # Get doc_idx, word_idx, count, dictionary, reverse_dictionary by processing doc data set file
        self.doc_idx, self.word_idx, self.count, self.dictionary, self.reverse_dictionary = load_data(
            doc_dataset_pickle)

        # Initialize the chosen model
        self._init_model()

        # Initialize graph according to chosen model
        self._init_graph()

        # Create session based on created graph
        self.sess = tf.Session(graph=self.graph)

    def _init_model(self):
        if self.model == 'pv-dbow':

            # PV-DBOW model chosen
            print('Initialize doc2vec pv-dbow model')

            # Initialize more params
            self._init_graph = self._init_graph_dbow
            self.span = self.skip_window * 2 + 1
            self.generate_batch = self.generate_batch_dbow
            self.fit = self.fit_dbow

        elif self.model == 'pv-dm':

            # PV-DM model chosen
            print('Initialize doc2vec pv-dm model')

            # Initialize more params
            self._init_graph = self._init_graph_dm
            self.span = self.window_size + 1
            self.generate_batch = self.generate_batch_dm
            self.fit = self.fit_dm

    # Instance method to generate the next batch of PV-DBOW model
    def generate_batch_dbow(self):
        assert self.batch_size % self.n_skip == 0
        assert self.n_skip <= 2 * self.skip_window

        # Initialize numpy array of each variable
        batch = np.ndarray(shape=(self.batch_size, self.n_skip), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        doc_labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        is_final = False

        # Buffers for reading frame of both word and document with max length = self.span
        buffer = deque(maxlen=self.span)
        buffer_doc = deque(maxlen=self.span)

        # Fill buffer with word_idx and buffer_doc with doc_idx in range of self.span
        for _ in range(self.span):
            buffer.append(self.word_idx[self.data_index])
            buffer_doc.append(self.doc_idx[self.data_index])

            # Check if we have reached the end of document. If so, signal the model by is_final variable
            # to stop generating batch and end current epoch
            if self.data_index == (len(self.word_idx) - 1):
                is_final = True
                print('Final reached')

            # Use modulo to avoid IndexOutOfBoundException
            self.data_index = (self.data_index + 1) % len(self.word_idx)

        # Fill batch with train-inputs and labels
        i = 0
        while i < self.batch_size:

            # Check if all elements in the current buffer_doc are in the same doc
            if len(set(buffer_doc)) == 1:

                # Initialize the target at the center of the spanning frame
                target = self.skip_window
                targets_to_avoid = [self.skip_window]

                # Create smaller batch
                batch_temp = np.ndarray(shape=(self.n_skip), dtype=np.int32)

                for j in range(self.n_skip):
                    while target in targets_to_avoid:

                        # Randomly pick another target and append to 'targets ti avoid'
                        target = random.randint(0, self.span - 1)
                    targets_to_avoid.append(target)
                    batch_temp[j] = buffer[target]

                batch[i] = batch_temp

                # Label is the target word in the center of the spanning fram
                labels[i, 0] = buffer[self.skip_window]
                doc_labels[i, 0] = self.doc_idx[self.data_index]
                i += 1

            buffer.append(self.word_idx[self.data_index])
            buffer_doc.append(self.doc_idx[self.data_index])
            if self.data_index == (len(self.word_idx) - 1):
                is_final = True
                print('Final reached')
            self.data_index = (self.data_index + 1) % len(self.word_idx)

        return batch, labels, doc_labels, is_final

    def _init_graph_dbow(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Initialize tensorflow placeholder
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.n_skip])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.train_doc_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Initialize word_embeddings with vectors generated by uniform random
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.word_embedding_size], -1.0, 1.0))

            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.doc_embedding_size], -1.0, 1.0))

            # Use embedding_lookup Op from tensorflow
            self.word_embed = tf.nn.embedding_lookup(self.word_embeddings, self.train_inputs)

            self.doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, self.train_doc_labels)

            print('Shapes:', self.word_embed.get_shape(), self.doc_embed.get_shape())

            # Use concat Op from tensorflow to concat word_embedding vector and doc_embedding vector
            embed = tf.concat(1, [self.word_embed, self.doc_embed])

            reduced_embed = tf.div(tf.reduce_sum(embed, 1), self.span)

            # Initialize variable of weights and biases
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.doc_embedding_size],
                                    stddev=1.0 / math.sqrt(self.doc_embedding_size)))
            self.biases = tf.Variable(tf.zeros([self.vocab_size]))

            # Use either sampled_softmax_loss Op or nce_loss Op from tensorflow to find loss
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, reduced_embed,
                                                  self.train_labels, self.n_neg_samples, self.vocab_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(self.weights, self.biases, reduced_embed,
                                      self.train_labels, self.n_neg_samples, self.vocab_size)
            self.loss = tf.reduce_mean(loss)

            # Use optimizer Op from tensorflow corresponding to params
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'GradDes':
                global_step = tf.Variable(0, trainable=False)

                # Use exponential_decay to stabilize learning rate
                modified_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                                    1000, 0.009, staircase=True)
                self.optimizer = tf.train.GradientDescentOptimizer(modified_learning_rate).minimize(loss)

            # Normalize word_embeddings and doc_embeddings
            word_norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / word_norm

            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / doc_norm

            # Creat init_op to initalize all tensorflow variables
            self.init_op = tf.initialize_all_variables()

            # Create saver to save the current session of instance
            self.saver = tf.train.Saver()

            print('Successfully _init_graph of pv-dbow model')

    def fit_dbow(self, n_epoch):

        print('Fitting pv-dbow model')

        # Run init Op
        sess = self.sess
        sess.run(self.init_op)

        # Try to open log if exists, otherwise start the training with epoch = 1 and
        # count total number of batches('total_batches') from 0
        try:
            with open(self.log, 'rb') as f:

                # Load number of trained epoch ('epoch') and number of total batches
                # ('total_batches') from log
                epoch, total_batches = pickle.load(f)
                epoch += 1
            print('STARTING WITH EPOCH:', epoch)
        except:
            epoch = 1
            total_batches = 0
        while epoch <= n_epoch:

            # Restore saved model if not first epoch
            if epoch != 1:
                self.saver.restore(self.sess, self.path + "model.ckpt")

            # Initialize variables for new epoch
            epoch_loss = 0
            batches_run = 0
            is_final = False
            time0 = datetime.now()

            # Keep generating new batches while end of file hasn't been reached
            while not is_final:

                # Get variables from generate_batch_dbow and feed into current session
                batch_inputs, batch_labels, batch_doc_labels, is_final = self.generate_batch_dbow()
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels,
                             self.train_doc_labels: batch_doc_labels}

                # Run session to get loss value ('loss_val')
                _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                epoch_loss += loss_val
                batches_run += 1

                # Keep track of 'total_batches' by increment if first epoch
                if epoch == 1:
                    total_batches += 1

                # Check process after every 5000 batches_run
                if batches_run % 5000 == 0:

                    # Formatting elapsed time
                    elapsed = (datetime.now() - time0)
                    second = elapsed.total_seconds()
                    time = time_format(second)
                    time_per_batch = float(second) / 5000
                    time0 = datetime.now()

                    # Total number of batches is not ready in the first epoch
                    if epoch != 1:
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch,
                              '| Average batch loss:', epoch_loss / (batches_run), '| Time:', time[0], 'hours',
                              time[1], 'minutes', "%.2f" % time[2], 'seconds.')
                        remaining_time = time_format(
                            (total_batches * time_per_batch * (n_epoch - epoch + 1)) - batches_run * time_per_batch)
                        print('Estimated remaining time:', remaining_time[0], 'hours', remaining_time[1], 'minutes',
                              "%.2f" % remaining_time[2], 'seconds.')
                        print('Index read:', self.data_index, '/', len(self.word_idx))
                    else:
                        print('Batch run:', batches_run, '/ NaN', '| Epoch:', epoch,
                              '| Average batch loss:', epoch_loss / (batches_run), '| Time:', time[0], 'hours',
                              time[1], 'minutes', "%.2f" % time[2], 'seconds.')
                        print('Estimated remaining time is only available after first epoch.')
                        print('Index read:', self.data_index, '/', len(self.word_idx))

            print('Epoch', epoch, 'completed out of', n_epoch, 'loss:', epoch_loss)

            # Write to log file 'epoch' and 'total_batches'
            with open(self.log, 'wb') as f:
                pickle.dump((epoch, total_batches), f)
            epoch += 1

            # Reset data_index to 0
            self.data_index = 0

            # Run session to get finalized_word_embeddings and finalized_docs_embeddings
            self.finalized_word_embeddings = self.sess.run(self.normalized_word_embeddings)
            self.finalized_doc_embeddings = self.sess.run(self.normalized_doc_embeddings)

            # Save current session
            print('Saving current session')
            self.save()
            print('Session saved successfully')

        return self

    def generate_batch_dm(self):
        assert self.batch_size % self.window_size == 0

        # Initialize numpy array of each variable
        batch = np.ndarray(shape=(self.batch_size, self.span), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        is_final = False

        # Buffer for reading frame of both word and document with max length = self.span
        buffer = deque(maxlen=self.span)
        buffer_doc = deque(maxlen=self.span)

        # FIll buffer with word_idx and buffer_doc with doc_idx in range of self.span
        for _ in range(self.span):
            buffer.append(self.word_idx[self.data_index])
            buffer_doc.append(self.doc_idx[self.data_index])

            # Check if we have reached the end of document. If so, signal the model by is_final variable
            # to stop generating batch and end current epoch
            if self.data_index == (len(self.word_idx) - 1):
                is_final = True
                print('Final reached')

            # Use modulo to avoid IndexOutOfBoundException
            self.data_index = (self.data_index + 1) % len(self.word_idx)

        # Mask the whole reading frame with 1 and the last element with 0
        mask = [1] * self.span
        mask[-1] = 0
        i = 0

        # Fill batch with train-inputs and labels
        while i < self.batch_size:

            # Check if all elements in the current buffer_doc are in the same doc
            if len(set(buffer_doc)) == 1:
                doc_id = buffer_doc[-1]

                # Form batch inputs with the words previous to predict words and doc_id
                batch[i, :] = list(compress(buffer, mask)) + [doc_id]

                # Label is the predict word (the last word in reading frame)
                labels[i, 0] = buffer[-1]
                i += 1

            buffer.append(self.word_idx[self.data_index])
            buffer_doc.append(self.doc_idx[self.data_index])
            if self.data_index == (len(self.word_idx) - 1):
                is_final = True
                print('Final reached')
            self.data_index = (self.data_index + 1) % len(self.word_idx)

        return batch, labels, is_final

    def _init_graph_dm(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Initialize tensorflow placeholder
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size + 1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Initialize word_embeddings with vectors generated by uniform random
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.word_embedding_size], -1.0, 1.0))

            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.doc_embedding_size], -1.0, 1.0))

            if self.concat:
                combined_embed_vector_length = self.word_embedding_size * self.window_size + self.doc_embedding_size
            else:
                combined_embed_vector_length = self.word_embedding_size + self.doc_embedding_size

            # Initialize variables of weights and biases
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, combined_embed_vector_length],
                                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            self.biases = tf.Variable(tf.zeros([self.vocab_size]))

            embed = []

            # Use embedding_lookup Op from tensorflow
            if self.concat:
                # If concat then append every word vector to embed
                for j in range(self.window_size):
                    word_embed = tf.nn.embedding_lookup(self.word_embeddings, self.train_inputs[:, j])
                    embed.append(word_embed)
            else:
                # else average word vectors
                word_embed = tf.zeros([self.batch_size, self.word_embedding_size])
                for j in range(self.window_size):
                    word_embed += tf.nn.embedding_lookup(self.word_embeddings, self.train_inputs[:, j])
                embed.append(word_embed)

            doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, self.train_inputs[:, self.window_size])
            embed.append(doc_embed)

            self.embed = tf.concat(1, embed)

            # Use either sampled_softmax_loss Op or nce_loss Op from tensorflow to find loss
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.embed,
                                                  self.train_labels, self.n_neg_samples, self.vocab_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.embed,
                                      self.train_labels, self.n_neg_samples, self.vocab_size)
            self.loss = tf.reduce_mean(loss)

            # Use optimizer Op from tensorflow corresponding to params
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'GradDes':
                # Use exponential_decay to stabilize learning rate
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # Normalize word_embeddings and doc_embeddings
            word_norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / word_norm

            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / doc_norm

            # Create init_op to initialize all tensorflow variables
            self.init_op = tf.initialize_all_variables()

            # Create saver to save current session of instance
            self.saver = tf.train.Saver()

            print('Successfully _init_graph of dm-model')

    def fit_dm(self, n_epoch):

        print('Fitting pv-dbow model')

        # Run init Op
        sess = self.sess
        sess.run(self.init_op)

        # Try to open log if exists, otherwise start the training with epoch = 1 and
        # count total number of batches('total_batches') from 0
        try:
            with open(self.log, 'rb') as f:

                # Load number of trained epoch ('epoch') and number of total batches
                # ('total_batches') from log
                epoch, total_batches = pickle.load(f)
                epoch += 1
            print('STARTING WITH EPOCH:', epoch)
        except:
            epoch = 1
            total_batches = 0
        while epoch <= n_epoch:

            # Restore saved model if not first epoch
            if epoch != 1:
                self.saver.restore(self.sess, self.path + "model.ckpt")

            # Initialize variables for new epoch
            epoch_loss = 0
            batches_run = 0
            is_final = False
            time0 = datetime.now()

            # Keep generating new batches while end of file hasn't been reached
            while not is_final:

                # Get variables from generate_batch_dbow and feed into current session
                batch_inputs, batch_labels, is_final = self.generate_batch_dm()
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

                # Run session to get loss value ('loss_val')
                _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                epoch_loss += loss_val
                batches_run += 1

                # Keep track of 'total_batches' by increment if first epoch
                if epoch == 1:
                    total_batches += 1

                # Check process after every 5000 batches_run
                if batches_run % 5000 == 0:
                    elapsed = (datetime.now() - time0)
                    second = elapsed.total_seconds()
                    time = time_format(second)
                    time_per_batch = float(second) / 5000
                    time0 = datetime.now()

                    # Total number of batches is not ready in the first epoch
                    if epoch != 1:
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch,
                              '| Average batch loss:', epoch_loss / (batches_run), '| Time:', time[0], 'hours',
                              time[1], 'minutes', "%.2f" % time[2], 'seconds.')
                        remaining_time = time_format(
                            (total_batches * time_per_batch * (n_epoch - epoch + 1)) - batches_run * time_per_batch)
                        print('Estimated remaining time:', remaining_time[0], 'hours', remaining_time[1], 'minutes',
                              "%.2f" % remaining_time[2], 'seconds.')
                        print('Index read:', self.data_index, '/', len(self.word_idx))
                    else:
                        print('Batch run:', batches_run, '/ NaN', '| Epoch:', epoch,
                              '| Average batch loss:', epoch_loss / (batches_run), '| Time:', time[0], 'hours',
                              time[1], 'minutes', "%.2f" % time[2], 'seconds.')
                        print('Estimated remaining time is only available after first epoch.')
                        print('Index read:', self.data_index, '/', len(self.word_idx))

            print('Epoch', epoch, 'completed out of', n_epoch, 'loss:', epoch_loss)

            # Write to log file 'epoch' and 'total_batches'
            with open(self.log, 'wb') as f:
                pickle.dump((epoch, total_batches), f)
            epoch += 1

            # Reset data_index to 0
            self.data_index = 0

            # Run session to get finalized_word_embeddings and finalized_docs_embeddings
            self.finalized_word_embeddings = self.sess.run(self.normalized_word_embeddings)
            self.finalized_doc_embeddings = self.sess.run(self.normalized_doc_embeddings)

            # Save current session
            print('Saving current session')
            self.save()
            print('Session saved successfully')

        return self

    # Instance method to generate the next batch of PV-DBOW model for new document
    def generate_batch_dbow_new_doc(self):

        batch = np.ndarray(shape=(self.batch_size, self.n_skip), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        doc_labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        is_final = False

        buffer = deque(maxlen=self.span)
        buffer_doc = deque(maxlen=self.span)

        for _ in range(self.span):
            buffer.append(self.new_word_idx[self.new_data_index])
            buffer_doc.append(self.new_doc_idx[self.new_data_index])
            if self.new_data_index == (len(self.new_word_idx) - 1):
                is_final = True
                print('Final reached')
            self.new_data_index = (self.new_data_index + 1) % len(self.new_word_idx)

        i = 0
        while i < self.batch_size:
            if len(set(buffer_doc)) == 1:
                target = self.skip_window
                targets_to_avoid = [self.skip_window]
                batch_temp = np.ndarray(shape=(self.n_skip), dtype=np.int32)

                for j in range(self.n_skip):
                    while target in targets_to_avoid:
                        target = random.randint(0, self.span - 1)
                    targets_to_avoid.append(target)
                    batch_temp[j] = buffer[target]

                batch[i] = batch_temp
                labels[i, 0] = buffer[self.skip_window]
                doc_labels[i, 0] = self.new_doc_idx[self.new_data_index]
                i += 1

            buffer.append(self.new_word_idx[self.new_data_index])
            buffer_doc.append(self.new_doc_idx[self.new_data_index])
            if self.new_data_index == (len(self.new_word_idx) - 1):
                is_final = True
                print('Final reached')
            self.new_data_index = (self.new_data_index + 1) % len(self.new_word_idx)

        return batch, labels, doc_labels, is_final

    def fit_dbow_new_doc(self, docs, n_epoch, predict_path):

        # Initialize new_data_index to 0
        self.new_data_index = 0

        # Update some params by calling process_new_docs function to read new documents
        self.new_word_idx, self.new_doc_idx, len_doc = process_new_docs(docs, self.dictionary, next_doc_idx=(self.doc_idx[-1] + 1))

        # Open and import current doc_embeddings
        with open(self.path + 'doc_embeddings.pickle', 'rb') as f:
            d_embeddings = pickle.load(f)

        print('Fitting new document to existing pv-dbow model')

        self.sess.run(self.init_op)

        # Initialize new vectors for doc_embeddings of new document by uniform random
        for i in range(len_doc):
            d_embeddings = np.insert(arr=d_embeddings, obj=(self.document_size + i),
                                     values=np.random.uniform(-1.0, 1.0, size=self.doc_embedding_size), axis=0)

        epoch = 1
        while epoch <= n_epoch:
            if epoch == 1:
                self.saver.restore(self.sess, self.path + "model.ckpt")
            else:
                self.saver.restore(self.sess, self.path + predict_path + "model.ckpt")
            is_final = False
            epoch_loss = 0
            batches_run = 0

            # Create resize-op to resize the current doc_embeddings Variable with
            # new dimensions for incoming documents
            resize_op = tf.assign(self.doc_embeddings, d_embeddings, validate_shape=False)
            self.sess.run(resize_op)

            while not is_final:
                batch_inputs, batch_labels, batch_doc_labels, is_final = self.generate_batch_dbow_new_doc()
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels,
                             self.train_doc_labels: batch_doc_labels}

                _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                epoch_loss += loss_val
                batches_run += 1

            print('Epoch', epoch, 'completed out of', n_epoch, 'loss:', epoch_loss, 'batches_run:', batches_run,
                  'loss per batch: %.2f' % float(epoch_loss/batches_run))
            epoch += 1

            self.new_data_index = 0
            self.new_doc_embeddings = self.sess.run(self.normalized_doc_embeddings)

            self.saver.save(self.sess, self.path + predict_path + 'model.ckpt')

            # Save new doc_embeddings pickle file
            with open(self.path + predict_path + 'doc_embeddings.pickle', 'wb') as f:
                pickle.dump(self.new_doc_embeddings, f)

        # Return the vectors of new documents
        return self.new_doc_embeddings[-len_doc:]

    # Save current session of class instance
    def save(self):
        params = self.get_params()

        # Save session and params
        model_path = self.saver.save(self.sess, self.path + 'model.ckpt')

        with open(self.path + 'word_embeddings.pickle', 'wb') as f:
            pickle.dump(self.word_embeddings, f)
        with open(self.path + 'doc_embeddings.pickle', 'wb') as f:
            pickle.dump(self.doc_embeddings, f)
        with open(self.path + 'model_params.pickle', 'wb') as f:
            pickle.dump(params, f)

        print('Model saved in path: %s' % model_path)

        return model_path

    def _restore(self, model_path):
        with self.graph.as_default():
            self.saver.restore(self.sess, model_path)

    @classmethod
    # Restore any instance of class by estimator
    def restore(cls, save_path):

        with open(save_path + 'model_params.pickle', 'rb') as f:
            params = pickle.load(f)

        estimator = doc2vec(**params)
        estimator._restore(save_path + 'model.ckpt')

        estimator.word_embeddings = estimator.sess.run(estimator.normalized_word_embeddings)
        estimator.doc_embeddings = estimator.sess.run(estimator.normalized_doc_embeddings)

        return estimator