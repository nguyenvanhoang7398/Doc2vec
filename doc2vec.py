import numpy as np
import tensorflow as tf
import math, pickle, random

from collections import Counter, deque
from itertools import compress
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing_data import load_data, time_format, custom_tokenizer, process_new_docs

class doc2vec(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=128, window_size=8, concat=True, doc_embedding_size=128, word_embedding_size=128,
                 vocab_size=30000, document_size=50000, n_neg_samples=64,
                 learning_rate=1.0, doc_dataset_pickle='docs_dataset.pickle',
                 loss_type='sampled_softmax_loss',
                 optimize='Adagrad',
                 model='pv-dm', n_skip=None,
                 skip_window=None,
                 log='doc2vec_log.pickle',
                 path='/doc2vec_model/'):

        print('Creating new doc2vec instance')

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
        self.doc_idx, self.word_idx, self.count, self.dictionary, self.reverse_dictionary = load_data(
            doc_dataset_pickle)

        self._init_model()
        self._init_graph()

        self.sess = tf.Session(graph=self.graph)

    def _init_model(self):
        if self.model == 'pv-dbow':
            print('Initialize doc2vec pv-dbow model')
            self._init_graph = self._init_graph_dbow
            self.span = self.skip_window * 2 + 1
            self.generate_batch = self.generate_batch_dbow
            self.fit = self.fit_dbow

        elif self.model == 'pv-dm':
            print('Initialize doc2vec pv-dm model')

            self._init_graph = self._init_graph_dm
            self.span = self.window_size + 1
            self.generate_batch = self.generate_batch_dm
            self.fit = self.fit_dm

    def generate_batch_dbow(self):
        assert self.batch_size % self.n_skip == 0
        assert self.n_skip <= 2 * self.skip_window

        batch = np.ndarray(shape=(self.batch_size, self.n_skip), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        doc_labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        is_final = False

        buffer = deque(maxlen=self.span)
        buffer_doc = deque(maxlen=self.span)

        for _ in range(self.span):
            buffer.append(self.word_idx[self.data_index])
            buffer_doc.append(self.doc_idx[self.data_index])
            if self.data_index == (len(self.word_idx) - 1):
                is_final = True
                print('Final reached')
            self.data_index = (self.data_index + 1) % len(self.word_idx)

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

            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.n_skip])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.train_doc_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.word_embedding_size], -1.0, 1.0))

            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.doc_embedding_size], -1.0, 1.0))

            self.word_embed = tf.nn.embedding_lookup(self.word_embeddings, self.train_inputs)

            self.doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, self.train_doc_labels)

            print('Shapes:', self.word_embed.get_shape(), self.doc_embed.get_shape())

            embed = tf.concat(1, [self.word_embed, self.doc_embed])

            reduced_embed = tf.div(tf.reduce_sum(embed, 1), self.span)

            self.weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.doc_embedding_size],
                                    stddev=1.0 / math.sqrt(self.doc_embedding_size)))
            self.biases = tf.Variable(tf.zeros([self.vocab_size]))

            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, reduced_embed,
                                                  self.train_labels, self.n_neg_samples, self.vocab_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(self.weights, self.biases, reduced_embed,
                                      self.train_labels, self.n_neg_samples, self.vocab_size)
            self.loss = tf.reduce_mean(loss)

            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'GradDes':
                global_step = tf.Variable(0, trainable=False)
                modified_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                                    1000, 0.009, staircase=True)
                self.optimizer = tf.train.GradientDescentOptimizer(modified_learning_rate).minimize(loss)

            word_norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / word_norm

            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / doc_norm

            self.init_op = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

            print('Successfully _init_graph of pv-dbow model')

    def fit_dbow(self, n_epoch):

        print('Fitting pv-dbow model')

        sess = self.sess
        sess.run(self.init_op)

        try:
            with open(self.log, 'rb') as f:
                epoch, total_batches = pickle.load(f)
                epoch += 1
            print('STARTING WITH EPOCH:', epoch)
        except:
            epoch = 1
            total_batches = 0
        while epoch <= n_epoch:
            if epoch != 1:
                self.saver.restore(self.sess, self.path + "model.ckpt")

            epoch_loss = 0
            batches_run = 0
            is_final = False
            time0 = datetime.now()

            while not is_final:

                batch_inputs, batch_labels, batch_doc_labels, is_final = self.generate_batch_dbow()
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels,
                             self.train_doc_labels: batch_doc_labels}

                _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                epoch_loss += loss_val
                batches_run += 1
                if epoch == 1:
                    total_batches += 1

                if batches_run % 5000 == 0:
                    elapsed = (datetime.now() - time0)
                    second = elapsed.total_seconds()
                    time = time_format(second)
                    time_per_batch = float(second) / 5000
                    time0 = datetime.now()

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

            with open(self.log, 'wb') as f:
                pickle.dump((epoch, total_batches), f)
            epoch += 1

            self.data_index = 0
            self.finalized_word_embeddings = self.sess.run(self.normalized_word_embeddings)
            self.finalized_doc_embeddings = self.sess.run(self.normalized_doc_embeddings)

            print('Saving current session')
            self.save()
            print('Session saved successfully')

        return self

    def generate_batch_dm(self):
        assert self.batch_size % self.window_size == 0

        batch = np.ndarray(shape=(self.batch_size, self.span), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        is_final = False

        buffer = deque(maxlen=self.span)
        buffer_doc = deque(maxlen=self.span)

        for _ in range(self.span):
            buffer.append(self.word_idx[self.data_index])
            buffer_doc.append(self.doc_idx[self.data_index])
            if self.data_index == (len(self.word_idx) - 1):
                is_final = True
                print('Final reached')
            self.data_index = (self.data_index + 1) % len(self.word_idx)

        mask = [1] * self.span
        mask[-1] = 0
        i = 0

        while i < self.batch_size:
            if len(set(buffer_doc)) == 1:
                doc_id = buffer_doc[-1]
                batch[i, :] = list(compress(buffer, mask)) + [doc_id]
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

            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size + 1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.word_embedding_size], -1.0, 1.0))

            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.doc_embedding_size], -1.0, 1.0))

            if self.concat:
                combined_embed_vector_length = self.word_embedding_size * self.window_size + self.doc_embedding_size
            else:
                combined_embed_vector_length = self.word_embedding_size + self.doc_embedding_size

            self.weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, combined_embed_vector_length],
                                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            self.biases = tf.Variable(tf.zeros([self.vocab_size]))

            embed = []
            if self.concat:
                for j in range(self.window_size):
                    word_embed = tf.nn.embedding_lookup(self.word_embeddings, self.train_inputs[:, j])
                    embed.append(word_embed)
            else:
                # averaging word vectors
                word_embed = tf.zeros([self.batch_size, self.word_embedding_size])
                for j in range(self.window_size):
                    word_embed += tf.nn.embedding_lookup(self.word_embeddings, self.train_inputs[:, j])
                embed.append(word_embed)

            doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, self.train_inputs[:, self.window_size])
            embed.append(doc_embed)

            self.embed = tf.concat(1, embed)

            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.embed,
                                                  self.train_labels, self.n_neg_samples, self.vocab_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.embed,
                                      self.train_labels, self.n_neg_samples, self.vocab_size)
            self.loss = tf.reduce_mean(loss)

            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'GradDes':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            word_norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / word_norm

            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / doc_norm

            self.init_op = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

            print('Successfully _init_graph of dm-model')

    def fit_dm(self, n_epoch):

        print('Fitting pv-dbow model')

        sess = self.sess
        sess.run(self.init_op)

        try:
            with open(self.log, 'rb') as f:
                epoch, total_batches = pickle.load(f)
                epoch += 1
            print('STARTING WITH EPOCH:', epoch)
        except:
            epoch = 1
            total_batches = 0
        while epoch <= n_epoch:
            if epoch != 1:
                self.saver.restore(self.sess, self.path + "model.ckpt")

            epoch_loss = 0
            batches_run = 0
            is_final = False
            time0 = datetime.now()

            while not is_final:

                batch_inputs, batch_labels, is_final = self.generate_batch_dm()
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

                _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                epoch_loss += loss_val
                batches_run += 1
                if epoch == 1:
                    total_batches += 1

                if batches_run % 5000 == 0:
                    elapsed = (datetime.now() - time0)
                    second = elapsed.total_seconds()
                    time = time_format(second)
                    time_per_batch = float(second) / 5000
                    time0 = datetime.now()

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

            with open(self.log, 'wb') as f:
                pickle.dump((epoch, total_batches), f)
            epoch += 1

            self.data_index = 0
            self.finalized_word_embeddings = self.sess.run(self.normalized_word_embeddings)
            self.finalized_doc_embeddings = self.sess.run(self.normalized_doc_embeddings)

            print('Saving current session')
            self.save()
            print('Session saved successfully')

        return self

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
        self.new_data_index = 0
        self.new_word_idx, self.new_doc_idx, len_doc = process_new_docs(docs, self.dictionary, next_doc_idx=(self.doc_idx[-1] + 1))

        with open(self.path + 'doc_embeddings.pickle', 'rb') as f:
            d_embeddings = pickle.load(f)

        print('Fitting new document to existing pv-dbow model')

        self.sess.run(self.init_op)
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
                  'loss per batch: %.2f' % float(loss_val/batches_run))
            epoch += 1

            self.new_data_index = 0
            self.new_doc_embeddings = self.sess.run(self.normalized_doc_embeddings)

            self.saver.save(self.sess, self.path + predict_path + 'model.ckpt')
            with open(self.path + predict_path + 'word_embeddings.pickle', 'wb') as f:
                pickle.dump(self.new_doc_embeddings, f)

        return self.new_doc_embeddings[-len_doc:]

    def save(self):
        params = self.get_params()

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
    def restore(cls, save_path):

        with open(save_path + 'model_params.pickle', 'rb') as f:
            params = pickle.load(f)

        estimator = doc2vec(**params)
        estimator._restore(save_path + 'model.ckpt')

        estimator.word_embeddings = estimator.sess.run(estimator.normalized_word_embeddings)
        estimator.doc_embeddings = estimator.sess.run(estimator.normalized_doc_embeddings)

        return estimator