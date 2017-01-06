from preprocessing_data import process_all_words, shuffle, build_doc_dataset, create_docs_pickle, create_train_test_data
from doc2vec import doc2vec
from SVM import SVM
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np

# process_all_words('train-pos.txt', 'train-neg.txt', 'test-pos.txt', 'test-neg.txt', 'labeled-data.csv')
# shuffle('labeled-data.csv', 'shuffled-labeled-data.csv')
# docs = create_docs_pickle('shuffled-labeled-data.csv', 'docs.pickle')
# build_doc_dataset(docs)

'''
# PV-DM model
D2V = doc2vec(optimize='Adagrad', loss_type='sampled_softmax_loss', log='doc2vec_log.pickle', model='pv-dm', path='/doc2vec_model/')
D2V.fit(n_epoch=25)

train_inputs, train_labels, test_inputs, test_labels = create_train_test_data('/doc2vec_model/doc_embeddings.pickle',
                                                                              'shuffled-labeled-data.csv', test_size=0.1)
'''

# PV-DBOW model
D2V = doc2vec(optimize='GradDes', learning_rate=0.025, loss_type='sampled_softmax_loss', log="doc2vecV2_log.pickle", model='pv-dbow', n_skip=2, skip_window=2, path="/doc2vecV2_model/")
D2V.fit(n_epoch=21)

train_inputs, train_labels, test_inputs, test_labels = create_train_test_data('/doc2vecV2_model/doc_embeddings.pickle', 'shuffled-labeled-data.csv', test_size=0.1)


'''
# SVM classifier from sklearn
clf = svm.SVC()
clf.fit(train_inputs, train_labels)
confidence = clf.score(test_inputs, test_labels)
print(confidence) # ~0.65
'''


# Logistic regression from sklearn
clf = LogisticRegression()
clf.fit(train_inputs, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print(clf.score(test_inputs, test_labels)) # DM - 25 epochs: ~0.674, DBOW - 20 epochs: ~0.83868

with open('reviews.txt', 'r') as f:
    new_doc_vectors = D2V.fit_dbow_new_doc(f, n_epoch=20, predict_path='predict/')
    print(new_doc_vectors)
    prediction = clf.predict(new_doc_vectors)
    print(prediction)

'''
# Customized SVM class - too long: ~ 4 hours

clf = SVM(kernel='gaussian_kernel')
clf.fit(np.array(train_inputs), np.array(train_labels))

test_predict = clf.predict(test_inputs)
correct = np.sum(test_predict == test_labels)
print("%d out of %d predictions correct" % (correct, len(test_predict)))
print("Accuracy:", correct/len(test_predict) * 100, "%")
'''

