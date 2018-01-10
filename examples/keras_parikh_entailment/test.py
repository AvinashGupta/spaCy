# -*- encoding:utf-8 -*-
# coding: utf-8

from __future__ import division, unicode_literals, print_function
import spacy
import os
import datetime

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import plac
from pathlib import Path
import ujson as json
import numpy
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import model_from_json, load_model

from spacy_hook import get_embeddings, get_word_ids
from spacy_hook import create_similarity_pipeline

from keras_decomposable_attention import build_model
import boto3

try:
    import cPickle as pickle
except ImportError:
    import pickle

def test():
    dir_ = os.path.expanduser('~')+ '/data/temp/'
    config_file = open(dir_+'config.json', 'r').read()

    spacy.util.set_data_path('/Users/avinashgupta/data/spacy/complete_data/data')
    nlp = spacy.load('en')
    assert nlp.path is not None

    max_length, nr_hidden = 100, 100
    tree_truncate=False
    gru_encode=False
    max_length=100
    nr_hidden=100
    dropout=0.2
    learn_rate=0.001
    batch_size=100
    nr_epoch=300

    shape = (max_length, nr_hidden, 3)
    settings = {
        'lr': learn_rate,
        'dropout': dropout,
        'batch_size': batch_size,
        'nr_epoch': nr_epoch,
        'tree_truncate': tree_truncate,
        'gru_encode': gru_encode
    }
    texts = [
      u'yes',
      u'no'
    ]

    model = model_from_json(config_file)

    model.load_weights(dir_+'model.hdf5')

    texts = get_word_ids(list(nlp.pipe(texts, n_threads=20, batch_size=20000)),
                 max_length=shape[0],
                 rnn_encode=settings['gru_encode'],
                 tree_truncate=settings['tree_truncate'])
    pred = model.predict([numpy.array([texts[0]]), numpy.array([texts[0]])])
    print('Predicts:')
    print(pred)




if __name__ == '__main__':
  test()