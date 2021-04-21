#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


from data_handling.vocab import BasicVocab


def one_hot(lbl, vocab, allow_unk=False):
    vec = [0] * len(vocab)
    if lbl not in vocab.token2ix and allow_unk:
        lbl = "UNK"

    lbl_ix = vocab.token2ix[lbl]
    vec[lbl_ix] = 1

    return vec


def set_to_vec(lbl_set, lbl_vocab):
    vec = [0] * len(lbl_vocab)
    for lbl_ix in range(len(lbl_vocab)):
        if lbl_vocab.ix2token[lbl_ix] in lbl_set:
            vec[lbl_ix] = 1

    return vec


def evaluate(model, validation_dataset, lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab):
    num_instances = 0
    num_correct = 0

    # for True class
    tp = 0
    fp = 0
    fn = 0

    for conj_instance in validation_dataset:
        input = conj_instance.to_svm_input(lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab)
        gold_output = conj_instance.to_svm_output()

        predicted_output = model.predict([input])[0]

        num_instances += 1
        if predicted_output == gold_output:
            num_correct += 1
        if predicted_output and gold_output:
            tp += 1
        elif predicted_output and not gold_output:
            fp += 1
        elif not predicted_output and gold_output:
            fn += 1

    # P, R, F1 for True class
    p = 100 * tp / (tp+fp)
    r = 100 * tp / (tp+fn)
    f1 = 2*p*r/(p+r)

    acc = num_correct / num_instances

    return p, r, f1, acc
