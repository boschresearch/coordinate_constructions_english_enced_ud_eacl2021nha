#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


from sklearn import svm
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_handling.dataset import ConjunctionPropagationDataset
from data_handling.vocab import BasicVocab
from svm_utils import evaluate


def main(model_path):
    model = svm.SVC(kernel="poly", degree=2, gamma="scale")

    train_dataset = ConjunctionPropagationDataset.from_corpus_file("../propagation_train_no-aux.tsv")
    dev_dataset = ConjunctionPropagationDataset.from_corpus_file("../propagation_dev_no-aux.tsv")

    lemma_vocab = BasicVocab("../vocabs/lemmas.txt")
    dep_label_vocab = BasicVocab("../vocabs/dep_labels.txt")
    upos_label_vocab = BasicVocab("../vocabs/upos_labels.txt")
    feats_vocab = BasicVocab("../vocabs/feat_labels.txt")
    num_items_vocab = BasicVocab("../vocabs/num_coord_items.txt")

    inputs, outputs = train_dataset.get_svm_data(lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab)
    print("Number of instances:", len(outputs))

    model.fit(inputs, outputs)
    print("Model training finished.")

    p, r, f1, acc = evaluate(model, dev_dataset, lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab)
    print("Accuracy on dev set: {:.2f}%". format(acc*100))
    print("For True class: precision={:.2f} recall={:.2f} F1={:.2f}".format(p, r, f1))
    print("\n{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(acc*100, p, r, f1))

    print("Saving SVM model to {} ...".format(model_path))
    dump(model, model_path)
    print("done.")


if __name__ == "__main__":
    main("../saved_models/svm.joblib")

