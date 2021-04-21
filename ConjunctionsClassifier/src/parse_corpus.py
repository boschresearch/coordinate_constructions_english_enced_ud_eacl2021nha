#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


import pyconll
import torch
import code
import sys

from sklearn import svm
from joblib import load

from data_handling.vocab import BasicVocab
from models.classifier import ConjunctionClassifier
from trainer.trainer import Trainer
from propagations import propagations


lemma_vocab = BasicVocab("../vocabs/lemmas.txt")
dep_label_vocab = BasicVocab("../vocabs/dep_labels.txt")
upos_label_vocab = BasicVocab("../vocabs/upos_labels.txt")
feats_vocab = BasicVocab("../vocabs/feat_labels.txt")
num_items_vocab = BasicVocab("../vocabs/num_coord_items.txt")


def load_model(model_path):
    if model_path.endswith(".pth"):
        model = ConjunctionClassifier()
        trainer = Trainer(model, None, None)  # Need this for model loading
        trainer._load_model(model_path)
        return model
    elif model_path.endswith(".joblib"):
        model = load(model_path)
        assert isinstance(model, svm.SVC)
        return model
    elif model_path == "baseline":
        return None
    else:
        raise Exception("Model must be SVM or BERT-based")


def get_output(model, prop_instance):
    if isinstance(model, svm.SVC):
        model_input = prop_instance.to_svm_input(lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab)
        model_output = model.predict([model_input])[0]
    elif isinstance(model, ConjunctionClassifier):
        model.eval()
        model_output = model([prop_instance]).squeeze()
        model_output = int(torch.argmax(model_output))
    elif model is None:
        model_output = 1  # "Always" baseline
    else:
        raise Exception("Model must be SVM or BERT-based")

    model_output = "yes" if model_output == 1 else "no"
    return model_output


def main(model, corpus_filename):
    for sentence in pyconll.load_from_file(corpus_filename):
        for prop_inst in propagations(sentence):
            model_output = get_output(model, prop_inst)

            try:
                if model_output != prop_inst.propagation:
                    conj_head_id = prop_inst.conj_head_id
                    conj_dep_id = prop_inst.conj_dep_id
                    candidate_id = prop_inst.candidate_id

                    if model_output == "yes":
                        assert prop_inst.propagation == "no"
                        if prop_inst.dependency_direction == "incoming":
                            sentence[conj_dep_id].deps[candidate_id] = sentence[conj_head_id].deps[candidate_id]
                        elif prop_inst.dependency_direction == "outgoing":
                            sentence[candidate_id].deps[conj_dep_id] = sentence[candidate_id].deps[conj_head_id]

                    else:
                        assert model_output == "no"
                        assert prop_inst.propagation == "yes"
                        if prop_inst.dependency_direction == "incoming":
                            del sentence[conj_dep_id].deps[candidate_id]
                        elif prop_inst.dependency_direction == "outgoing":
                            del sentence[candidate_id].deps[conj_dep_id]
            except KeyError:
                print(sentence.text)
                code.interact(local=locals())
                exit(1)

        print(sentence.conll())
        print()


if __name__ == '__main__':
    model_path = sys.argv[1]
    corpus_filename = sys.argv[2]

    model = load_model(model_path)

    main(model, corpus_filename)
