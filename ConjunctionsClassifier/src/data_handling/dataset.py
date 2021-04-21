#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


import csv
from torch.utils.data import Dataset, DataLoader
from data_handling.conjunction_instance import ConjunctionInstance


class ConjunctionPropagationDataset(Dataset):
    """Class for representing a dataset of conjunction instances. The individual objects contained
       within should be of type ConjunctionInstance."""
    def __init__(self):
        self.instances = list()

    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def append_instance(self, sent):
        """Add one sentence to the dataset."""
        self.instances.append(sent)

    def get_svm_data(self, lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab):
        inputs = [inst.to_svm_input(lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab) for inst in self]
        outputs = [inst.to_svm_output() for inst in self]

        return inputs, outputs

    @staticmethod
    def from_corpus_file(corpus_filename):
        """Read in a dataset from a file in TSV format."""
        dataset = ConjunctionPropagationDataset()

        with open(corpus_filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:

                sentence = row[0].strip().split(" ")
                conj_head_ix = int(row[1])
                conj_head_tok = row[2]
                conj_head_upos = row[3]
                conj_head_feats = set(row[4].split(","))
                conj_dep_ix = int(row[5])
                conj_dep_tok = row[6]
                conj_dep_upos = row[7]
                conj_dep_feats = set(row[8].split(","))
                candidate_ix = int(row[9])
                candidate_tok = row[10]
                candidate_upos = row[11]
                candidate_feats = set(row[12].split(","))
                relation_label = row[13]
                dependency_direction = row[14]
                linear_directions = row[15]
                already_has_dependent_type = row[16]
                outgoing_edges_candidate_gov = set(row[17].split(","))
                outgoing_edges_candidate_dep = set(row[18].split(","))
                dependency_head_rel = row[19]
                num_coord_items = row[20]
                propagation = 1 if row[21] == "yes" else 0

                instance = ConjunctionInstance(sentence,
                                               conj_head_ix, conj_head_tok, conj_head_upos, conj_head_feats,
                                               conj_dep_ix, conj_dep_tok, conj_dep_upos, conj_dep_feats,
                                               candidate_ix, candidate_tok, candidate_upos, candidate_feats,
                                               relation_label, dependency_direction, linear_directions,
                                               already_has_dependent_type,
                                               outgoing_edges_candidate_gov, outgoing_edges_candidate_dep,
                                               dependency_head_rel, num_coord_items,
                                               propagation)

                dataset.append_instance(instance)

        return dataset


if __name__ == "__main__":
    # For testing purposes only

    dataset = ConjunctionPropagationDataset.from_corpus_file("/home/pic7rng/conjunctionsclassifier/propagation_train.tsv")
    for stuff in dataset:
        print(stuff.relation_label)
    #data_loader = DataLoader(dataset, batch_size=3, shuffle=True)
    #for instance_batch in data_loader:
    #    print(instance_batch)
    #pass
