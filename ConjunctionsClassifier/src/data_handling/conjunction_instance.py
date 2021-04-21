#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


from svm_utils import one_hot, set_to_vec


class ConjunctionInstance:
    """An instance of this class represents one (positive or negative) example
       of conjunction propagation."""

    def __init__(self, sentence, conj_head_ix, conj_head_lemma, conj_head_upos, conj_head_feats,
                 conj_dep_ix, conj_dep_lemma, conj_dep_upos, conj_dep_feats,
                 candidate_ix, candidate_lemma, candidate_upos, candidate_feats,
                 relation_label, dependency_direction, linear_directions,
                 already_has_dependent_type, outgoing_edges_candidate_gov, outgoing_edges_candidate_dep,
                 dependency_head_rel, num_coord_items, propagation,
                 conj_head_id=None, conj_dep_id=None, candidate_id=None):
        self.sentence = sentence

        # Token features
        self.conj_head_id = conj_head_id
        self.conj_head_ix = conj_head_ix
        self.conj_head_lemma = conj_head_lemma
        self.conj_head_upos = conj_head_upos
        self.conj_head_feats = conj_head_feats

        self.conj_dep_id = conj_dep_id
        self.conj_dep_ix = conj_dep_ix
        self.conj_dep_lemma = conj_dep_lemma
        self.conj_dep_upos = conj_dep_upos
        self.conj_dep_feats = conj_dep_feats

        self.candidate_id = candidate_id
        self.candidate_ix = candidate_ix
        self.candidate_lemma = candidate_lemma
        self.candidate_upos = candidate_upos
        self.candidate_feats = candidate_feats

        # Tree features
        self.relation_label = relation_label
        self.dependency_direction = dependency_direction
        self.linear_directions = linear_directions
        self.already_has_dependent_type = already_has_dependent_type
        self.outgoing_edges_candidate_gov = outgoing_edges_candidate_gov
        self.outgoing_edges_candidate_dep = outgoing_edges_candidate_dep
        self.dependency_head_rel = dependency_head_rel
        self.num_coord_items = str(num_coord_items)

        # Output
        self.propagation = propagation

    def to_svm_input(self, lemma_vocab, upos_label_vocab, feats_vocab, dep_label_vocab, num_items_vocab):
        # Transform this instance into the input format of an SVM.
        # Features:
        #   * Token features of conj head, conj dep, and potential propagation target:
        #       + Lemma
        #       + POS tag
        #       + Morph. features
        #   * Tree features:
        #       + Dependency label
        #       + Dependency direction
        #       + Propagation target already has dependent of same type? (Only relevant for propagated dependents)
        #       + Set of outgoing dependency types for the candidate governor and dependent
        #       + Dependency type governing the head of the dependency being propagated
        #       + Whether the linear direction of the candidate propagated dependency is the same as the linear
        #         direction of the dependency being propagated (possible values being both-left, both-right, and
        #         differing-directions)
        #       + Number of coordinated items in the coordination expressed as a binary feature
        #         (i.e. one binary feature for every discrete value).
        vec = []

        # Token features (conj head)
        # Lemma
        #vec += one_hot(self.conj_head_lemma, lemma_vocab, allow_unk=True)

        # POS tag
        #vec += one_hot(self.conj_head_upos, upos_label_vocab)

        # Morph features
        vec += set_to_vec(self.conj_head_feats, feats_vocab)

        # Token features (conj dep)
        # Lemma
        #vec += one_hot(self.conj_dep_lemma, lemma_vocab, allow_unk=True)

        # POS tag
        #vec += one_hot(self.conj_dep_upos, upos_label_vocab)

        # Morph features
        vec += set_to_vec(self.conj_dep_feats, feats_vocab)

        # Token features (candidate)
        # Lemma
        #vec += one_hot(self.candidate_lemma, lemma_vocab, allow_unk=True)

        # POS tag
        #vec += one_hot(self.candidate_upos, upos_label_vocab)

        # Morph features
        vec += set_to_vec(self.candidate_feats, feats_vocab)

        # Tree features
        # Dependency label
        vec += one_hot(self.relation_label, dep_label_vocab)

        # Dependency direction
        vec += [1] if self.dependency_direction == "incoming" else [0]

        # Linear direction of existing dependency vs. candidate propagated dependency
        vec += [1, 0, 0] if self.linear_directions == "both-left" else \
               [0, 1, 0] if self.linear_directions == "both-right" else \
               [0, 0, 1]

        # Already has dependent of same type?
        vec += [1] if self.already_has_dependent_type else [0]

        # Outgoing dependency types of candidate governor and dependent
        vec += set_to_vec(self.outgoing_edges_candidate_gov, dep_label_vocab)
        vec += set_to_vec(self.outgoing_edges_candidate_dep, dep_label_vocab)

        # Dependency type governing the head of the dependency being propagated
        # Useless; leave commented!
        #vec += one_hot(self.dependency_head_rel, dep_label_vocab)

        # Number of items in coordination
        vec += one_hot(self.num_coord_items, num_items_vocab, allow_unk=True)

        return vec

    def to_svm_output(self):
        # Transform this instance into the output format of an SVM: a binary decision whether to propagate or not
        # 0: no propagation
        # 1: propagation
        return self.propagation
