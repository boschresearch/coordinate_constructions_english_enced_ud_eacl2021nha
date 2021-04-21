#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


import pyconll
import sys
import csv

from data_handling.conjunction_instance import ConjunctionInstance


def write_instance(inst, writer):
    writer.writerow([" ".join(inst.sentence),
                     inst.conj_head_ix, inst.conj_head_lemma, inst.conj_head_upos, ",".join(inst.conj_head_feats),
                     inst.conj_dep_ix, inst.conj_dep_lemma, inst.conj_dep_upos, ",".join(inst.conj_dep_feats),
                     inst.candidate_ix, inst.candidate_lemma, inst.candidate_upos, ",".join(inst.candidate_feats),
                     inst.relation_label, inst.dependency_direction, inst.linear_directions, inst.already_has_dependent_type,
                     ",".join(inst.outgoing_edges_candidate_gov), ",".join(inst.outgoing_edges_candidate_dep),
                     inst.dependency_head_rel, inst.num_coord_items, inst.propagation])


def propagations(sentence):
    ignore_labels = {"root", "conj", "punct", "cc"}
    full_labels_to_keep = {'acl', 'acl:relcl', 'advcl', 'advmod', 'aux',
                           'aux:pass', 'nmod', 'nsubj', 'nsubj:pass',
                           'nsubj:xsubj', 'obl', 'obj', 'xcomp', 'ccomp'}

    sent_toks = [tok.form for tok in sentence]

    # Create id-to-ix mapping
    id_to_ix = dict()
    ix = 0
    for token in sentence:
        id_to_ix[token.id] = ix
        ix += 1

    conjunctions = set()

    for token in sentence:
        if token.deprel == 'conj':
            conjunctions.add((token.head, token.id))

    for (conj_head_id, conj_dep_id) in conjunctions:
        conj_head_ix = id_to_ix[conj_head_id]
        conj_head_lemma = sentence[conj_head_id].lemma
        conj_head_upos = sentence[conj_head_id].upos
        conj_head_feats = extract_relevant_features(sentence, conj_head_id)

        conj_dep_ix = id_to_ix[conj_dep_id]
        conj_dep_lemma = sentence[conj_dep_id].lemma
        conj_dep_upos = sentence[conj_dep_id].upos
        conj_dep_feats = extract_relevant_features(sentence, conj_dep_id)

        incoming_links = sentence[conj_head_id].deps.items()
        outgoing_links = dependents(sentence, conj_head_id)

        for (candidate_id, dep) in incoming_links:
            if dep[0] in ignore_labels:
                continue

            candidate_ix = id_to_ix[candidate_id]
            candidate_lemma = sentence[candidate_id].lemma
            candidate_upos = sentence[candidate_id].upos
            candidate_feats = extract_relevant_features(sentence, candidate_id)

            outgoing_edges_candidate_gov = {deprel for _, deprel in basic_dependents(sentence, candidate_id)}
            outgoing_edges_candidate_dep = {deprel for _, deprel in basic_dependents(sentence, conj_dep_id)}

            main_label = dep[0]
            full_label = ":".join(subtype for subtype in dep if subtype is not None)
            label = full_label if full_label in full_labels_to_keep else main_label

            if (candidate_id, dep) in sentence[conj_dep_id].deps.items():
                propagation = "yes"
            else:
                propagation = "no"

            already_has_dependent_type = False  # Since this is an incoming dependency, we always set this to False

            # What is the dependency relation governing the head of the relation to be propagated?
            dependency_head_rel = sentence[candidate_id].deprel

            # Linear direction of the existing dependency vs. linear direction of potential propagated dependency
            if candidate_ix < conj_head_ix and candidate_ix < conj_dep_ix:
                linear_directions = "both-right"
            elif candidate_ix > conj_head_ix and candidate_ix > conj_dep_ix:
                linear_directions = "both-left"
            else:
                linear_directions = "differing-directions"

            # Number of items in the coordination
            num_coord_items = 1 + len([deprel for _, deprel in basic_dependents(sentence, conj_head_id) if deprel == "conj"])

            yield ConjunctionInstance(sent_toks,
                                      conj_head_ix, conj_head_lemma, conj_head_upos, conj_head_feats,
                                      conj_dep_ix, conj_dep_lemma, conj_dep_upos, conj_dep_feats,
                                      id_to_ix[candidate_id], candidate_lemma, candidate_upos, candidate_feats,
                                      label, "incoming", linear_directions, already_has_dependent_type,
                                      outgoing_edges_candidate_gov, outgoing_edges_candidate_dep, dependency_head_rel,
                                      num_coord_items, propagation, conj_head_id, conj_dep_id, candidate_id)

        for (candidate_id, dep) in outgoing_links:
            if dep[0] in ignore_labels:
                continue

            candidate_ix = id_to_ix[candidate_id]
            candidate_lemma = sentence[candidate_id].lemma
            candidate_upos = sentence[candidate_id].upos
            candidate_feats = extract_relevant_features(sentence, candidate_id)

            outgoing_edges_candidate_gov = {deprel for _, deprel in basic_dependents(sentence, conj_dep_id)}
            outgoing_edges_candidate_dep = {deprel for _, deprel in basic_dependents(sentence, candidate_id)}

            main_label = dep[0]
            full_label = ":".join(subtype for subtype in dep if subtype is not None)
            label = full_label if full_label in full_labels_to_keep else main_label

            if (candidate_id, dep) in dependents(sentence, conj_dep_id):
                propagation = "yes"
            else:
                propagation = "no"

            # Check if conjunction dependent already has a dependent of this type
            already_has_dependent_type = False
            for _, lbl_basetype in basic_dependents(sentence, conj_dep_id):
                if lbl_basetype == main_label:
                    already_has_dependent_type = True
                    break

            # What is the dependency relation governing the head of the relation to be propagated?
            dependency_head_rel = sentence[conj_head_id].deprel

            # Linear direction of the existing dependency vs. linear direction of potential propagated dependency
            if candidate_ix < conj_head_ix and candidate_ix < conj_dep_ix:
                linear_directions = "both-left"
            elif candidate_ix > conj_head_ix and candidate_ix > conj_dep_ix:
                linear_directions = "both-right"
            else:
                linear_directions = "differing-directions"

            # Number of items in the coordination
            num_coord_items = 1 + len([deprel for _, deprel in basic_dependents(sentence, conj_head_id) if deprel == "conj"])

            yield ConjunctionInstance(sent_toks,
                                      id_to_ix[conj_head_id], conj_head_lemma, conj_head_upos, conj_head_feats,
                                      id_to_ix[conj_dep_id], conj_dep_lemma, conj_dep_upos, conj_dep_feats,
                                      id_to_ix[candidate_id], candidate_lemma, candidate_upos, candidate_feats,
                                      label, "outgoing", linear_directions, already_has_dependent_type,
                                      outgoing_edges_candidate_gov, outgoing_edges_candidate_dep, dependency_head_rel,
                                      num_coord_items, propagation, conj_head_id, conj_dep_id, candidate_id)


def dependents(sentence, id):
    for token in sentence:
        if id in token.deps:
            yield token.id, token.deps[id]


def basic_dependents(sentence, id):
    for token in sentence:
        if token.head == id:
            yield token.id, token.deprel


def extract_relevant_features(sentence, id):
    feats = set()
    for key in "Number", "Person", "Voice", "VerbForm":
        if key in sentence[id].feats:
            assert len(sentence[id].feats[key]) == 1
            val = next(iter(sentence[id].feats[key]))
            feats.add("{}={}".format(key, val))

    return feats


if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    with open(output_filename, "w") as output_file:
        sentences = pyconll.load_from_file(input_filename)
        csv_writer = csv.writer(output_file, delimiter="\t")

        for sentence in sentences:
            for prop in propagations(sentence):
                write_instance(prop, csv_writer)

