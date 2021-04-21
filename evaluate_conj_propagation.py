#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


import argparse
import pyconll
import code

from collections import defaultdict


UD_SUBTYPES_TO_KEEP = {"acl:relcl",
                "aux:pass",
                "cc:preconj",
                "compound:prt",
                "csubj:pass",
                "det:predet",
                "flat:foreign",
                "nmod:npmod",
                "nmod:poss",
                "nmod:tmod",
                "nsubj:pass",
                "nsubj:xsubj",
                "obl:npmod",
                "obl:tmod"}


def evaluate(gold_conllu_filename, parsed_conllu_filename):
    gold_sentences = pyconll.load_from_file(gold_conllu_filename)
    parsed_sentences = pyconll.load_from_file(parsed_conllu_filename)

    assert len(gold_sentences) == len(parsed_sentences)

    overall_parsing_counts = defaultdict(lambda: {"predicted": 0, "gold": 0, "correct": 0})
    for gold_sentence, parsed_sentence in zip(gold_sentences, parsed_sentences):
        assert len(gold_sentence) == len(parsed_sentence)

        gold_relations = get_propagated_relations(gold_sentence)
        parsed_relations = get_propagated_relations(parsed_sentence)
        correct_relations = gold_relations & parsed_relations

        for gold_relation in gold_relations:
            label = get_simplified_label(gold_relation)
            overall_parsing_counts[label]["gold"] += 1
            overall_parsing_counts["TOTAL"]["gold"] += 1
        for parsed_relation in parsed_relations:
            label = get_simplified_label(parsed_relation)
            overall_parsing_counts[label]["predicted"] += 1
            overall_parsing_counts["TOTAL"]["predicted"] += 1
        for correct_relation in correct_relations:
            label = get_simplified_label(correct_relation)
            overall_parsing_counts[label]["correct"] += 1
            overall_parsing_counts["TOTAL"]["correct"] += 1

    return overall_parsing_counts, compute_prf_all_labels(overall_parsing_counts)


def compute_prf_all_labels(counts_dict):
    """Compute precision, recall and F-score for multiple labels individually, as well as for all labels taken together.
    Input is a nested dictionary of parsing counts (gold, predicted, correct) for each dependency label."""
    prf = dict()
    for label in counts_dict:
        prf[label] = compute_prf(counts_dict[label])

    return prf


def compute_prf(counts_dict):
    """Compute precision, recall and F-score for a single class based on counts (gold, predicted, correct)."""
    precision = counts_dict["correct"] / counts_dict["predicted"] if counts_dict["predicted"] else 0.0
    recall = counts_dict["correct"] / counts_dict["gold"] if counts_dict["gold"] else 0.0
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"precision": precision, "recall": recall, "fscore": fscore}


def get_propagated_relations(conllu_sentence):
    """Extract from the given sentence all relations resulting from conjunction propagation.
       The relations have the form (head_id, dep_id, label)"""
    relations = set()
    for token in conllu_sentence:
        # Find out if this is a conjunction dependent.
        # If yes: Find conjunction head
        # If no: Ignore this token
        has_conj_head = False
        for head_id, (main_lbl, _, _, _) in token.deps.items():
            if main_lbl == "conj":
                has_conj_head = True
                conj_head = conllu_sentence[head_id]
                break  # We assume that there is always only one conjunction head

        if not has_conj_head:
            continue

        # Propagated incoming links
        for head_id, dep_lbl in token.deps.items():
            if (head_id, dep_lbl) in conj_head.deps.items():
                relations.add((head_id, token.id, dep_lbl))

        # Propagated outgoing links
        for dep_id, dep_lbl in dependents(conllu_sentence, token.id):
            if (dep_id, dep_lbl) in dependents(conllu_sentence, conj_head.id):
                relations.add((token.id, dep_id, dep_lbl))

    return relations


def dependents(sentence, id):
    for token in sentence:
        if id in token.deps:
            yield token.id, token.deps[id]


def basic_dependents(sentence, id):
    for token in sentence:
        if token.head == id:
            yield token.id, token.deprel


def get_simplified_label(relation):
    label = ":".join(sublabel for sublabel in relation[2] if sublabel is not None)
    if label not in UD_SUBTYPES_TO_KEEP:
        label = label.split(":")[0]
    return label


def pretty_print_stats(counts, prf, sort_key="alphabet", sort_reverse=True, include_lexicalized=False, include_meta=False, latex=False):
    assert set(counts.keys()) == set(prf.keys())

    # Instantiate sort key
    if sort_key == "alphabet":
        sort_key = lambda x: x
    elif sort_key == "predicted":
        sort_key = lambda x: counts[x]["predicted"]
    elif sort_key == "gold":
        sort_key = lambda x: counts[x]["gold"]
    elif sort_key == "correct":
        sort_key = lambda x: counts[x]["correct"]
    elif sort_key == "precision":
        sort_key = lambda x: prf[x]["precision"]
    elif sort_key == "recall":
        sort_key = lambda x: prf[x]["recall"]
    elif sort_key == "fscore":
        sort_key = lambda x: prf[x]["fscore"]

    # Calculate column widths
    lbl_column_width = max(len(lbl) for lbl in counts.keys()) + 1
    predicted_column_width = max(len("Predicted"), max(len(str(counts[label]["predicted"])) for label in counts))
    gold_column_width = max(len("Gold"), max(len(str(counts[label]["gold"])) for label in counts))
    correct_column_width = max(len("Correct"), max(len(str(counts[label]["correct"])) for label in counts))
    precision_column_width = max(len("Precision"), 6)
    recall_column_width = max(len("Recall"), 6)
    fscore_column_width = max(len("F-Score"), 6)

    # Header
    if latex:
        print("\\begin{tabular}{lcccccc}")
        print("\\toprule")
        print(" & \\textbf{Predicted} & \\textbf{Gold} & \\textbf{Correct} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F-Score} \\\\")
        print("\\midrule")
    else:
        print(' ' * lbl_column_width + f'| {"Predicted":>{predicted_column_width}} | {"Gold":>{gold_column_width}} | {"Correct":>{correct_column_width}} '
                                       f'| {"Precision":>{precision_column_width}} | {"Recall":>{recall_column_width}} | {"F-Score":>{fscore_column_width}} |')
        print("-" * lbl_column_width + "+-" + "-" * predicted_column_width + "-+-" + "-" * gold_column_width + "-+-" + "-" * correct_column_width + "-+-" +
              "-" * precision_column_width + "-+-" + "-" * recall_column_width + "-+-" + "-" * fscore_column_width + "-+-")


    # Actual data
    for label in sorted(counts.keys(), key=sort_key, reverse=sort_reverse):
        if latex:
            print(f'{label} & {str(counts[label]["predicted"])} & {str(counts[label]["gold"])} & {str(counts[label]["correct"])} '
                  f'& {prf[label]["precision"] * 100:.1f} & {prf[label]["recall"] * 100:.1f} & {prf[label]["fscore"] * 100:.1f} \\\\')
        else:
            print(f'{label:<{lbl_column_width}}'
                  f'| {str(counts[label]["predicted"]):>{predicted_column_width}} | {str(counts[label]["gold"]):>{gold_column_width}} | {str(counts[label]["correct"]):>{correct_column_width}} '
                  f'| {prf[label]["precision"] * 100:>{precision_column_width}.1f} | {prf[label]["recall"] * 100:>{recall_column_width}.1f} | {prf[label]["fscore"] * 100:>{fscore_column_width}.1f} |')

    if latex:
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Evaluation (P,R,F) of enhanced dependencies resulting from conjunction propagation')
    argparser.add_argument('gold_filename', type=str, help='path to gold file (required)')
    argparser.add_argument('parsed_filename', type=str, help='path to parser output file (required)')
    args = argparser.parse_args()

    counts, prf = evaluate(args.gold_filename, args.parsed_filename)  

    pretty_print_stats(counts, prf, sort_reverse=False, latex=False)

