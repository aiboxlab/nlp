"""Módulo utilitário com funcionalidades
para extração de cláusulas.
"""


def find_root_sentence(sentence):
    root_token = None
    for t in sentence:
        if t.dep_ == 'ROOT':
            root_token = t
    return root_token


def find_other_verbs(sentence, root_token):
    other_verbs = []
    excludes_verbs = []
    for t in sentence:
        ancestors = list(t.ancestors)
        ancestors = [a for a in ancestors if a not in excludes_verbs]
        if (t.pos_ == 'VERB' or t.pos_ == 'AUX') and root_token in ancestors:
            if t.dep_ == 'xcomp':
                excludes_verbs.append(t)
            else:
                other_verbs.append(t)
    return other_verbs


def get_all_children(token_, all_verbs_):
    children_ = [c for c in token_.children if c not in all_verbs_]
    if len(children_) == 0:
        return []
    for c in children_:
        children_2 = get_all_children(c, all_verbs_)
        for c2 in children_2:
            if c2 not in children_:
                children_.append(c2)
    return children_


def get_clause_token_span_for_verb(verb, all_verbs_):
    first_token_idx_ = verb.i
    last_token_idx_ = verb.i
    this_verb_children = get_all_children(verb, all_verbs_)
    for child in this_verb_children:
        if child not in all_verbs_ and child.pos_ != 'PUNCT':
            if child.i < first_token_idx_:
                first_token_idx_ = child.i
            if child.i > last_token_idx_:
                last_token_idx_ = child.i
    return first_token_idx_, last_token_idx_ + 1


def extract_clauses_by_verbs(sentence):
    root_token = find_root_sentence(sentence)
    if root_token is None:
        return None
    other_verbs = find_other_verbs(sentence, root_token)
    token_spans = []
    all_verbs = [root_token] + other_verbs
    for other_verb in all_verbs:
        first_token_idx, last_token_idx = get_clause_token_span_for_verb(
            other_verb, all_verbs)
        token_spans.append((first_token_idx, last_token_idx))
    sentence_clauses = []
    token_spans = sorted(token_spans, key=lambda tup: tup[0])
    previous_end = -1
    for token_span in token_spans:
        start = token_span[0]
        end = token_span[1]
        if start < end:
            if previous_end == -1:
                previous_end = end
            elif start < previous_end:
                continue
            clause = get_clause(sentence, start, end)
            if len(clause) > 0:
                sentence_clauses.append(clause)
    return sentence_clauses


def get_clause(sentence_, start_, end_):
    clause_ = []
    tokens = [t for t in sentence_]
    for t in tokens:
        if t.i >= end_:
            break
        elif start_ <= t.i < end_:
            clause_.append(t.text)
    return ' '.join(clause_).strip()


def split_clauses(clauses: list, delimiter: str) -> list:
    new_clauses = []
    for clause in clauses:
        if delimiter in clause:
            clauses_aux = clause.split(delimiter)
            new_clauses.extend([c.lower().strip()
                               for c in clauses_aux if len(c.strip()) > 1])
        else:
            new_clauses.append(clause.lower())
    new_clauses = [c if len(c.split()) > 1 else c + ' ' for c in new_clauses]
    return new_clauses
