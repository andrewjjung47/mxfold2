from itertools import groupby
from torch.utils.data import Dataset
import torch
import re
import math
import logging
import pandas as pd

class FastaDataset(Dataset):
    def __init__(self, fasta):
        it = self.fasta_iter(fasta)
        try:
            self.data = list(it)
        except RuntimeError:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq, torch.tensor([]))


class RnaSdbDataset(Dataset):

    def __init__(self, pq_file_path: str):  # path to pq file
        # create dataset in format expected by mxfold
        # each example is tuple of 3 elements:
        # - [str] name of the "file" - we'll use `seq_id`
        # - [str] sequence
        # - [list] pair_indices
        logging.info(f"Converting pq dataset: {pq_file_path}...")
        df = pd.read_parquet(pq_file_path)
        self.data = []
        for _, row in df.iterrows():
            seq_id = row['seq_id']
            seq = row['seq']
            db_str = row['db_structure']
            target = self.db_to_target(db_str)
            self.data.append((seq_id, seq, target))
        logging.info(f"Converted {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def db_to_target(db_str: str) -> list:
        pairs = db2pairs(db_str)
        # mxfold2 expect the target to be the pairing indices (i.e. 3rd col of BPSEQ format),
        # with a leading 0 (i.e. list length is len(seq)+1)
        # start with a list of 0's, 0 represent no-pairing
        target = [0] * len(db_str)
        # go through the pairs, set their corresponding entry to the pairing index
        # note that we need 1-based index!!!
        for i, j in pairs:
            target[i] = j + 1  # 1-based
        # add the leading 0
        target = [0] + target
        # check len
        assert len(target) == len(db_str) + 1
        return target


class BPseqDataset(Dataset):
    def __init__(self, bpseq_list):
        self.data = []
        with open(bpseq_list) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l)==1:
                    self.data.append(self.read(l[0]))
                elif len(l)==2:
                    self.data.append(self.read_pdb(l[0], l[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read(self, filename):
        with open(filename) as f:
            structure_is_known = True
            p = [0]
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    l = l.rstrip('\n').split()
                    if len(l) == 3:
                        if not structure_is_known:
                            raise('invalid format: {}'.format(filename))
                        idx, c, pair = l
                        pos = 'x.<>|'.find(pair)
                        if pos >= 0:
                            idx, pair = int(idx), -pos
                        else:
                            idx, pair = int(idx), int(pair)
                        s.append(c)
                        p.append(pair)
                    elif len(l) == 4:
                        structure_is_known = False
                        idx, c, nll_unpaired, nll_paired = l
                        s.append(c)
                        nll_unpaired = math.nan if nll_unpaired=='-' else float(nll_unpaired)
                        nll_paired = math.nan if nll_paired=='-' else float(nll_paired)
                        p.append([nll_unpaired, nll_paired])
                    else:
                        raise('invalid format: {}'.format(filename))
        
        if structure_is_known:
            seq = ''.join(s)
            return (filename, seq, torch.tensor(p))
        else:
            seq = ''.join(s)
            p.pop(0)
            return (filename, seq, torch.tensor(p))

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq)

    def read_pdb(self, seq_filename, label_filename):
        it = self.fasta_iter(seq_filename)
        h, seq = next(it)

        p = []
        with open(label_filename) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l) == 2 and l[0].isdecimal() and l[1].isdecimal():
                    p.append([int(l[0]), int(l[1])])

        return (h, seq, torch.tensor(p))
    


# from https://github.com/andrewjjung47/rna_sdb/blob/75fc8895d8bb4f2fd2d31ef7156972a8cfba389c/rna_sdb/utils.py#L307
NON_PAIRING_CHARS = re.compile(r"[a-zA-Z_\,-\.\:~]")


class AllPairs(object):
    def __init__(self, db_str):
        self.db_str = db_str
        self.pairs = (
            []
        )  # list of tuples, where each tuple is one paired positions (i, j)
        # hold on to all bracket groups
        self.bracket_round = PairedBrackets(left_str="(", right_str=")")
        self.bracket_square = PairedBrackets(left_str="[", right_str="]")
        self.bracket_triang = PairedBrackets(left_str="<", right_str=">")
        self.bracket_curly = PairedBrackets(left_str="{", right_str="}")

    def parse_db(self):
        # parse dot-bracket notation
        for i, s in enumerate(self.db_str):
            # add s into bracket collection, if paired
            # also check if any bracket group is completed, if so, flush
            if re.match(NON_PAIRING_CHARS, s):
                continue
            elif self.bracket_round.is_compatible(s):
                self.bracket_round.add_s(s, i)
                # if self.bracket_round.is_complete():
                #     self.pairs.extend(self.bracket_round.flush())
            elif self.bracket_square.is_compatible(s):
                self.bracket_square.add_s(s, i)
                # if self.bracket_square.is_complete():
                #     self.pairs.extend(self.bracket_square.flush())
            elif self.bracket_triang.is_compatible(s):
                self.bracket_triang.add_s(s, i)
                # if self.bracket_triang.is_complete():
                #     self.pairs.extend(self.bracket_triang.flush())
            elif self.bracket_curly.is_compatible(s):
                self.bracket_curly.add_s(s, i)
                # if self.bracket_curly.is_complete():
                #     self.pairs.extend(self.bracket_curly.flush())
            else:
                raise ValueError(
                    "Unrecognized character {} at position {}".format(s, i)
                )

        # check that all groups are empty!!
        bracket_groups = [
            self.bracket_round,
            self.bracket_curly,
            self.bracket_triang,
            self.bracket_square,
        ]
        for bracket in bracket_groups:
            if not bracket.is_empty():
                raise ValueError(
                    "Bracket group {}-{} not symmetric: left stack".format(
                        bracket.left_str, bracket.right_str, bracket.left_stack
                    )
                )

        # collect and sort all pairs
        pairs = []
        for bracket in bracket_groups:
            pairs.extend(bracket.pairs)
        pairs = sorted(pairs)
        self.pairs = pairs


class PairedBrackets(object):
    def __init__(self, left_str, right_str):
        self.left_str = left_str
        self.right_str = right_str
        self.pairs = []  # list of tuples (i, j)
        self.left_stack = []  # left positions

    def is_empty(self):
        return len(self.left_stack) == 0

    def is_compatible(self, s):
        return s in [self.left_str, self.right_str]

    def add_s(self, s, pos):
        if s == self.left_str:
            self.left_stack.append(pos)
        elif s == self.right_str:
            # pop on item from left_stack
            i = self.left_stack.pop()
            self.pairs.append((i, pos))
        else:
            raise ValueError(
                "Expect {} or {} but got {}".format(self.left_str, self.right_str, s)
            )


def db2pairs(s):
    ap = AllPairs(s)
    ap.parse_db()
    return ap.pairs
