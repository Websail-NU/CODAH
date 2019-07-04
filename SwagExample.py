import csv
import sys

class SwagExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def to_dict(self):
        return {
            'swag_id': self.swag_id,
            'context_sentence': self.context_sentence,
            'start_ending': self.start_ending,
            'ending_0': self.endings[0],
            'ending_1': self.endings[1],
            'ending_2': self.endings[2],
            'ending_3': self.endings[3],
            'label': self.label,
        }

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "swag_id: {}".format(self.swag_id),
            "context_sentence: {}".format(self.context_sentence),
            "start_ending: {}".format(self.start_ending),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)

def read_swag_examples(input_file, is_training, answer_only=False, data_format='swag'):
    if data_format == 'codah':
        return _read_swag_examples_ours(input_file, is_training, answer_only)
    elif data_format == 'swag':
        return _read_swag_examples_normal(input_file, is_training, answer_only)
    else:
        raise ValueError("Bad data format {}".format(data_format))

def _read_swag_examples_ours(input_file, is_training, answer_only):
    """To be clear, this reads our TSV data format for our SWAG-style questions."""
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        lines = []
        for line in reader:
            lines.append(line)

    examples = [
        SwagExample(
            swag_id = i, # Made-up ID since our data format doesn't include an ID
            context_sentence = '' if answer_only else line[1],
            start_ending = '',
            ending_0 = line[2],
            ending_1 = line[3],
            ending_2 = line[4],
            ending_3 = line[5],
            label = int(line[6]) if is_training else None
        ) for i, line in enumerate(lines)
    ]

    return examples

def _read_swag_examples_normal(input_file, is_training, answer_only):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)

    if is_training and lines[0][-1] != 'label':
        raise ValueError(
            "For training, the input file must contain a label column."
        )

    examples = [
        SwagExample(
            swag_id = line[2],
            context_sentence = '' if answer_only else line[4],
            start_ending = line[5], # in the swag dataset, the
                                         # common beginning of each
                                         # choice is stored in "sent2".
            ending_0 = line[7],
            ending_1 = line[8],
            ending_2 = line[9],
            ending_3 = line[10],
            label = int(line[11]) if is_training else None
        ) for line in lines[1:] # we skip the line with the column names
    ]

    return examples

