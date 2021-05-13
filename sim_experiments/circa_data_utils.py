import os
import csv
import argparse
import logging
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from unidecode import unidecode
from utils import removeNonAscii, isNaN


class CircaExample(object):
    '''used for training models with circa data'''
    def __init__(self,
                 id,
                 context,
                 question_x,
                 canquestion_x,
                 answer_y,
                 judgements,
                 goldstandard1,
                 goldstandard2,
                 model_output,
                 explanation):
        self.id = id
        self.context = context
        self.question_x = question_x
        self.canquestion_x = canquestion_x
        self.answer_y = answer_y
        self.judgements = judgements.split("#")
        self.goldstandard1 = goldstandard1
        self.goldstandard2 = goldstandard2
        self.model_output = model_output
        self.explanation = explanation
            
    def __str__(self):
        return self.__repr__()

    def __repr__(self):

        list_ = [f"question: {self.question_x}"] + \
            [f"indirect answer: {self.answer_y}"] + \
            [f"goldstandard1: {self.goldstandard1}"] + \
            [f"goldstandard2: {self.goldstandard2}"] + \
            [f"model_output: {self.model_output}"] + \
            [f"explanation: {self.explanation}"]

        return "\n".join(list_)



def read_circa(files, context_type=None):
    """Yields all examples if context_type is None,
            or examples per context:
            1. X wants to know about Y's food preferences
            2. X wants to know what activities Y likes to do during weekends.
            3. X wants to know what sorts of books Y likes to read.
            4. Y has just moved into a neighbourhood and meets his/her new neighbour X.
            5. X and Y are colleagues who are leaving work on a Friday at the same time.
            6. X wants to know about Y's music preferences.
            7. Y has just travelled from a different city to meet X.
            8. X and Y are childhood neighbours who unexpectedly run into each other at a cafe.
            9. Y has just told X that he/she is thinking of buying a flat in New York.
            10. Y has just told X that he/she is considering switching his/her job.
            """

    context_keyword_mapping = {
        1: "food preference",
        2: "during weekends",
        3: "sorts of books",
        4: "new neighbour",
        5: "on a Friday",
        6: "music preferences",
        7: "different city",
        8: "childhood neighbours",
        9: "buying a flat",
        10: "switching",
    }
    context_str = context_keyword_mapping.get(context_type, None)

    column_names = [
        "id",
        "context",
        "question_x",
        "canquestion_x",
        "answer_y",
        "judgements",
        "goldstandard1",
        "goldstandard2",
        "model_output",
        "explanation"
    ]

    examples = []

    for filepath in files:
        with tf.io.gfile.GFile(filepath) as f:
            tsv_reader = csv.DictReader(f, delimiter="\t", fieldnames=column_names)
            next(tsv_reader)  # skip header row

            for line in tsv_reader:
                for k, v in line.items():
                    if "goldstandard" in k:
                        line[k] = unidecode(v)  # strange apostrophe in text
                    elif k == "judgments":
                        line[k] = list(map(unidecode, v.split("#")))

                line_id = np.array([int(line["id"])])
                line["id"] = line_id

                if (context_str is None) or (context_str in line["context"]):
                    examples.append(CircaExample(line['id'], line['context'], line['question_x'], line['canquestion_x'], line['answer_y'],
                                                 line['judgements'], line['goldstandard1'], line['goldstandard2'], line['model_output'],
                                                 line['explanation']))

    return examples


def get_tensors_for_T5_split(args, examples, tokenizer, max_seq_length : int, condition_on_explanations : bool, multi_explanation : bool,
                             spliced_explanation_len = None, explanations_only = False):
    """
    Converts a list of CQAExamples into features for use with T5.

    Spliced explanation len is used in 2-agent setup, where input_ids are spliced into with sampled explanations from a model. (need to leave enough room for this)

    Format:
        Sequence 1: "[task/explain]: What is the answer to this question? The choices are choice0, choice1, choice2."
        Task Sequence 2: "The answer is: {answer}"
        Exp. Sequence 2: "The answer is {choice} because {explanation}"

    Note:
        tensor_ids serves as input_ids to model.forward
        tensors_labels serves as lm_labels to model.forward
                
    Returns: list of tensors
        
    """
    input_padding_id = tokenizer.pad_token_id   
    label_padding_id = -100
    eos_token_id = tokenizer.eos_token_id
    task_prefix_ids = tokenizer.encode("task:", add_special_tokens = False)
    explanation_prefix_ids = tokenizer.encode("explain:", add_special_tokens = False)

    return_data = []

    for example_index, example in enumerate(examples):

        # per-question variables
        question_str = example.question_x
        answer_str = example.answer
        explanation_str = example.explanation
        model_output_str = example.model_output
        goldstandard1_str = example.goldstandard1
        if isNaN(explanation_str):
            print("got nan explanation")
            example.explanation = '__'
        choice_label = example.label
        task_input_ids_list = []
        task_output_ids_list = []
        task_output_labels_list = []
        explanation_context_ids_list = []

        # first screen for length. want to keep input formatting as is due to tokenization differences with spacing before words (rather than adding all the ids)
        input_str = f"[CLS] {question_str} {answer_str} [SEP]"
        if spliced_explanation_len is not None:
            cap_length = max_seq_length-len(task_prefix_ids)-spliced_explanation_len
        else:
            cap_length = max_seq_length-len(task_prefix_ids)

        init_input_ids = tokenizer.encode(input_str)
        if len(init_input_ids) > (cap_length):
            over_by = len(init_input_ids) - cap_length 
            question_tokens = tokenizer.encode(question_str)
            keep_up_to = len(question_tokens) - over_by - 1  # leaves buffer question mark below
            new_question_tokens = question_tokens[:keep_up_to]
            question_str = tokenizer.decode(new_question_tokens) + '?'
            # print("Trimmed a question by %d tokens" % (len(question_tokens) - len(new_question_tokens)))
            # print("OLD:", tokenizer.decode(question_tokens))
            # print("NEW:", question_str)
            # print()

        # in explanations only, remove the question
        if explanations_only:
            question_str = ""

        # get string formats
        if not condition_on_explanations:
            input_str = f"[CLS] {question_str} {answer_str} [SEP]"
        if condition_on_explanations and not multi_explanation:
            input_str = f"[CLS] {question_str} {answer_str} [SEP] My commonsense tells me {explanation_str}"
        elif condition_on_explanations and multi_explanation:
            # make task_input_ids in answer loop below
            input_str = ""
        task_answer_str = f"The answer is: {model_output_str}"
        explanation_output_str = f"The answer is {model_output_str} because {explanation_str}" \
                                    if multi_explanation \
                                    else \
                                 f"My commonsense tells me that {explanation_str}"

        # get token_ids 
        _input_ids = tokenizer.encode(input_str, add_special_tokens = False)
        task_input_ids = task_prefix_ids + _input_ids 
        explanation_input_ids = explanation_prefix_ids + _input_ids
        explanation_only_ids = tokenizer.encode(example.explanation, add_special_tokens = False)
        _task_answer_ids = tokenizer.encode(task_answer_str, add_special_tokens = False)
        _explanation_output_ids = tokenizer.encode(explanation_output_str, add_special_tokens = False) + [eos_token_id]

        _truncate_seq_pair(task_input_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_input_ids, [], max_seq_length)
        _truncate_seq_pair(_explanation_output_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_only_ids, [], max_seq_length)

        task_output_str = f"The answer is: {model_output_str}"
        _task_output_ids = tokenizer.encode(task_output_str, add_special_tokens = False)
        ids_padding = [input_padding_id] * (max_seq_length - len(_task_output_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_task_output_ids))
        task_output_ids = _task_output_ids + ids_padding
        task_output_labels = _task_output_ids + labels_padding
        task_output_ids_list.append(task_output_ids)
        task_output_labels_list.append(task_output_labels)

        explanation_context_str = f"The answer is {model_output_str} because" \
                                    if multi_explanation \
                                    else \
                                  f"My commonsense tells me that"
        explanation_context_ids = tokenizer.encode(explanation_context_str, add_special_tokens = False)
        if model_output_str == goldstandard1_str:
            context_len = len(explanation_context_ids)
        explanation_context_ids += [input_padding_id] * (max_seq_length - len(explanation_context_ids))
        _truncate_seq_pair(explanation_context_ids, [], max_seq_length)
        explanation_context_ids_list.append(explanation_context_ids)

        # pad up to the max sequence len. NOTE input_padding_id goes on inputs to either the encoder or decoder. label_padding_id goes on lm_labels for decode
        padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
        task_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_input_ids))
        explanation_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_only_ids))
        explanation_only_ids += padding

        # store explanation_len for dropout/masking purposes
        explanation_len = len([e for e in explanation_context_ids if e != input_padding_id]) + len([e for e in explanation_only_ids if e != input_padding_id]) 
        
        ids_padding = [input_padding_id] * (max_seq_length - len(_task_answer_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_task_answer_ids))
        task_answer_ids = _task_answer_ids + ids_padding
        task_answer_labels = _task_answer_ids + labels_padding
        
        ids_padding = [input_padding_id] * (max_seq_length - len(_explanation_output_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_explanation_output_ids))
        explanation_output_ids = _explanation_output_ids + ids_padding
        explanation_output_labels = _explanation_output_ids + labels_padding
        explanation_output_labels[:context_len] = [label_padding_id]*context_len # no LM loss on the explanation_context_str 
        
        # make into tensors and accumulate
        task_input_ids = torch.tensor(task_input_ids if len(task_input_ids_list) < 1 else task_input_ids_list, dtype = torch.long)
        task_input_masks = (task_input_ids!=input_padding_id).float()
        task_answer_ids = torch.tensor(task_answer_ids, dtype = torch.long)
        task_answer_masks = (task_answer_ids!=input_padding_id).float()
        task_answer_labels = torch.tensor(task_answer_labels, dtype = torch.long)
        task_output_ids = torch.tensor(task_output_ids_list, dtype = torch.long)
        task_output_masks = (task_output_ids!=input_padding_id).float()
        task_output_labels = torch.tensor(task_output_labels_list, dtype = torch.long)
        explanation_input_ids = torch.tensor(explanation_input_ids, dtype = torch.long)
        explanation_input_masks = (explanation_input_ids!=input_padding_id).float()        
        explanation_output_ids = torch.tensor(explanation_output_ids, dtype = torch.long)
        explanation_output_masks = (explanation_output_ids!=input_padding_id).float()
        explanation_output_labels = torch.tensor(explanation_output_labels, dtype = torch.long)
        explanation_context_ids = torch.tensor(explanation_context_ids_list, dtype = torch.long)
        task_choice_label = torch.tensor(choice_label, dtype = torch.long)
        explanation_only_ids = torch.tensor(explanation_only_ids, dtype = torch.long)
        explanation_len = torch.tensor(explanation_len).long()
        
        data_point = [task_input_ids, task_input_masks, 
                      task_answer_ids, task_answer_masks, task_answer_labels,
                      task_output_ids, task_output_masks, task_output_labels, task_choice_label,
                      explanation_input_ids, explanation_input_masks,
                      explanation_output_ids, explanation_output_masks, explanation_output_labels,
                      explanation_context_ids, explanation_only_ids, explanation_len]
        return_data.append(data_point)

    # now reshape list of lists of tensors to list of tensors
    n_cols = len(return_data[0])
    return_data = [torch.stack([data_point[j] for data_point in return_data], dim=0) for j in range(n_cols)]

    return return_data



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()