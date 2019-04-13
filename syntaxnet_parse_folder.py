# -*- coding: utf-8 -*-

import os
import sys
import codecs
import collections
import re
import asciitree
from pathlib2 import Path
from tqdm import tqdm
import tensorflow as tf
from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from dragnn.python import spec_builder
from dragnn.python import load_dragnn_cc_impl  # This loads the actual op definitions
from dragnn.python import render_parse_tree_graphviz
from dragnn.python import visualization
from google.protobuf import text_format
from syntaxnet import load_parser_ops  # This loads the actual op definitions
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from tensorflow.python.platform import tf_logging as logging

def load_model(base_dir, master_spec_name, checkpoint_name):
    # Read the master spec
    master_spec = spec_pb2.MasterSpec()
    with open(os.path.join(base_dir, master_spec_name), "r") as f:
        text_format.Merge(f.read(), master_spec)
    spec_builder.complete_master_spec(master_spec, None, base_dir)
    logging.set_verbosity(logging.WARN)  # Turn off TensorFlow spam.

    # Initialize a graph
    graph = tf.Graph()
    with graph.as_default():
        hyperparam_config = spec_pb2.GridPoint()
        builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
        # This is the component that will annotate test sentences.
        annotator = builder.add_annotation(enable_tracing=True)
        builder.add_saver()  # "Savers" can save and load models; here, we're only going to load.

    sess = tf.Session(graph=graph)
    with graph.as_default():
        #sess.run(tf.global_variables_initializer())
        #sess.run('save/restore_all', {'save/Const:0': os.path.join(base_dir, checkpoint_name)})
        builder.saver.restore(sess, os.path.join(base_dir, checkpoint_name))
        
    def annotate_sentence(sentence):
        with graph.as_default():
            return sess.run([annotator['annotations'], annotator['traces']],
                            feed_dict={annotator['input_batch']: [sentence]})
    return annotate_sentence

segmenter_model = load_model("data/Russian-SynTagRus/segmenter", "spec.textproto", "checkpoint")
parser_model = load_model("data/Russian-SynTagRus", "parser_spec.textproto", "checkpoint")

def annotate_text(text):
    sentence = sentence_pb2.Sentence(
        text=text,
        token=[sentence_pb2.Token(word=text, start=-1, end=-1)]
    )

    # preprocess
    with tf.Session(graph=tf.Graph()) as tmp_session:
        char_input = gen_parser_ops.char_token_generator([sentence.SerializeToString()])
        preprocessed = tmp_session.run(char_input)[0]
    segmented, _ = segmenter_model(preprocessed)

    annotations, traces = parser_model(segmented[0])
    assert len(annotations) == 1
    assert len(traces) == 1
    return sentence_pb2.Sentence.FromString(annotations[0]), traces[0]
annotate_text("John is eating pizza with a fork"); None  # just make sure it works

def to_dict(sentence):
    """Builds a dictionary representing the parse tree of a sentence.

        Note that the suffix "@id" (where 'id' is a number) is appended to each
        element to handle the sentence that has multiple elements with identical
        representation. Those suffix needs to be removed after the asciitree is
        rendered.

    Args:
        sentence: Sentence protocol buffer to represent.
    Returns:
        Dictionary mapping tokens to children.
    """
    token_str = list()
    children = [[] for token in sentence.token]
    root = -1
    for i in range(0, len(sentence.token)):
        token = sentence.token[i]
        token_str.append('%s %s %s @%d' %
                        (token.word, token.tag, token.label, (i+1)))
        if token.head == -1:
            root = i
        else:
            children[token.head].append(i)

    def _get_dict(i):
        d = collections.OrderedDict()
        for c in children[i]:
            d[token_str[c]] = _get_dict(c)
        return d

    tree = collections.OrderedDict()
    tree[token_str[root]] = _get_dict(root)
    return tree

def parse_attributes(tag_string):
    res = {}
    splitted = tag_string.split()
    for i, word in enumerate(splitted):
        if word == u'name:':
            key = splitted[i + 1]
            key = key[1:len(key)-1]
            value = splitted[i + 3]
            value = value[1:len(value)-1]
            res[key] = value
    return res

def get_upos_tag(tag_string):
    return parse_attributes(tag_string)['fPOS'].split('++')[0]

def get_xpos_tag(tag_string):
    return parse_attributes(tag_string)['fPOS'].split('++')[1]

def get_feats(tag_string):
    feats_dict = parse_attributes(tag_string)
    del feats_dict['fPOS']
    res = [key + '=' + feats_dict[key] for key in feats_dict]
    res = '|'.join(res)
    if not res:
        res = '_'
    return res

def get_space_after(token, sentence):
    token_list = list(sentence.token)
    idx = token_list.index(token)
    if idx == len(token_list) - 1:
        return '_'
    elif sentence.token[idx + 1].break_level == 1:
        return '_'
    elif sentence.token[idx + 1].break_level == 0:
        return 'SpaceAfter=No'

if __name__ == '__main__':
    texts_dir = Path(sys.argv[1])
    conllu_dir = Path(sys.argv[2])
    for file_path in tqdm(texts_dir.iterdir()):
        if file_path.suffix != '.sts':
            continue
        input_file = file_path.open('r', encoding='cp1251')

        conllu = ''
        for line in input_file:
            inputSentence = " ".join(line.split()[6:])
            parse_tree, trace = annotate_text(inputSentence)

            conllu += '# text = ' + parse_tree.text + '\n'
            for i, token in enumerate(parse_tree.token):
                conllu += str(i + 1) + '\t'
                conllu += token.word + '\t'
                conllu += '_' + '\t'
                attributes = parse_attributes(token.tag)
                conllu += get_upos_tag(token.tag) + '\t'
                conllu += get_xpos_tag(token.tag) or "_" + '\t'
                conllu += get_feats(token.tag) + '\t'
                conllu += str(token.head + 1) + '\t'
                conllu += token.label + '\t'
                conllu += '_' + '\t' + get_space_after(token, parse_tree) + '\n'
            conllu += '\n'

            # tr = asciitree.LeftAligned()
            # d = to_dict(parse_tree)
            # outputFile.write('Input: %s\n' % parse_tree.text.encode('utf8'))
            # outputFile.write('Parse:\n')
            # tr_str = tr(d)
            # pat = re.compile(r'\s*@\d+$')
            # for tr_ln in tr_str.splitlines():
            #     outputFile.write(pat.sub('', tr_ln).encode('utf8'))
            #     outputFile.write('\n')

        output_file_path = (conllu_dir / (file_path.stem + '_syntaxnet.conllu'))
        with output_file_path.open('w', encoding='utf8') as f:
            f.write(conllu)
        input_file.close()
