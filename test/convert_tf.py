#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from tennisbuilder.tf.converter import convert
import tennisbuilder.tf.parser as parser

import tensorflow as tf
if parser.version_satisfy(tf.__version__, "1.14"):
    tf = tf.compat.v1


def test():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            parser.session_load_graph_def(sess, "tf/tensorflow.pb")
            graph = sess.graph
            print([n.name for n in graph.as_graph_def().node])

            inputs = sess.graph.get_tensor_by_name("inputs_image:0")
            outputs = sess.graph.get_tensor_by_name("outputs:0")

            convert(graph,
                    inputs=inputs,
                    outputs=outputs,
                    output_file="tf.tsm")


if __name__ == '__main__':
    test()
