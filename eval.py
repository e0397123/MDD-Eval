#! usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import tensorflow as tf
import codecs
import pickle
from best_checkpoint_copier import BestCheckpointCopier
from bert import modeling
from bert import optimization
from nli_eval import create_concat_model, SingleInputFeatures, InputExample
from tf_metrics import precision, recall, f1
from transformers import RobertaTokenizer as tokenization

__all__ = ['DataProcessor', 'NliProcessor', 'convert_single_example',
           'filed_based_convert_examples_to_features', 'file_based_input_fn_builder',
           'model_fn_builder', 'main']

flags = tf.flags

FLAGS = flags.FLAGS

# required arguments

flags.DEFINE_string('data_dir', default=None, help="train, dev and test data dir")
flags.DEFINE_string('bert_config_file', default=None, help="bert config file path")
flags.DEFINE_string('output_dir', default=None, help='directory of trained model')
flags.DEFINE_string('init_checkpoint', None,
                    help='Initial checkpoint (usually from a pre-trained model).')
# default arguments
flags.DEFINE_string('task', default='nli', help='which modle to train')
flags.DEFINE_integer('max_seq_len', default=60,
                     help='The maximum total input sequence length after Sentencepiece tokenization.')
flags.DEFINE_integer('batch_size', default=32, help='Total batch size for training, eval and predict.')
flags.DEFINE_integer('num_train_epochs', default=10, help='Total number of training epochs to perform.')
flags.DEFINE_integer('seed', default=123456, help='random seed')
flags.DEFINE_integer('keep_checkpoint_max', default=3, help='keep_checkpoint_max')
flags.DEFINE_integer('save_checkpoints_steps', default=2000, help='save_checkpoints_steps')
flags.DEFINE_integer('save_summary_steps', default=2000, help='save_summary_steps.')
flags.DEFINE_float('learning_rate', default=1e-5, help='The initial learning rate for Adam.')
flags.DEFINE_float('dropout_rate', default=0.5, help='Dropout rate')
flags.DEFINE_float('l2_reg_lambda', default=0.2, help='l2_reg_lambda')
flags.DEFINE_float('warmup_proportion', default=0.025,
                   help='Proportion of training to perform linear learning rate warmup for '
                        'E.g., 0.1 = 10% of training.')
flags.DEFINE_bool('do_train', default=False, help='Whether to run training.')
flags.DEFINE_bool('do_eval', default=False, help='Whether to run eval on the dev set.')
flags.DEFINE_bool('do_predict', default=False, help='Whether to run the predict in inference mode on the test set.')
flags.DEFINE_bool('filter_adam_var', default=False,
                  help='after training do filter Adam params from model and save no Adam params model in file.')
flags.DEFINE_bool('do_lower_case', default=True, help='Whether to lower case the input text.')
flags.DEFINE_bool('clean', default=False, help="whether to clean output folder")
flags.DEFINE_string('eval_file_path', default=None, help="path to evaluation file")
flags.DEFINE_bool('do_evaluate', default=False, help='Whether to perform dialogue evaluation')
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")


logger = tf.get_logger()
logger.propagate = False
tokenizer = tokenization.from_pretrained('library/roberta-base/')
tf.random.set_random_seed(FLAGS.seed)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        raise NotImplementedError()


class NliProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = []
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt"), split='train'), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "valid.txt"), split='valid'), "valid"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt"), split='test'), "test")

    def get_eval_examples(self):
        return self._create_example(
            self._read_data(FLAGS.eval_file_path, split='eval'), "eval")

    def get_labels(self):
        self.labels.append('random')
        self.labels.append('adversarial')
        self.labels.append('original')
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        if set_type != 'eval':
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                label = line[0]
                text_a = line[1]
                text_b = line[2]
                text_c = line[3]
                # if i == 0:
                #     logger.info('label: ', label)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
        else:
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                label = 'original'
                text_a = line[1]
                text_b = line[2]
                text_c = line[3]
                # if i == 0:
                #     logger.info('label: ', label)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))

        return examples

    def _read_data(self, input_file, split='train'):
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                content = line.strip().split('|||')
                label = content[0]
                sentence_a = content[1]
                sentence_b = content[2]
                sentence_c = content[3]
                lines.append([label, sentence_a, sentence_b, sentence_c])
        return lines


def _truncate_seq_back(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop(-2)


def _truncate_seq_front(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop(1)


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_size, rng):
    """Creates the predictions for the masked LM objective."""
    vocab_words = list(range(vocab_size))
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == 0 or token == 2:
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = 50264
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(4, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def convert_single_example(ex_index, example, label_list, max_seq_len, output_dir, rng, flag=True):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    a_input_ids = tokenizer.encode(example.text_a)
    b_input_ids = tokenizer.encode(example.text_b)
    c_input_ids = tokenizer.encode(example.text_c)

    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=a_input_ids[1:-1],
                                                           token_ids_1=b_input_ids[1:-1])
    input_ids_perm = tokenizer.build_inputs_with_special_tokens(token_ids_0=a_input_ids[1:-1],
                                                                token_ids_1=c_input_ids[1:-1])
    _truncate_seq_front(input_ids, max_seq_len)
    _truncate_seq_front(input_ids_perm, max_seq_len)

    (input_ids, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(input_ids,
                                                                                      FLAGS.masked_lm_prob,
                                                                                      FLAGS.max_predictions_per_seq,
                                                                                      tokenizer.vocab_size,
                                                                                      rng)
    masked_lm_ids = masked_lm_labels
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < FLAGS.max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    input_type_ids = [1] * len(input_ids)
    token_len = len(input_ids)
    input_mask = [1] * len(input_ids)
    
    input_type_ids_perm = [1] * len(input_ids_perm)
    token_len_perm = len(input_ids_perm)
    input_mask_perm = [1] * len(input_ids_perm)

    label_id = label_map[example.label]

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(1)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(input_type_ids) == max_seq_len
    
    while len(input_ids_perm) < max_seq_len:
        input_ids_perm.append(1)
        input_mask_perm.append(0)
        input_type_ids_perm.append(0)

    assert len(input_ids_perm) == max_seq_len
    assert len(input_mask_perm) == max_seq_len
    assert len(input_type_ids_perm) == max_seq_len

    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("unique_id: %s" % example.guid)
        logger.info("sequence length: %s" % str(token_len))
        logger.info("sequence input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("sequence input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("sequence input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
        logger.info("sequence length perm: %s" % str(token_len_perm))
        logger.info("sequence input_ids perm: %s" % " ".join([str(x) for x in input_ids_perm]))
        logger.info("sequence input_mask perm: %s" % " ".join([str(x) for x in input_mask_perm]))
        logger.info("sequence input_type_ids perm: %s" % " ".join([str(x) for x in input_type_ids_perm]))
        logger.info("label id: %s" % str(label_id))

    feature = SingleInputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=input_type_ids,
        seq_len=token_len,
        input_ids_perm=input_ids_perm,
        input_mask_perm=input_mask_perm,
        segment_ids_perm=input_type_ids_perm,
        seq_len_perm=token_len_perm,
        label_id=label_id,
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_ids,
        masked_lm_weights=masked_lm_weights)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_len, output_file, output_dir, rng, flag=True):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_len, output_dir, rng, flag=flag)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["seq_len"] = create_int_feature([feature.seq_len])
        features["input_ids_perm"] = create_int_feature(feature.input_ids_perm)
        features["input_mask_perm"] = create_int_feature(feature.input_mask_perm)
        features["segment_ids_perm"] = create_int_feature(feature.segment_ids_perm)
        features["seq_len_perm"] = create_int_feature([feature.seq_len_perm])
        features["label_id"] = create_int_feature([feature.label_id])
        features["masked_lm_ids"] = create_int_feature(feature.masked_lm_ids)
        features["masked_lm_positions"] = create_int_feature(feature.masked_lm_positions)
        features["masked_lm_weights"] = create_float_feature(feature.masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, max_seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "seq_len": tf.FixedLenFeature([1], tf.int64),
        "input_ids_perm": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask_perm": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids_perm": tf.FixedLenFeature([max_seq_length], tf.int64),
        "seq_len_perm": tf.FixedLenFeature([1], tf.int64),
        "label_id": tf.FixedLenFeature([1], tf.int64),
        "masked_lm_ids": tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "masked_lm_positions": tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps):
    """
    :param albert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :return:
    """

    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        seq_len = tf.squeeze(features["seq_len"])
        input_ids_perm = features["input_ids_perm"]
        input_mask_perm = features["input_mask_perm"]
        segment_ids_perm = features["segment_ids_perm"]
        seq_len_perm = tf.squeeze(features["seq_len_perm"])
        label_id = tf.squeeze(features["label_id"])

        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        nsp_probability, nsp_logits, nsp_loss, \
        masked_lm_loss, masked_lm_example_loss, \
        masked_lm_log_probs = create_concat_model(bert_config=bert_config,
                                                  input_ids=input_ids,
                                                  input_mask=input_mask,
                                                  segment_ids=segment_ids,
                                                  input_ids_perm=input_ids_perm,
                                                  input_mask_perm=input_mask_perm,
                                                  segment_ids_perm=segment_ids_perm,
                                                  labels=label_id,
                                                  masked_lm_positions=masked_lm_positions,
                                                  masked_lm_ids=masked_lm_ids,
                                                  masked_lm_weights=masked_lm_weights,
                                                  num_labels=num_labels,
                                                  use_one_hot_embeddings=False,
                                                  l2_reg_lambda=FLAGS.l2_reg_lambda)
        total_loss = masked_lm_loss + nsp_loss

        tvars = tf.trainable_variables()

        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logger.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logger.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            # train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)

            train_op = optimization.create_optimizer(total_loss,
                                                     learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps,
                                                     False)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['mlm_loss'] = masked_lm_loss
            hook_dict['nsp_loss'] = nsp_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=FLAGS.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            logger.info("shape of label_id: {}".format(label_id.shape))

            def metric_fn(label_id, nsp_logits, masked_lm_log_probs,
                          masked_lm_example_loss, masked_lm_ids, masked_lm_weights):
                prec = precision(labels=label_id,
                                 predictions=tf.argmax(nsp_logits, axis=1),
                                 num_classes=num_labels, pos_indices=[0, 1, 2], average='macro')
                rec = recall(labels=label_id,
                             predictions=tf.argmax(nsp_logits, axis=1),
                             num_classes=num_labels,  pos_indices=[0, 1, 2], average='macro')
                fscore = f1(labels=label_id,
                            predictions=tf.argmax(nsp_logits, axis=1),
                            num_classes=num_labels, pos_indices=[0, 1, 2], average='macro')
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                return {
                    "precision": prec,
                    "recall": rec,
                    "f1-score": fscore,
                    "accuracy": tf.metrics.accuracy(labels=label_id, predictions=tf.argmax(nsp_logits, axis=1)),
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_mean_loss": masked_lm_mean_loss
                }

            eval_metrics = metric_fn(label_id, nsp_logits,
                                     masked_lm_log_probs, masked_lm_example_loss, masked_lm_ids, masked_lm_weights)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )

        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=nsp_probability
            )
        return output_spec

    return model_fn


def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        logger.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """
    :param model_path:
    :return:
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))


def main(_):

    processors = {
        "nli": NliProcessor
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_len, bert_config.max_position_embeddings))

    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                logger.info(e)
                logger.info('please remove the files of output dir and data.conf')
                exit(-1)

    # check output dir exists
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    processor = processors[FLAGS.task](FLAGS.output_dir)

    logger.info("total vocabulary size is: {}".format(bert_config.vocab_size))

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
#    dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=session_config,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None
    rng = random.Random(FLAGS.seed)
    if FLAGS.do_train and FLAGS.do_eval:

        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) * 1.0 / FLAGS.batch_size * FLAGS.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", FLAGS.batch_size)

    label_list = processor.get_labels()

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    params = {
        'batch_size': FLAGS.batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if FLAGS.do_train and FLAGS.do_eval:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_len, train_file, FLAGS.output_dir, rng, flag=True)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            max_seq_length=FLAGS.max_seq_len,
            is_training=True,
            drop_remainder=True)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_len, eval_file, FLAGS.output_dir, rng, flag=True)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            max_seq_length=FLAGS.max_seq_len,
            is_training=False,
            drop_remainder=True)

        # train and eval togither
        # early stop hook
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=FLAGS.save_checkpoints_steps)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])

        best_copier = BestCheckpointCopier(
            name='best',  # directory within model directory to copy checkpoints to
            checkpoints_to_keep=1,  # number of checkpoints to keep
            score_metric='f1-score',  # metric to use to determine "best"
            compare_fn=lambda x, y: x.score > y.score,
            sort_key_fn=lambda x: x.score,
            sort_reverse=True)  # sort order when discarding excess checkpoints

        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=1000, exporters=best_copier)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    if FLAGS.do_evaluate:

        predict_examples = processor.get_eval_examples()
        predict_file = FLAGS.eval_file_path + ".tf_record"
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_len,
                                                 predict_file, FLAGS.output_dir, rng, flag=False)

        logger.info("***** Running Evaluation*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", FLAGS.batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            max_seq_length=FLAGS.max_seq_len,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = FLAGS.eval_file_path + ".score"

        def result_to_pair(write_agent):
            for predict_line, prediction in zip(predict_examples, result):
                line = ''
                try:
                    line += '\t'.join([str(item) for item in prediction]) + '\n'
                except Exception as e:
                    logger.info(e)
                    logger.info(predict_line.text_a)
                    logger.info(predict_line.text_b)
                    break
                write_agent.write(line)

        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)

        logger.info("evaluation has completed!")

    # filter model
    if FLAGS.filter_adam_var:
        adam_filter(FLAGS.output_dir)


if __name__ == "__main__":
    tf.app.run()
