import tensorflow as tf
import copy
from bert import modeling

__all__ = ['InputExample', 'create_concat_model', 'SingleInputFeatures', 'downsample_embedding', 'MLPClassifier',
           'kl_for_log_probs', 'create_initializer', 'gather_indexes', 'get_masked_lm_output']

def kl_for_log_probs(log_p, log_q):
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def downsample_embedding(inputs):
    with tf.variable_scope("downsample_embedding", reuse=tf.AUTO_REUSE):
        embed = tf.layers.dense(inputs, 300,
                                kernel_initializer=tf.keras.initializers.glorot_normal())
        return embed


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text_a=None, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          response_type: type of response
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: string. The untokenized text of the second sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class SingleInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, 
                 input_ids, 
                 input_mask, 
                 segment_ids, 
                 seq_len,
                 input_ids_perm,
                 input_mask_perm,
                 segment_ids_perm,
                 seq_len_perm,
                 label_id,
                 masked_lm_positions,
                 masked_lm_ids,
                 masked_lm_weights):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_len = seq_len
        
        self.input_ids_perm = input_ids_perm
        self.input_mask_perm = input_mask_perm
        self.segment_ids_perm = segment_ids_perm
        self.seq_len_perm = seq_len_perm
        self.label_id = label_id

        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights


def create_concat_model(bert_config,
                        is_training,
                        input_ids,
                        input_mask,
                        segment_ids,
                        input_ids_perm,
                        input_mask_perm,
                        segment_ids_perm,
                        labels,
                        masked_lm_positions,
                        masked_lm_ids,
                        masked_lm_weights,
                        num_labels,
                        use_one_hot_embeddings,
                        l2_reg_lambda=0.1,
                        dropout_rate=1.0,
                        seed=42):

    config = copy.deepcopy(bert_config)

    if not is_training:
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0

    with tf.variable_scope("bert", reuse=tf.AUTO_REUSE):

        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            # Perform embedding lookup on the word ids.
            (embedding_output, embedding_table) = modeling.embedding_lookup(
                input_ids=input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            # Add positional embeddings and token type embeddings, then layer
            # normalize and perform dropout.
            embedding_output = modeling.embedding_postprocessor(
                input_tensor=embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)
            
            # Perform embedding lookup on the word ids.
            (embedding_output_perm, embedding_table_perm) = modeling.embedding_lookup(
                input_ids=input_ids_perm,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            # Add positional embeddings and token type embeddings, then layer
            # normalize and perform dropout.
            embedding_output_perm = modeling.embedding_postprocessor(
                input_tensor=embedding_output_perm,
                use_token_type=not config.roberta,
                token_type_ids=segment_ids_perm,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
            # mask of shape [batch_size, seq_length, seq_length] which is used
            # for the attention scores.
            attention_mask = modeling.create_attention_mask_from_input_mask(input_ids, input_mask)
            attention_mask_perm = modeling.create_attention_mask_from_input_mask(input_ids_perm, input_mask_perm)

            # Run the stacked transformer.
            # `sequence_output` shape = [batch_size, seq_length, hidden_size].
            all_encoder_layers = modeling.transformer_model(
                input_tensor=embedding_output,
                attention_mask=attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)

            sequence_output = all_encoder_layers[-2]
            
            all_encoder_layers_perm = modeling.transformer_model(
                input_tensor=embedding_output_perm,
                attention_mask=attention_mask_perm,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)

            sequence_output_perm = all_encoder_layers_perm[-2]

        # The "pooler" converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size]. This is necessary for segment-level
        # (or segment-pair-level) classification tasks where we need a fixed
        # dimensional representation of the segment.
        with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained
            first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
            pooled_output = tf.layers.dense(first_token_tensor,
                                            config.hidden_size,
                                            activation=tf.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
            first_token_tensor_perm = tf.squeeze(sequence_output_perm[:, 0:1, :], axis=1)
            pooled_output_perm = tf.layers.dense(first_token_tensor_perm,
                                                 config.hidden_size,
                                                 activation=tf.tanh,
                                                 kernel_initializer=create_initializer(config.initializer_range))

    (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
        bert_config, sequence_output, embedding_table,
        masked_lm_positions, masked_lm_ids, masked_lm_weights)

    embedding_shape = modeling.get_shape_list(pooled_output, expected_rank=2)

    ctr_entropy = MLPClassifier(embeddings=pooled_output,
                                embeddings_perm=pooled_output_perm,
                                y=labels,
                                embedding_dim=embedding_shape[1],
                                num_labels=num_labels,
                                l2_reg_lambda=l2_reg_lambda,
                                dropout_rate=dropout_rate,
                                is_training=is_training,
                                seed=seed)

    next_sentence_probability, next_sentence_logits, next_sentence_cost = ctr_entropy.create_model()

    return next_sentence_probability, next_sentence_logits, next_sentence_cost, \
           masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs


class MLPClassifier(object):

    def __init__(self,
                 embeddings,
                 embeddings_perm,
                 y,
                 embedding_dim,
                 num_labels,
                 l2_reg_lambda,
                 dropout_rate,
                 is_training,
                 seed):

        self.embeddings = embeddings
        self.embeddings_perm = embeddings_perm
        self.y = y
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.l2_reg_lambda = l2_reg_lambda
        self.keep_rate = 1 - dropout_rate
        self.is_training = is_training
        self.seed = seed

    def create_model(self):
        emb = downsample_embedding(self.embeddings)
        emb_perm = downsample_embedding(self.embeddings_perm)
        if self.is_training:
            emb = tf.nn.dropout(emb, self.keep_rate, seed=self.seed)
            emb_perm = tf.nn.dropout(emb, self.keep_rate, seed=self.seed)
        # final classifier
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):

            logits = tf.layers.dense(emb, self.num_labels,
                                     activation=tf.nn.tanh,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            logits_perm = tf.layers.dense(emb_perm, self.num_labels,
                                          activation=tf.nn.tanh,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            log_probs = tf.nn.log_softmax(logits / 0.85, axis=-1)
            log_probs = tf.stop_gradient(log_probs)
            log_probs_perm = tf.nn.log_softmax(logits_perm, axis=-1)

        with tf.variable_scope("losses", reuse=tf.AUTO_REUSE):
            # add label smoothing
            smoothing = 0.1
            l_one_hot = tf.one_hot(self.y, depth=self.num_labels, dtype=tf.float32)
            l_one_hot -= smoothing * (l_one_hot - 1. / tf.cast(self.num_labels, l_one_hot.dtype))
            ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=l_one_hot, logits=logits)
            
            probability = tf.nn.softmax(logits)
            # probability_perm = tf.nn.softmax(logits_perm)
            # x_prob = tf.distributions.Categorical(probs=probability)
            # y_prob = tf.distributions.Categorical(probs=probability_perm)
            # kl_loss = tf.distributions.kl_divergence(x_prob, y_prob)
            kl_loss = kl_for_log_probs(log_probs, log_probs_perm)
            cost = tf.reduce_mean(ce_loss) + tf.reduce_mean(kl_loss)
        
        return probability, logits, cost


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
          tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)

