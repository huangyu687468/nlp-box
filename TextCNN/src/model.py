import tensorflow as tf
from data import Vocab
import utils
import pdb


class TextCNN(object):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.name_to_features = {
            "input": tf.FixedLenFeature([config['data']['n_steps']], tf.int64),
            "target": tf.FixedLenFeature([1], tf.int64)
        }
        self.shuffle_size = config['data']['shuffle_size']
        self.num_parallel_calls = config['data']['num_parallel_calls']
        self.batch_size = config['data']['batch_size']
        self.train_drop_remainder = config['data']['train_drop_remainder']
        self.prefetch_size = config['data']['prefetch_size']

        self.vocab = Vocab.load(config['model']['vocab'])
        self.emb_size = config['model']['emb_size']
        self.filter_sizes = config['model']['filter_sizes']
        self.n_filters = config['model']['n_filters']
        self.dropout = config['model']['dropout']
        self.n_classes = config['model']['n_classes']

    def input_fn(self, mode, data_dir):
        def decode_record(record):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_single_example(record, self.name_to_features)

            # tf.Example don't supports tf.int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, tf.int32)
                example[name] = t
            return {'input': example['input']}, example['target'][0]

        files = utils.listdir(data_dir)
        print('in mode {}, files: {}'.format(mode, files))

        with tf.name_scope('dataset'):
            d = tf.data.TFRecordDataset(files)
            if mode == 'train':
                # d = d.repeat()
                d = d.shuffle(buffer_size=self.shuffle_size)

            d = d.map(
                lambda record: decode_record(record),
                num_parallel_calls=self.num_parallel_calls
            )
            if mode == 'train':
                d = d.batch(
                    batch_size=self.batch_size,
                    drop_remainder=self.train_drop_remainder
                )
            else:
                d = d.batch(
                    batch_size=self.batch_size,
                    drop_remainder=False
                )

            d = d.prefetch(buffer_size=self.prefetch_size)

        return d

    def get_train_input_fn(self, data_dir):
        def train_input_fn():
            return self.input_fn('train', data_dir)
        return train_input_fn

    def get_eval_input_fn(self, data_dir):
        def eval_input_fn():
            return self.input_fn('eval', data_dir)
        return eval_input_fn

    def get_infer_input_fn(self, data_dir):
        def predict_input_fn():
            return self.input_fn('infer', data_dir)
        return predict_input_fn

    def model_fn(self, features, labels, mode, params):
        ids = features['input']

        with tf.variable_scope('embedding', initializer=tf.truncated_normal_initializer(stddev=0.02)):
            emb_table = tf.get_variable('table', [self.vocab.vocab_size, self.emb_size], tf.float32)
            embs = tf.nn.embedding_lookup(emb_table, ids)
            if mode == 'train':
                embs = tf.nn.dropout(embs, 1-self.dropout)

        with tf.variable_scope('conv', tf.truncated_normal_initializer(stddev=0.02)):
            # [batch, in_height(=n_steps), in_width(=emb_size), in_channels(=1)]
            embs = tf.expand_dims(embs, -1)
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope('conv%d' % i):

                    # [filter_height, filter_width, in_channels, out_channels]
                    filters = [filter_size, self.emb_size, 1, self.n_filters]
                    w = tf.get_variable('w', filters, tf.float32)
                    b = tf.get_variable('b', [self.n_filters], tf.float32, tf.constant_initializer(0.1))

                    # [batch, H(after), 1, out_channels]
                    conv = tf.nn.conv2d(embs, w, [1, 1, 1, 1], 'VALID')
                    h_conv = tf.nn.relu(tf.nn.bias_add(conv, b))

                    # [batch, 1, 1, out_channels]
                    pooled = tf.reduce_max(h_conv, axis=1, keep_dims=True)
                    pooled_outputs.append(pooled)

        concat = tf.concat(pooled_outputs, 3)
        flat = tf.squeeze(concat, [1, 2])
        if mode == 'train':
            flat = tf.nn.dropout(flat, 1-self.dropout)

        with tf.variable_scope("output", tf.truncated_normal_initializer(stddev=0.02)):
            n_filters_total = len(self.filter_sizes) * self.n_filters
            w = tf.get_variable('w', [n_filters_total, self.n_classes], tf.float32)
            b = tf.get_variable('b', [self.n_classes], tf.float32, tf.constant_initializer(0.1))
            logits = tf.nn.xw_plus_b(flat, w, b)
            pred = tf.cast(tf.argmax(logits, 1), tf.int32)

        if mode != 'infer':
            with tf.variable_scope("loss"):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                loss = tf.reduce_mean(losses)

        if mode == 'train':
            train_op = self.get_train_op(loss)
            tf.summary.scalar('train_accuracy', accuracy)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        if mode == 'eval':
            tf.summary.scalar('eval_accuracy', accuracy)
            tp_op = tf.metrics.true_positives(labels=labels, predictions=pred)
            fp_op = tf.metrics.false_positives(labels=labels, predictions=pred)
            tn_op = tf.metrics.true_negatives(labels=labels, predictions=pred)
            fn_op = tf.metrics.false_negatives(labels=labels, predictions=pred)

            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={
                'tp': tp_op,
                'fp': fp_op,
                'tn': tn_op,
                'fn': fn_op
            })
        if mode == 'infer':
            predictions = {'pred': pred, 'input': ids}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    def get_train_op(self, loss, d_model=512, warmup_steps=4000, clip_norm=5.0, enable_mixed_precision=False):
        global_step = tf.train.get_or_create_global_step()
        lrate = self.get_learning_rate(step_num=global_step, d_model=d_model, warmup_steps=warmup_steps)
        opt = tf.train.AdamOptimizer(lrate)
        vars = tf.trainable_variables()
        if enable_mixed_precision:
            opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
            grads_and_vars = opt.compute_gradients(loss, var_list=vars, colocate_gradients_with_ops=True)
            grads, _ = zip(*grads_and_vars)
        else:
            grads = tf.gradients(loss, vars, colocate_gradients_with_ops=True)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
        train_op = opt.apply_gradients(zip(clipped_grads, vars), global_step=global_step)

        tf.summary.scalar('learning_rate', lrate)
        tf.summary.scalar('global_norm', global_norm)

        # Show the complexity of the model
        complexity = self.get_complexity(vars)
        print("Trainable variables, total param size={}".format(complexity))
        print("Format: <name>, <shape>, <(soft) device placement>")
        for var in vars:
            print("    {}, {}, {}".format(var.name, str(var.get_shape()), var.op.device))
        return train_op

    def get_learning_rate(self, step_num, d_model, warmup_steps=4000):
        """Get learning rate ."""
        lrate = tf.math.pow(tf.cast(d_model, tf.float32), -0.5)
        lrate = lrate * tf.math.minimum(
            tf.math.pow(tf.cast(step_num, tf.float32), -0.5),
            tf.cast(step_num, tf.float32) * tf.math.pow(tf.cast(warmup_steps, tf.float32), -1.5)
        )
        return lrate

    def get_complexity(self, vars):
        """ Count the total num of trainable parameters """
        total_parameters = 0
        for variable in vars:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


