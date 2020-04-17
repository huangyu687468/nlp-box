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
        self.rnn_units = config['model']['rnn_units']
        self.rnn_layers = config['model']['rnn_layers']
        self.rnn_bidi = config['model']['rnn_bidi']
        # if rnn_residual is true, emb_size should be equal to rnn_units
        self.rnn_residual = config['model']['rnn_residual']
        self.rnn_state_scheme = config['model']['rnn_state_scheme']
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

        with tf.variable_scope('rnn', tf.truncated_normal_initializer(stddev=0.02)):
            input_len = self.get_length(ids, pad=self.vocab.pad_id)
            outputs, final_state = self.encode(
                mode, embs, input_len, self.rnn_units,
                n_layers=self.rnn_layers,
                bidi=self.rnn_bidi,
                dropout=self.dropout,
                residual=self.rnn_residual
            )

            if self.rnn_state_scheme == 'last':
                state = final_state[-1]  # [batch_size, n_units * 2]
            elif self.rnn_state_scheme == 'all':
                state = tf.concat(final_state, -1)  # [batch_size, n_layers * n_units * 2]
            else:
                raise ValueError("scheme '%s' is not in ['last', 'all']" % self.rnn_state_scheme)

        if mode == 'train':
            state = tf.nn.dropout(state, 1-self.dropout)

        with tf.variable_scope("output", tf.truncated_normal_initializer(stddev=0.02)):
            w = tf.get_variable('w', [state.shape.as_list()[-1], self.n_classes], tf.float32)
            b = tf.get_variable('b', [self.n_classes], tf.float32, tf.constant_initializer(0.1))
            logits = tf.nn.xw_plus_b(state, w, b)
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

    def get_length(self, x, pad=0):
        """
        获取序列的实际长度
        Note：返回的长度考虑了 bos token eos，只是不考虑 pad
        :param x: [batch_size, n_step]
        :param pad: int
        :return:
                length: [batch_size,]
        """
        not_pad = tf.not_equal(x, tf.constant(pad, x.dtype))
        length = tf.reduce_sum(tf.cast(not_pad, x.dtype), axis=-1)
        return length

    def rnn_cell(self, mode, n_units, n_layers=1, dropout=0., residual=False):
        """
        创建一个 gru 单元，可以选择层数，dropout，残差连接
        :param mode:
        :param n_units: int,
        :param n_layers: int,
        :param dropout: float,
        :param residual: boolean,
        :return:
                cell: gru cell
        """
        dropout = dropout if mode == 'train' else 0.0
        cells = []
        for i in range(n_layers):
            with tf.variable_scope('layer%d' % i):
                cell = tf.nn.rnn_cell.GRUCell(n_units)
                if dropout > 0.0:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))
                if residual:
                    cell = tf.nn.rnn_cell.ResidualWrapper(cell)
                cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True) if n_layers > 1 else cells[0]
        return cell

    def encode(self, mode, input_emb, input_len, n_units, n_layers=2, bidi=True, dropout=0., residual=False):
        # 返回 tuple(n_layers, [batch_size, n_units * 2])
        batch_size = tf.shape(input_len)[0]

        if bidi:
            cell_fw = self.rnn_cell(mode, n_units, n_layers, dropout, residual)
            cell_bw = self.rnn_cell(mode, n_units, n_layers, dropout, residual)
            init_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)

            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                input_emb,
                sequence_length=input_len,
                initial_state_fw=init_state_fw,
                initial_state_bw=init_state_bw,
            )

            outputs = tf.concat(outputs, -1)
            if n_layers == 1:
                final_state = tf.concat(final_state, -1)
                final_state = tuple([final_state])
            else:
                new_final_state = []
                for layer_id in range(n_layers):
                    concat_fw_bw = tf.concat([final_state[0][layer_id], final_state[1][layer_id]], -1)
                    new_final_state.append(concat_fw_bw)
                final_state = tuple(new_final_state)

        else:
            cell = self.rnn_cell(mode, n_units, n_layers, dropout, residual)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=input_emb,
                sequence_length=input_len,
                initial_state=init_state,
            )
            final_state = tuple([final_state])

        return outputs, final_state

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


