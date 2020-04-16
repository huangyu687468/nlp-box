from collections import Counter, OrderedDict
from itertools import islice
from multiprocessing import Process, Pipe, Queue, Value
import tensorflow as tf
import queue
import argparse
import jieba
import numpy
import logging
import time
import codecs
import random
import json
import os
import pdb
logging.basicConfig(level=logging.DEBUG)


class Vocab(object):
    def __init__(self, vocab_list):
        self.pad_token = '<blank>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.sep_token = '<sep>'
        self.unk_token = '<unk>'
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.sep_id = 3
        self.unk_id = 4
        self.tokens = ['<blank>', '<s>', '</s>', '<sep>', '<unk>'] + vocab_list
        self.vocab_size = len(self.tokens)
        self.token_2_id = dict((token, _id) for _id, token in enumerate(self.tokens))

    def has_token(self, token):
        return token in self.token_2_id

    def has_id(self, id):
        return id < self.vocab_size

    def handle_unk(self, token):
        raise NotImplementedError('handle_unk is not implement')

    def tokens_2_ids(self, tokens):
        if isinstance(tokens, numpy.ndarray):
            return self.tokens_2_ids(tokens.tolist())
        elif isinstance(tokens, list):
            ids = []
            n_oov = 0
            for tok in tokens:
                tid, _n_oov = self.tokens_2_ids(tok)
                if isinstance(tok, bytes) or isinstance(tok, str):
                    ids.extend(tid)
                else:
                    ids.append(tid)
                n_oov += _n_oov
            return ids, n_oov
        elif isinstance(tokens, bytes) or isinstance(tokens, str):
            if self.has_token(tokens):
                return [self.token_2_id[tokens]], 0
            return [self.unk_id], 1
        else:
            assert False
    
    def ids_2_tokens(self, ids):
        if isinstance(ids, numpy.ndarray):
            return self.ids_2_tokens(ids.tolist())
        elif isinstance(ids, list):
            return [self.ids_2_tokens(it) for it in ids]
        elif isinstance(ids, int):
            return self.tokens[ids]
        else:
            raise TypeError('unknown type: {}'.format(type(ids)))

    def dump(self, vocab_file):
        dir_out = os.path.dirname(vocab_file)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        with codecs.open(vocab_file, 'w', 'utf8') as fout:
            for w in self.tokens:
                fout.write(w + '\n')

    @classmethod
    def load(cls, vocab_file):
        vocab = []
        with codecs.open(vocab_file, 'r', 'utf8') as fin:
            for line in fin:
                vocab.append(line.strip())
        return cls(vocab[5:])


class Tokenizer(object):
    def __init__(self, stopwords=None):
        self.stopwords = dict()
        if stopwords:
            self.load_stopwords(stopwords)

    def line_2tokens(self, line):
        """ rewrite this function to process line into tokens """
        # preprocess
        line = line[2:]
        # segmentation
        tokens = jieba.cut(line)
        # postprocess
        tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def load_stopwords(self, path):
        with codecs.open(path, 'r', 'utf8') as fin:
            for i, word in enumerate(fin):
                self.stopwords[word.strip()] = i


class VocabGenerator(object):
    def __init__(self, tokenizer, vocab_size, n_process=4, n_line=100):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.n_process = n_process
        self.n_line = n_line
        self.vocab = None
        self.tokens = Counter()

    def generate(self, files):
        for file in files:
            logging.info('generate from file: {}'.format(file))
            with codecs.open(file, 'r', 'utf-8') as fin:
                self.parse_one_file(fin)
        # 5 slots are reserved for [pad, bos, eos, sep, unk]
        token_tf = self.tokens.most_common(max(self.vocab_size-5, 0))
        tokens = [v for v, f in token_tf]
        return tokens

    def parsing_worker(self, pid, text_queue, eoq, child_conn):
        logging.info('processor %d start' % pid)
        token_counter = Counter()
        n_consumed_line = 0
        while True:
            try:
                lines = text_queue.get(timeout=10)
            except Exception as e:
                if eoq.value:
                    logging.info('processor %d eof' % pid)
                    break
                logging.warning(e)
                continue

            for line in lines:
                tokens = self.tokenizer.line_2tokens(line.strip())
                token_counter.update(tokens)
            new_cnt = n_consumed_line + len(lines)
            if new_cnt // 100 != n_consumed_line // 100:
                logging.info('processor {} consumed {} lines'.format(pid, new_cnt))
            n_consumed_line = new_cnt
        child_conn.send(token_counter)
        logging.info('processor %d end' % pid)

    def parse_one_file(self, fin):
        text_queue = Queue()
        eoq = Value('b', False)
        conns = []
        processes = []
        for i in range(self.n_process):
            parent_conn, child_conn = Pipe()
            proc = Process(target=self.parsing_worker, args=(i, text_queue, eoq, child_conn))
            processes.append(proc)
            conns.append(parent_conn)
            processes[-1].start()

        n_produced_line = 0
        while True:
            lines = list(islice(fin, self.n_line))
            if not lines:
                break
            text_queue.put(lines)
            new_cnt = n_produced_line + len(lines)
            if new_cnt // 100 != n_produced_line // 100:
                logging.info('main processor produced {} lines'.format(new_cnt))
            n_produced_line = new_cnt
        eoq.value = True

        for i in range(self.n_process):
            tok_cnt = conns[i].recv()
            self.tokens.update(tok_cnt)
            processes[i].join()


class Data(object):
    def __init__(self, tokenizer, vocab, n_process=4, n_line=100, n_input_step=10, debug=False):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.n_process = n_process
        self.n_line = n_line
        self.n_input_step = n_input_step  # TODO: 输入为固定长度
        self.debug = debug

    def generate(self, input_files, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, file in enumerate(input_files):
            out_file = os.path.join(output_dir, os.path.basename(file))
            logging.info('in file "{}", out file "{}"'.format(file, out_file))
            fin = codecs.open(file, 'r', 'utf-8')
            self.parse_one_file(fin, out_file)
            fin.close()

    def parse_one_file(self, fin, out_file):
        order = 0
        text_queue = Queue()
        text_queue_end = Value('b', False)
        inst_queue = Queue()
        inst_queue_end = Value('b', False)

        compute_processes = []
        for pid in range(self.n_process):
            proc = Process(target=self.compute_processor, args=(pid, text_queue, text_queue_end, inst_queue))
            compute_processes.append(proc)
            compute_processes[-1].start()

        io_process = Process(target=self.io_processor, args=(inst_queue, inst_queue_end, out_file))
        io_process.start()

        n_produced_line = 0
        while True:
            lines = list(islice(fin, self.n_line))
            if not lines:
                break
            text_queue.put((order, lines))
            order += 1
            new_cnt = n_produced_line + len(lines)
            if new_cnt // 100 != n_produced_line // 100:
                logging.info('main processor read {} lines'.format(new_cnt))
            n_produced_line = new_cnt

        text_queue_end.value = True
        for i in range(self.n_process):
            compute_processes[i].join()

        while not inst_queue.empty():
            time.sleep(1.0)
        inst_queue_end.value = True
        io_process.join()

    def io_processor(self, inst_queue, inst_queue_end, out_file):
        logging.info('io processor started, out_file: {}'.format(out_file))
        writer = self.open_out_file(out_file)
        desired_order = 0
        buffer = {}
        n_out_line = 0
        while True:
            try:
                order, instances, n_inst = inst_queue.get(timeout=10)
                buffer[order] = (instances, n_inst)
                while desired_order in buffer:
                    instances, n_inst = buffer[desired_order]
                    self.write_out_file(writer, instances)
                    buffer.pop(desired_order)
                    desired_order += 1

                    new_cnt = n_out_line + n_inst
                    if new_cnt // 1000 != n_out_line // 1000:
                        logging.info('io processor output {} instances'.format(new_cnt))
                    n_out_line = new_cnt
            except Exception as e:
                if inst_queue_end.value:
                    logging.info('io processor end')
                    break
                logging.warning(e)
        self.close_out_file(writer)

    def compute_processor(self, pid, text_queue, text_queue_end, inst_queue):
        logging.info('generate data, processor %d start. ' % pid)
        n_consumed_line = 0
        while True:
            try:
                order, lines = text_queue.get(timeout=10)
                batch_instances = []
                total_n_inst = 0
                for line in lines:
                    instances, n_inst = self._parse_line(line)
                    batch_instances.extend(instances)
                    total_n_inst += n_inst
                inst_queue.put((order, batch_instances, total_n_inst))
                new_cnt = n_consumed_line + len(lines)
                if new_cnt // 100 != n_consumed_line // 100:
                    logging.info('processor {} consumed {} lines'.format(pid, new_cnt))
                n_consumed_line = new_cnt
            except Exception as e:
                if text_queue_end.value:
                    logging.info('generate data, processor %d eof.' % pid)
                    break
                logging.warning(e)
        logging.info('generate data, processor %d end.' % pid)

    def _parse_line(self, line):
        """ Interface: Parse a line into instances.
        """
        line = line.strip()
        review, label = line[2:], int(line[0])
        tokens = self.tokenizer.line_2tokens(review)

        ids, n_oov = self.vocab.tokens_2_ids(tokens)
        ids = ids[:self.n_input_step]
        ids = ids + [self.vocab.pad_id] * (self.n_input_step - len(ids))

        instance = {'input': ids, 'target': [label]}
        instances, n_inst = [instance], 1
        return instances, n_inst

    def ids_2_instances(self, ids):
        """Interface used to padding, add bos/eos, handle strange cases.
        """
        raise NotImplementedError('ids_2_instances is not implement')
        # return ids, 1

    def open_out_file(self, out_file):
        """ Interface used to open multiple files
        """
        tf_writer = tf.io.TFRecordWriter(out_file)
        if self.debug:
            debug_out_file = out_file + '_debug'
            debug_writer = codecs.open(debug_out_file, 'w', encoding='utf-8')
            return tf_writer, debug_writer
        return tf_writer

    def close_out_file(self, writers):
        """ Interface used to close multiple files
        """
        if self.debug:
            writers[0].close()
            writers[1].close()
        else:
            writers.close()

    def write_out_file(self, writers, instances):
        """ Interface used to write to multiple files
        """
        for inst in instances:
            feature = OrderedDict()
            feature["input"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(inst['input'])))
            feature["target"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(inst['target'])))
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

            if self.debug:
                writers[0].write(tf_example.SerializeToString())  # tfrecord
                writers[1].write(json.dumps(inst) + '\n')  # debug
            else:
                writers.write(tf_example.SerializeToString())  # tfrecord


def split(file_in, dir_out, rate):
    if len(rate) != 3 or sum(rate) > 1:
        raise ValueError("illegal rate")

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    out_dirs = [os.path.join(dir_out, sub) for sub in ['train', 'dev', 'test']]
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    fin = codecs.open(file_in, 'r', encoding='utf8')
    fout0 = codecs.open(os.path.join(out_dirs[0], os.path.basename(file_in)), 'w', encoding='utf8')
    fout1 = codecs.open(os.path.join(out_dirs[1], os.path.basename(file_in)), 'w', encoding='utf8')
    fout2 = codecs.open(os.path.join(out_dirs[2], os.path.basename(file_in)), 'w', encoding='utf8')

    for line in fin:
        rand = random.random()
        if rand < rate[0]:
            fout0.write(line)
        elif rand < sum(rate[:2]):
            fout1.write(line)
        else:
            fout2.write(line)

    fin.close()
    fout0.close()
    fout1.close()
    fout2.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, choices=['split', 'gen_vocab', 'gen_data'])
    parser.add_argument("--file_in", type=str)
    parser.add_argument("--dir_in", type=str)
    parser.add_argument('--rate', type=float, nargs='+')
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--dir_out", type=str)
    parser.add_argument("--vocab", type=str)
    parser.add_argument("--stopwords", type=str)
    parser.add_argument("--vocab_size", type=int, default=20000)
    args = parser.parse_args()
    return args


def main(args):
    if args.action == 'split':
        split(args.file_in, args.dir_in, args.rate)

    if args.action == 'gen_vocab':
        train_files = os.listdir(os.path.join(args.dir_in, 'train'))
        train_files = [os.path.join(args.dir_in, 'train', file) for file in train_files]
        tokenizer = Tokenizer(stopwords=args.stopwords)
        vg = VocabGenerator(tokenizer, args.vocab_size, n_process=4, n_line=1000)
        tokens = vg.generate(train_files)
        Vocab(tokens).dump(args.vocab)

    if args.action == 'gen_data':
        train_files = os.listdir(os.path.join(args.dir_in, 'train'))
        train_files = [os.path.join(args.dir_in, 'train', file) for file in train_files]
        dev_files = os.listdir(os.path.join(args.dir_in, 'dev'))
        dev_files = [os.path.join(args.dir_in, 'dev', file) for file in dev_files]
        test_files = os.listdir(os.path.join(args.dir_in, 'test'))
        test_files = [os.path.join(args.dir_in, 'test', file) for file in test_files]

        tokenizer = Tokenizer(stopwords=args.stopwords)
        vocab = Vocab.load(args.vocab)
        data = Data(tokenizer, vocab, n_process=4, n_line=1000, n_input_step=args.n_steps, debug=False)

        data.generate(train_files, os.path.join(args.dir_out, 'train'))
        data.generate(dev_files, os.path.join(args.dir_out, 'dev'))
        data.generate(test_files, os.path.join(args.dir_out, 'test'))


if __name__ == '__main__':
    args = parse_args()
    main(args)

