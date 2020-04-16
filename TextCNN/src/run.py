# coding=utf-8
import argparse
import tensorflow as tf
import codecs
import yaml
import os
import pdb
from model import TextCNN
from data import Vocab


def pred_2_str(pred, vocab):
    ids, label = pred['input'], pred['pred']
    ids = list(filter(lambda x: x != vocab.pad_id, ids))
    ids = list(map(lambda x: int(x), ids))
    tokens = vocab.ids_2_tokens(ids)
    res = str(label)+':' + ' '.join(tokens)
    return res


def main(args):
    print(args)

    with codecs.open(args.config, 'r', "utf8") as fin:
        config = yaml.load(fin, Loader=yaml.SafeLoader)
    model = TextCNN(config)

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        tf_random_seed=config['train']['tf_random_seed'],
        save_checkpoints_steps=config['train']['save_checkpoints_steps'],
        keep_checkpoint_max=config['train']['keep_checkpoint_max'],
        save_summary_steps=config['train']['save_summary_steps']
    )
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=config['train']['model_dir'],
        config=run_config,
        params={}
    )

    for epoch in range(config['train']['epochs']):
        if args.train:
            estimator.train(
                input_fn=model.get_train_input_fn(config['train']['train_dir']),
                steps=config['train']['steps_per_train']
            )

        if args.eval:
            res = estimator.evaluate(
                input_fn=model.get_eval_input_fn(config['train']['eval_dir']),
                steps=config['train']['steps_per_eval']
            )

            dir_out = os.path.dirname(config['train']['eval_result_dir'])
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)

            with codecs.open(config['train']['eval_result_dir'], 'a', 'utf8') as fout:
                tp, tn, fp, fn = res['tp'], res['tn'], res['fp'], res['fn']
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = (2 * precision * recall) / (precision + recall)

                res_str = 'global_step: %d\n' \
                          'loss: %.7f\n' \
                          'TP: %d  TN: %d  FP: %d  FN: %d\n' \
                          'accuracy: %.6f\n' \
                          'precision: %.6f\n' \
                          'recall: %.6f\n' \
                          'f1: %.6f\n\n\n' \
                          % (res['global_step'], res['loss'], tp, tn, fp, fn, accuracy, precision, recall, f1)
                print(res_str)
                fout.write(res_str)

    if args.test:
        vocab = Vocab.load(config['model']['vocab'])

        dir_out = os.path.dirname(config['train']['test_result_dir'])
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        with codecs.open(config['train']['test_result_dir'], 'w', 'utf8') as fout:
            for pred in estimator.predict(
                    input_fn=model.get_infer_input_fn(config['train']['test_dir']),
                    checkpoint_path=args.ckpt
            ):
                utt = pred_2_str(pred, vocab)
                fout.write(utt + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--eval", default=False, action='store_true')
    parser.add_argument("--test", default=False, action='store_true')
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--config", type=str, default='config/config.yaml')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
