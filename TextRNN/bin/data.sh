set -x

cd $(dirname $0)/..

python src/data.py --action split --file_in data/waimai_10k.csv --dir_in tmp/data --rate 0.7 0.2 0.1

python src/data.py --action gen_vocab --dir_in tmp/data --stopwords stopwords/cn.txt --vocab tmp/vocab/vocab.txt --vocab_size 20000

python src/data.py --action gen_data --dir_in tmp/data --stopwords stopwords/cn.txt --vocab tmp/vocab/vocab.txt --dir_out tmp/tfrecord