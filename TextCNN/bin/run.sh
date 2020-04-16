set -x

cd $(dirname $0)/..

python src/run.py --train --eval --config config/config.yaml

# use the latest ckpt to make predictions
python src/run.py --test --config config/config.yaml

# use the specified ckpt to make predictions
# python src/run.py --test --config config/config.yaml --ckpt tmp/result/model/model.ckpt-910