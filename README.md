### Quick Start ###

1.clone this repo

2.genera data

`sh bin/data.sh`

3.train and evaluate

`python src/run.py --train --eval --config config/config.yaml` 

4.test

`python src/run.py --test --config config/config.yaml`

or

`python src/run.py --test --config config/config.yaml --ckpt tmp/result/model/model.ckpt-910`

5.test score

`python src/score.py --ground_truth tmp/data/test/waimai_10k.csv --prediction tmp/result/test/test.txt`



    accuracy: 0.831829
    precision: 0.759067
    recall: 0.723457
    f1: 0.740834
