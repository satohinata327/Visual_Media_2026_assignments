# 映像メディア学　レポート課題

## 環境構築

```
conda create -n vcd python=3.9 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcd
python -m pip install --upgrade pip
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
```

```
mkdir -p datasets/coco

wget -c \
  http://images.cocodataset.org/zips/val2014.zip \
  -O datasets/coco/val2014.zip

unzip -q datasets/coco/val2014.zip -d datasets/coco
```

```
mkdir -p experiments/data/POPE/coco

wget -O experiments/data/POPE/coco/coco_pope_random.json \
  https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/d6568ff81b8fd306a49e630df44f2db5c2300191/experiments/data/POPE/coco/coco_pope_random.json

wget -O experiments/data/POPE/coco/coco_pope_popular.json \
  https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/d6568ff81b8fd306a49e630df44f2db5c2300191/experiments/data/POPE/coco/coco_pope_popular.json

wget -O experiments/data/POPE/coco/coco_pope_adversarial.json \
  https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/d6568ff81b8fd306a49e630df44f2db5c2300191/experiments/data/POPE/coco/coco_pope_adversarial.json
```

## テスト

```
python experiments/eval/object_hallucination_vqa_llava.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-folder datasets/coco/val2014 \
  --question-file experiments/data/POPE/coco/coco_pope_random.json \
  --answers-file outputs/pope_random_regular_seed55.jsonl \
  --conv-mode llava_v1 \
  --seed 55
```

```
python experiments/eval/object_hallucination_vqa_llava.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-folder datasets/coco/val2014 \
  --question-file experiments/data/POPE/coco/coco_pope_random.json \
  --answers-file outputs/pope_random_vcd_seed55.jsonl \
  --conv-mode llava_v1 \
  --use_cd \
  --noise_step 999 \
  --cd_alpha 1 \
  --cd_beta 0.1 \
  --seed 55
```
## 評価
以下Randomでの評価コマンド
RandomとPopularとRegularそれぞれで評価可能
```
python experiments/eval/eval_pope.py \
  --gt_files experiments/data/POPE/coco/coco_pope_random.json \
  --gen_files outputs/pope_random_regular_seed55.jsonl

python experiments/eval/eval_pope.py \
  --gt_files experiments/data/POPE/coco/coco_pope_random.json \
  --gen_files outputs/pope_random_vcd_seed55.jsonl
```


## 結果
以下出力例と指標の見方
AccuracyとF1の数値を主に見ることになる
```
Accuracy: 0.8773
Precision: 0.9142
Recall: 0.8328
F1: 0.8716
yes: 0.4554
unknow: 0.0
```

```
Accuracy: 全質問のうち、yes/Noを正しく回答できた割合
Precision: yesのうち、実際に画像内に対象物がある数
Recall: 実際に対象物が存在する質問のうち、正しくyesと回答できた割合
F1: PrecisionとRecallの割合
yes: yes/全質問 (0.5くらいがベスト)
unknow: 異常数値の割合
```