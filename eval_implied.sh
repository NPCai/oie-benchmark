#!/bin/bash
mkdir -p ./eval_implied/
python3 benchmark.py --gold=./oie_corpus/implied.oie --out=eval_implied/Stanford.dat --stanford=./systems_output/stanford_output.txt
python3 benchmark.py --gold=./oie_corpus/implied.oie --out=eval_implied/OLLIE.dat --ollie=./systems_output/ollie_output.txt
python3 benchmark.py --gold=./oie_corpus/implied.oie --out=eval_implied/ReVerb.dat --reverb=./systems_output/reverb_output.txt
python3 benchmark.py --gold=./oie_corpus/implied.oie --out=eval_implied/ClausIE.dat --clausie=./systems_output/clausie_output.txt
python3 benchmark.py --gold=./oie_corpus/implied.oie --out=eval_implied/OpenIE-4.dat --openiefour=./systems_output/openie4_output.txt
python3 benchmark.py --gold=./oie_corpus/implied.oie --out=eval_implied/PropS.dat --props=./systems_output/props_output.txt
python3 benchmark.py --gold=./oie_corpus/implied.oie --out=eval_implied/Neural.dat --neural=./systems_output/neural_output.txt
python3 pr_plot.py --in=./eval_implied --out=./eval_implied/eval.png
echo "DONE"