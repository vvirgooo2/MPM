python pretrainer.py -f 243 -b 256 --model MAE -k gt --train 1 --layers 3 -tds 2 --lr 0.0002 -lrd 0.97 --name auggt -tmr 0.8 -smn 5 --backbone transformer --gpu 0,1 --dataset h36m --MAE

python pretrainer.py -f 243 -b 200 --model MAE -k gt --train 1 --layers 3 -tds 2 --lr 0.0002 -lrd 0.97 --name auggt -tmr 0.8 -smn 5 --backbone transformer --gpu 0,1 --dataset h36m --MAE