rawname=$1

wget https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz --no-check-certificate
tar xf cased_L-24_H-1024_A-16.tar.gz
mv cased_L-24_H-1024_A-16 $rawname/assets
rm cased_L-24_H-1024_A-16.tar.gz

cd $rawname/assets/params
name=${rawname//L_12_H_768_A_12/L-12_H-768_A-12}
name=${name//L_24_H_1024_A_16/L-24_H-1024_A-16}
for f in * ; do mv "$f" "@HUB_$name@$f"; done

cd -
python /qjx/PaddleHub/paddlehub/commands/hub.py install $rawname
