wget https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz --no-check-certificate
tar xvf multilingual_L-12_H-768_A-12.tar.gz
mv multilingual_L-12_H-768_A-12 assets
rm multilingual_L-12_H-768_A-12.tar.gz

rawname=$1
name=${rawname//L_12_H_768_A_12/L-12_H-768_A-12}
name=${name//L_24_H_1024_A_16/L-24_H-1024_A-16}
cd rawname/assets/params
for f in * ; do mv "$f" "@HUB_$name@$f"; done

cd -
python /qjx/PaddleHub/paddlehub/commands/hub.py install $rawname
