rawname=$1

wget https://paddlehub.bj.bcebos.com/model/nlp/chinese_rbt3_L-3_H-768_A-12_fluid.tar.gz --no-check-certificate
tar xzf chinese_rbt3_L-3_H-768_A-12_fluid.tar.gz
mv chinese_rbt3_L-3_H-768_A-12_fluid $rawname/assets
rm chinese_rbt3_L-3_H-768_A-12_fluid.tar.gz


name=${rawname//L_12_H_768_A_12/L-12_H-768_A-12}
name=${name//L_24_H_1024_A_16/L-24_H-1024_A-16}
cd $rawname/assets/params
for f in * ; do mv "$f" "@HUB_$name@$f"; done

cd -
python /qjx/PaddleHub/paddlehub/commands/hub.py install $rawname
