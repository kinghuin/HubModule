filename=$1

modulename=${filename//L_12_H_768_A_12/L-12_H-768_A-12}
modulename=${modulename//L_24_H_1024_A_16/L-24_H-1024_A-16}


mv ~/.paddlehub/modules/$filename/assets $filename
mv ~/.paddlehub/modules/$filename/model $filename/assets/params
rm -r ~/.paddlehub/modules/$filename

python /qjx/PaddleHub/paddlehub/commands/hub.py install $filename
