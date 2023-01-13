#!/bin/sh
ACTOR_NUM=1


mkdir video
for ((i=1; i<=$ACTOR_NUM; i++));
do
    if [ $i -lt 10 ]
    then
        URL="https://zenodo.org/record/1188976/files/Video_Speech_Actor_0$i.zip?download=1"
        unziped_folder="Actor_0$i"
    else
        URL="https://zenodo.org/record/1188976/files/Video_Speech_Actor_$i.zip?download=1"
        unziped_folder="Actor_$i"
    fi
    curl -L $URL -o "$i.zip"
    unzip -q "$i.zip"
    rm "$i.zip"
    mv "$unziped_folder/"* video
    rmdir $unziped_folder
    echo "$unziped_folder download completed..."
done