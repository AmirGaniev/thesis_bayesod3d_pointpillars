#!/bin/bash

CUR_DIR=$(pwd)
PROJ_DIR=$(dirname $CUR_DIR)
KITTI_TRAIN=None
KITTI_TEST=None
WAYMO_RAW=None
WAYMO_PROCESSED_DATA=None
TARGET='base'
BASH_HISTORY_FILE=~/.docker_pcdet_bash_history
ZSH_HISTORY_FILE=~/.docker_pcdet_zsh_history

# Usage info
show_help() {
echo "
Usage: ./run.sh [-h]
[--kitti]
[--waymo]
[--target TARGET]
[--bash_history_file BASH_HISTORY_FILE]
[--zsh_history_file ZSH_HISTORY_FILE]
--kitti        Links KIITI dataset
--waymo        Links Waymo dataset
--target                       TARGET Multi-stage target in Dockerfile [default=$TARGET]
--bash_history_file BASH_HISTORY_FILE Bash history file                [default=$BASH_HISTORY_FILE]
--zsh_history_file   ZSH_HISTORY_FILE Zsh history file                 [default=$ZSH_HISTORY_FILE]
"
}

while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    -k|--kitti)       # Takes an option argument; ensure it has been specified.
        KITTI_TRAIN=$(readlink -f ../data/kitti/training):/OpenPCDet/data/kitti/training
        KITTI_TEST=$(readlink -f ../data/kitti/testing):/OpenPCDet/data/kitti/testing
        ;;
    -w|--waymo)       # Takes an option argument; ensure it has been specified.
        WAYMO_RAW=$(readlink -f ../data/waymo/raw_data):/OpenPCDet/data/waymo/raw_data
        WAYMO_PROCESSED_DATA=$(readlink -f ../data/waymo/waymo_processed_data):/OpenPCDet/data/waymo/waymo_processed_data
        ;;
    -g|--target)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TARGET=$2
            shift
        else
            die 'ERROR: "--target" requires a non-empty option argument.'
        fi
        ;;
    -b|--bash_history_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            BASH_HISTORY_FILE=$2
            shift
        else
            die 'ERROR: "--bash_history_file" requires a non-empty option argument.'
        fi
        ;;
    -z|--zsh_history_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            ZSH_HISTORY_FILE=$2
            shift
        else
            die 'ERROR: "--zsh_history_file" requires a non-empty option argument.'
        fi
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

# Ensure history files exist
touch $BASH_HISTORY_FILE
touch $ZSH_HISTORY_FILE

OPENPCDET_VOLUMES=""
for entry in $PROJ_DIR/pcdet/*
do
    name=$(basename $entry)

    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        OPENPCDET_VOLUMES+="--volume $entry:/OpenPCDet/pcdet/$name "
    fi
done

docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$XAUTHORITY:/root/.Xauthority:rw" \
        --volume $BASH_HISTORY_FILE:/home/ddet/.bash_history \
        --volume $ZSH_HISTORY_FILE:/home/ddet/.zsh_history \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --env="WANDB_API_KEY=$WANDB_API_KEY" \
        --hostname="inside-DOCKER" \
        --name="openpcdet-amir" \
        --volume $PROJ_DIR/data:/OpenPCDet/data \
        --volume $PROJ_DIR/output:/OpenPCDet/output \
        --volume $PROJ_DIR/tools:/OpenPCDet/tools \
        --volume $PROJ_DIR/checkpoints:/OpenPCDet/checkpoints \
        --volume ${KITTI_TRAIN} \
        --volume ${KITTI_TEST} \
        --volume ${WAYMO_RAW} \
        --volume ${WAYMO_PROCESSED_DATA} \
        $OPENPCDET_VOLUMES \
        --rm \
        openpcdet-docker-$TARGET zsh
