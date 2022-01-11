#!/bin/bash

# Log function
function echo_log() { echo "$@" |& tee -a $OUTPUT_DIR/$LOG_FILE ; }

# Default command line args
# Apollo parameters
SRC_DIR=/wave_hpc_input/projects/DensityDet
OUTPUT_DIR=/wave_hpc_job_output
DATA_DIR=/datasets
DATASET='kitti'
# List all available GPUs and count the number of lines
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
LOG_FILE=output.log
export WANDB_API_KEY=$(cat /wave_hpc_input/projects/DensityDet/wandb_data/wandb_api_key.txt)

# train.py script parameters
CFG_FILE=tools/cfgs/kitti_models/pv_rcnn.yaml
BATCH_SIZE=12
EXTRA_TAG='default'
CKPT=None
PRETRAINED_MODEL=None
TCP_PORT=18888
SYNC_BN=true
FIX_RANDOM_SEED=false
CKPT_SAVE_INTERVAL=1
MAX_CKPT_SAVE_NUM=10
WORKERS=16

# Usage info
show_help() {
echo_log "
Usage: bash tools/scripts/${0##*/} [-h]
Apollo parameters
[--src_dir SRC_DIR]
[--output_dir OUTPUT_DIR]
[--data_dir DATA_DIR]
[--dataset DATASET]
[--num_gpus NUM_GPUS]

train.py parameters
[--cfg_file CFG_FILE]
[--batch_size BATCH_SIZE]
[--extra_tag 'EXTRA_TAG']
[--ckpt CKPT]
[--pretrained_model PRETRAINED_MODEL]
[--tcp_port TCP_PORT]
[--sync_bn SYNC_BN]
[--fix_random_seed
[--ckpt_save_interval CKPT_SAVE_INTERVAL]
[--max_ckpt_save_num MAX_CKPT_SAVE_NUM]
[--workers WORKERS]

--src_dir              SRC_DIR            Source directory                                   [default=$SRC_DIR]
--output_dir           OUTPUT_DIR         Output directory                                   [default=$OUTPUT_DIR]
--data_dir             DATA_DIR           Data directory                                     [default=$DATA_DIR]
--dataset              DATASET            Dataset [kitti/kitti_tracking/nuscenes/waymo]      [default=$DATASET]
--num_gpus             NUM_GPUS           Number of GPUS                                     [default=$NUM_GPUS]

--cfg_file             CFG_FILE           Config file                                        [default=$CFG_FILE]
--batch_size           BATCH_SIZE         Batch size                                         [default=$BATCH_SIZE]
--extra_tag            EXTRA_TAG          Extra experiment tag                               [default=$EXTRA_TAG]
--ckpt		           CKPT	              Checkpoint File                                    [default=$CKPT]
--pretrained_model     PRETRAINED_MODEL   Pretrained model                                   [default=$PRETRAINED_MODEL]
--tcp_port             TCP_PORT           TCP port for distributed training                  [default=$TCP_PORT]
--sync_bn	           SYNC_BN            Sync batchnorm                                     [default=$SYNC_BN]
--fix_random_seed      FIX_RANDOM_SEED    Fix random seed                                    [default=$FIX_RANDOM_SEED]
--ckpt_save_interval   CKPT_SAVE_INTERVAL Checkpoint save interval                           [deafult=$CKPT_SAVE_INTERVAL]
--max_ckpt_save_num    MAX_CKPT_SAVE_NUM  Maximum number of checkpoints saved                [default=$MAX_CKPT_SAVE_NUM]
--workers              WORKERS            Number of workers for dataloader                   [default=$WORKERS]
"
}

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    -s|--src_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SRC_DIR=$2
            shift
        else
            die 'ERROR: "--src_dir" requires a non-empty option argument.'
        fi
        ;;
    -o|--output_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            OUTPUT_DIR=$2
            shift
        else
            die 'ERROR: "--output_dir" requires a non-empty option argument.'
        fi
        ;;
    -d|--data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    -a|--dataset)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATASET=$2
            shift
        else
            die 'ERROR: "--dataset" requires a non-empty option argument.'
        fi
        ;;
    -n|--num_gpus)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_GPUS=$2
            shift
        else
            die 'ERROR: "--num_gpus" requires a non-empty option argument.'
        fi
        ;;
    -c|--cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CFG_FILE=$2
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -b|--batch_size)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            BATCH_SIZE=$2
            shift
        else
            die 'ERROR: "--batch_size" requires a non-empty option argument.'
        fi
        ;;
    -t|--extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -k|--ckpt)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CKPT=$2
            shift
        else
            die 'ERROR: "--ckpt" requires a non-empty option argument.'
        fi
        ;;
    -p|--pretrained_model)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_MODEL=$2
            shift
        else
            die 'ERROR: "--pretrained_model" requires a non-empty option argument.'
        fi
        ;;
    -p|--tcp_port)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TCP_PORT=$2
            shift
        else
            die 'ERROR: "--tcp_port" requires a non-empty option argument.'
        fi
        ;;
    -y|--sync_bn)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SYNC_BN=$2
            shift
        else
            die 'ERROR: "--sync_bn" requires a non-empty option argument.'
        fi
        ;;
    -f|--fix_random_seed)       # Takes an option argument; ensure it has been specified.
        FIX_RANDOM_SEED="true"
        ;;
    -v|--ckpt_save_interval)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CKPT_SAVE_INTERVAL=$2
            shift
        else
            die 'ERROR: "--ckpt_save_interval" requires a non-empty option argument.'
        fi
        ;;
    -m|--max_ckpt_save_num)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MAX_CKPT_SAVE_NUM=$2
            shift
        else
            die 'ERROR: "--max_ckpt_save_num" requires a non-empty option argument.'
        fi
        ;;
    -w|--workers)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WORKERS=$2
            shift
        else
            die 'ERROR: "--workers" requires a non-empty option argument.'
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

echo_log "Running with the following arguments:
Apollo parameters:
SRC_DIR=$SRC_DIR
OUTPUT_DIR=$OUTPUT_DIR
DATA_DIR=$DATA_DIR
DATASET=$DATASET
NUM_GPUS=$NUM_GPUS

train.py parameters
CFG_FILE=$CFG_FILE
BATCH_SIZE=$BATCH_SIZE
EXTRA_TAG=$EXTRA_TAG
CKPT=$CKPT
PRETRAINED_MODEL=$PRETRAINED_MODEL
TCP_PORT=$TCP_PORT
SYNC_BN=$SYNC_BN
FIX_RANDOM_SEED=$FIX_RANDOM_SEED
CKPT_SAVE_INTERVAL=$CKPT_SAVE_INTERVAL
MAX_CKPT_SAVE_NUM=$MAX_CKPT_SAVE_NUM
WORKERS=$WORKERS
"

# Make sure everything is accesible
echo_log "Syncing directories"
PROJ_DIR=$OUTPUT_DIR/projects/DensityDet
INFOS_DIR=$PROJ_DIR/data/$DATASET/
mkdir -p $PROJ_DIR
mkdir -p $INFOS_DIR
ln -s $SRC_DIR/data/$DATASET/* $INFOS_DIR

# Datasets
echo_log "Syncing dataset"
if [ "$DATASET" == "kitti" ]
then
    DATA_DIR=$DATA_DIR/Kitti/object
elif [ "$DATASET" == "kitti_tracking" ]
then
     DATA_DIR=$DATA_DIR/Kitti/tracking
elif [ "$DATASET" == "nuscenes" ] || [ "$DATASET" == "nuscenes-tracking" ]
then
     DATA_DIR=$DATA_DIR/nuscenes
fi
SYNC_CMD="ln -s $DATA_DIR/* $INFOS_DIR"
echo_log "$SYNC_CMD"
eval $SYNC_CMD |& tee -a $OUTPUT_DIR/$LOG_FILE

# Source code
cp -r $SRC_DIR/pcdet $PROJ_DIR/pcdet
cp -r $SRC_DIR/tools $PROJ_DIR/tools
cp -r $SRC_DIR/.git $PROJ_DIR/.git
cp -r $SRC_DIR/setup.py $PROJ_DIR/setup.py

echo_log "Create packages directory"
PACKAGES_DIR=$OUTPUT_DIR/packages
mkdir -p $PACKAGES_DIR
export PYTHONPATH=$PYTHONPATH:$PACKAGES_DIR
export PYTHONUSERBASE=$PACKAGES_DIR
export LD_LIBRARY_PATH=/usr/local/lib/python3.7/dist-packages/spconv:$LD_LIBRARY_PATH

echo_log "Building pcdet"
cd $PROJ_DIR
python setup.py develop --user
cd tools

echo_log "Setup Weights & Biases"
export WANDB_CONFIG_DIR=$OUTPUT_DIR/.config/wandb

if [ $NUM_GPUS == 1 ]
then
    TRAIN_CMD="python train.py
"
else
    TRAIN_CMD="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS
    train.py
    --launcher pytorch
    --sync_bn
    --tcp_port $TCP_PORT"
fi

TRAIN_CMD+="
    --cfg_file $CFG_FILE
    --batch_size $BATCH_SIZE
    --pretrained_model $PRETRAINED_MODEL
    --extra_tag $EXTRA_TAG
    --ckpt_save_interval $CKPT_SAVE_INTERVAL
    --max_ckpt_save_num $MAX_CKPT_SAVE_NUM
    --workers $WORKERS
"
if [ $FIX_RANDOM_SEED = "true" ]
then
    TRAIN_CMD+="    --fix_random_seed
"
fi
echo "Running training"
echo_log "$TRAIN_CMD"
eval $TRAIN_CMD |& tee -a $OUTPUT_DIR/$LOG_FILE
echo "Done training
"
