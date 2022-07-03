#!/bin/bash

DATA_DIR="/cluster/scratch/jminder/RoadSegmentation/data"


if [ ! -d $DATA_DIR ]; then
  mkdir $DATA_DIR
fi

# Download Massachussets Road Dataset
MASSACHUSSETS_PATH="$DATA_DIR/massachusetts/"
if [ ! -d $MASSACHUSSETS_PATH ]; then
  mkdir $MASSACHUSSETS_PATH
fi
kaggle datasets download balraj98/massachusetts-roads-dataset -p $MASSACHUSSETS_PATH --unzip
if [ "$?" -ne 0 ]; then
   exit 1;
fi

# Download Massachussets Road Dataset
DEEPGLOBE_PATH="$DATA_DIR/deepglobe/"
if [ ! -d $DEEPGLOBE_PATH ]; then
  mkdir $DEEPGLOBE_PATH
fi
kaggle datasets download balraj98/deepglobe-road-extraction-dataset -p $DEEPGLOBE_PATH --unzip
if [ "$?" -ne 0 ]; then
    exit 1;
fi
