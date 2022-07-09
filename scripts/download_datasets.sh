#!/bin/bash

DATA_DIR="/cluster/scratch/jminder/RoadSegmentation/data"
#DATA_DIR="data" # Only for local runs


if [ ! -d $DATA_DIR ]; then
  mkdir $DATA_DIR
fi

# Download Massachussets Road Dataset
MASSACHUSSETS_PATH="$DATA_DIR/massachusetts-roads/"
if [ ! -d $MASSACHUSSETS_PATH ]; then
  mkdir $MASSACHUSSETS_PATH
  kaggle datasets download balraj98/massachusetts-roads-dataset -p $MASSACHUSSETS_PATH --unzip
  if [ "$?" -ne 0 ]; then
    echo "Failed to download Massachussets Road Dataset"
    rm -rf $MASSACHUSSETS_PATH
    exit 1;
  fi
fi


# Download Massachussets Road Dataset
DEEPGLOBE_PATH="$DATA_DIR/deepglobe/"
if [ ! -d $DEEPGLOBE_PATH ]; then
  mkdir $DEEPGLOBE_PATH
  kaggle datasets download balraj98/deepglobe-road-extraction-dataset -p $DEEPGLOBE_PATH --unzip
  if [ "$?" -ne 0 ]; then
    echo "Failed to download DeepGlobe Road Dataset"
    rm -rf $DEEPGLOBE_PATH
    exit 1;
  fi
fi

# Download CIL Dataset
CLI_PATH="$DATA_DIR/cil/"
if [ ! -d $CLI_PATH ]; then
  mkdir $CLI_PATH
  kaggle competitions download -c cil-road-segmentation-2022 -p $CLI_PATH
  if [ "$?" -ne 0 ]; then
    echo "Error downloading CIL dataset"
    rm -rf $CLI_PATH
    exit 1;
  fi
  unzip $CLI_PATH/cil-road-segmentation-2022.zip -d $CLI_PATH
  rm $CLI_PATH/cil-road-segmentation-2022.zip
fi

# Download AIRS Dataset
AIRS_PATH="$DATA_DIR/airs/"
if [ ! -d $AIRS_PATH ]; then
  mkdir $AIRS_PATH
  kaggle datasets download atilol/aerialimageryforroofsegmentation -p $AIRS_PATH --unzip
  if [ "$?" -ne 0 ]; then
    echo "Error downloading AIRS dataset"
    rm -rf $AIRS_PATH
    exit 1;
  fi
fi
