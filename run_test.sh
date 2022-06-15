#!/bin/bash

set -ex

path_test=MLDL_EGO_feature_classifier/test.py

python -u $path_test \
--backbone tsm \
--modality RGB \
--source 3 \
--target 3 \
