#!/bin/bash

set -ex

path_test=MLDL_EGO_feature_classifier/test.py

python -u $path_test \
--backbone i3d \
--modality Flow \
--source 2 \
--target 2 \
