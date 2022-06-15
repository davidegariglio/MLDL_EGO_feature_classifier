#!/bin/bash

set -ex

path_test=MLDL_EGO_Project/test.py

python -u $path_test \
--backbone i3d \
--modality RGB\
--source 3\
--target 3\
