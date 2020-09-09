#!/bin/bash

# This script runs 5 iterations of the eval script
# TODO: parameterize number of iterations

# Displaying power/fan settings requires sudo
echo "Current power/fan settings:"
# sudo /usr/sbin/nvpmodel -q

# for i in {1..5}; do
#     echo "~~~~~~~~"
#     echo "Run $i"
#     docker run -it --rm --runtime nvidia --network host \
#         -v $(pwd)/baseline:/infer  \
#         -v $(pwd)/models:/infer/models \
#         -v $(pwd)/data/validation:/infer/validation \
#         baseline
#     echo "~~~~~~~~"
# done


docker run -it --rm --runtime nvidia --network host \
    -v $(pwd)/baseline:/infer  \
    -v $(pwd)/models:/infer/models \
    -v $(pwd)/data/validation:/infer/validation \
    baseline