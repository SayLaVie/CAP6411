#!/bin/bash

# TODO: create validation/ if it doesn't exist

# Build docker image and tag as 'baseline'
# TODO: parameterize image tag
docker build -t baseline .