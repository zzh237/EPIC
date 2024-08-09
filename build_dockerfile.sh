#!/usr/bin/env bash
# build a docker image with the current code and
# put it in GHCR

docker build -t ghcr.io/ckchow/epic .   
docker push ghcr.io/ckchow/epic