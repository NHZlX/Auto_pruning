#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/lenet_pruning_solver.prototxt -weights=examples/mnist/lenet_iter_10000.caffemodel -gpu 1
