#!/bin/bash

# interval in milliseconds
interval="1000"
logfile="tegra_$(date +"%Y-%m-%d_%I-%M").stats"

tegrastats --stop
test="$(tegrastats --interval $interval --logfile $logfile --start 2>&1)"

while ! test -z $test; do
    test=${test#*=}
    pid=${test%)*}
    kill $pid
    test="$(tegrastats --interval $interval --logfile $logfile --start 2>&1)"
done;

popd