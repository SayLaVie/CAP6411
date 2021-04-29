#!/bin/bash 
# This script borrowed from Marlon Smith

# CMD_UNDER_TEST="python3 run_eval.py"
# CMD_UNDER_TEST="python3 test.py --img 640 --batch 4 --data ./data/FLIR.yaml.bak --weights weights/yolov5l_best.pt"
CMD_UNDER_TEST="python3 demo.py /dev/video0 --input-width=800 --input-height=600"
echo -e "Please enter name of the log file: "
read name
echo -e "Log file will be written to $name"

# Attach header to the log file
echo "CPU+GPU Current(mA), CPU+GPU Power (mW)" > $name

# Start the process
$CMD_UNDER_TEST &
pid=$!

# If this script is killed, kill the command under test.
trap "kill $pid 2> /dev/null" EXIT

# While CMD_UNDER_TEST is running...
while kill -0 $pid 2> /dev/null; do
    # Get current usage of GPU
    amps=$( cat /sys/bus/i2c/drivers/ina3221x/7-0040/iio_device/in_current1_input )
    watts=$( cat /sys/bus/i2c/drivers/ina3221x/7-0040/iio_device/in_power1_input  )
    # Log current usage
    echo "$amps,$watts" >> $name
    sleep 1
done

# Disable the trap on a normal exit.
trap - EXIT
