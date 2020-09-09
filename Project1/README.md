### Notes
- resized-cifar-100-python.tar.gz needs to be extracted, then datadistributer.py needs to be run with test data to produce `validation/` folder, which belongs in `data/`
- right now, run-eval.py has **gpu_model** hardcoded to evaluate. **gpu_model** should be placed in `models/`
- the dockerfile expects a cached version of *vgg16_bn_6c64b313.pth* in `models/`
- The default CMD for the image is to execute run-eval.py. You can overwrite this behavior by appendending a different command to the end of `docker run`
    - e.g. `docker run ... baseline /bin/bash`
- run-inference-eval.sh will run the docker image 5 times with the default CMD and print output to stdout