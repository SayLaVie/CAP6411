### Notes
- resized-cifar-100-python.tar.gz needs to be extracted, then datadistributer.py needs to be run with test data to produce `validation/` folder
- right now, run-eval.py has **gpu_model** hardcoded to evaluate. **gpu_model** should be placed in `models/` folder
- both `validation/` and `models/` are expected to be in `model_code/`
- The default CMD for the image is to execute run-eval.py. You can overwrite this behavior by appendending a different command to the end of `docker run`
    - e.g. `docker run ... baseline /bin/bash`

### run_eval.py in Docker container
- Instructions as of 9-27-2020
- The run_eval.py script needs to be run in the docker image produced from Dockerfile. It neesd access to a clone of the torch2trt repository as well as the `model_code/` directory
- Example commands to get running would be (from `Project1/`):
```bash
docker build -t eval-models .
docker run -it --rm --runtime nvidia --network host -v $(pwd)/model_code:/infer -v $(pwd)/torch2trt:/torch2trt eval-models /bin/bash
```
- This will bring you into a shell environment in the `/infer` directory, where you can execute the script with:
```bash
python3 run_eval.py
```

### Script configuration
- The behavior the the **run_eval.py** script can be customized by setting global variables at the top of the file. Some settings are incompatible with each other so care should be taken when configuring


### Using TegraStats
- The eval code has a mechanism for calling TegraStats to run on the host Xavier device only while the container is running inferences. In order for this to work, the eval code should have credentials to login to the host Xavier machine. These credentials, along with the host machine address, should be set in `model_code/run_eval.py` in the **start_tegrastats()** and **stop_tegrastats()** methods.
- The output location of the tegrastat logs is set in `scripts/start-tegrastats.sh`. Note that the user credentials used by the docker container should have permissions to write to the output location