# Production RL Summit '22

## Quick setup instructions

```
$ conda create -n rllib_tutorial python=3.9
$ conda activate rllib_tutorial
$ pip install "ray[rllib,serve]" recsim jupyterlab tensorflow torch sklearn

$ pip install grpcio  # <- Mac only
$ pip install pywin32  # <- Win10 only

$ git clone https://github.com/sven1977/rllib_tutorials
$ cd rllib_tutorials/rl_conference_2022
$ jupyter-lab
```