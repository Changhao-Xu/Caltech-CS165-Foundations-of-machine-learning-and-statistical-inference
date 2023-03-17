FROM nvcr.io/nvidia/pytorch:22.07-py3
WORKDIR .
RUN pip install hydra-core --upgrade
CMD ['python',train_with_gradient_descent.py,'name=fbaug_gradreg_lr08','hyp=gradreg']  
