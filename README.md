# Ai-Essentials

#download and install First:
Use minianaconda -> https://docs.conda.io/en/latest/miniconda.html as python package

after install open anacoda terminal:

conda create -y --name tensorflow2

activate tensorflow

conda install -y jupyter

jupyter notebook

conda install -y scipy
pip install --exists-action i --upgrade sklearn
pip install --exists-action i --upgrade pandas
pip install --exists-action i --upgrade pandas-datareader
pip install --exists-action i --upgrade matplotlib
pip install --exists-action i --upgrade pillow
pip install --exists-action i --upgrade tqdm
pip install --exists-action i --upgrade requests
pip install --exists-action i --upgrade h5py
pip install --exists-action i --upgrade pyyaml
pip install --exists-action i --upgrade tensorflow_hub
pip install --exists-action i --upgrade bayesian-optimization
pip install --exists-action i --upgrade spacy
pip install --exists-action i --upgrade gensim
pip install --exists-action i --upgrade flask
pip install --exists-action i --upgrade boto3
pip install --exists-action i --upgrade gym
pip install --exists-action i --upgrade tensorflow
pip install --exists-action i --upgrade keras-rl2
pip install tensorflow_datasets
conda update -y --all

#starting setup

first download and make the dataset

Run "LoadToCsv.py" with the command: pyhton LoadToCsv.py
Run "LoadToCsvNoFive.py" with the command: pyhton LoadToCsvNoFive.py

after the above commands you can run any of the "model" files

exemple command: python model_1_100000_bc.py
this will run de first model with te full dataset and the loss function binary_crossentropy
