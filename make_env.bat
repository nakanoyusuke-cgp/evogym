conda env create -f ./environment.yml -n evogym_mod
conda activate evogym_mod
pip install torch==1.13.1
pip install torchaudio, torchvision, jupyter
python setup.py install
