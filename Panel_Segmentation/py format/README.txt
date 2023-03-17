# run these commands on the prompt to install dependencies
pip install -q pyyaml==5.1
pip install -q torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
pip install -q tqdm
git clone -q https://github.com/facebookresearch/detectron2.git

# run this command to create requirements file  
pip freeze > requirements.txt

# run this command later to install dependencies via single file
pip install -r requirements.txt