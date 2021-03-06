python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mujoco-py<2.2,>=2.1
pip install -r requirements.txt
