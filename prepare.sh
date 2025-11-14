# Install required packages
pip install gdown

# Data download and extraction
gdown --id 1sntFzLea6vFRiGB5taQiCOidvKFeBXmL -O data.zip
unzip data.zip -d ./
rm data.zip

# Base weights download and extraction
gdown --id 1BYemJoiMtD1tz6Yym_1ToErLmwXGh7EY -O weights.tar.gz
mkdir -p ./.cache/
tar -xzf weights.tar.gz -C ./.cache/
rm weights.tar.gz

# Base locational weights download and extraction
gdown --id 163inf0A1zO0SdwHfCU6JHCyrXWQZA4LE -O weights.zip
unzip weights.zip -d ./ordinalclip/models/
rm weights.zip

# Build conda environment
pip install "pip<24.1"
conda create --name yearclip python=3.8 -y
conda activate yearclip
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP
pip install -e .