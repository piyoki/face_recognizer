# install dependencies
sudo apt-get update
sudo apt-get install -y build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-setuptools \
    zip

# build from source
cd /usr/local/src 
sudo wget https://objectstorage.ca-toronto-1.oraclecloud.com/n/yzpqsgba6ssd/b/bucket-20200425-1645-softwares/o/dlib-19.18.zip
sudo unzip dlib-19.18.zip
cd dlib-19.18/
sudo mkdir build
cd build
sudo cmake ..
cd ..
sudo python3 setup.py install

# install face_recognition
sudo pip3 install -U pip
sudo pip3 install face_recognition
cd ~
echo -e 'Congratulations! All Set.'

