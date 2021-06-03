FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3
RUN apt-get update && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev libgl1-mesa-glx ffmpeg \
       && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir librosa pandas opencv-python tabulate \
                     --index-url=http://ftp.daumkakao.com/pypi/simple \
                      --trusted-host=ftp.daumkakao.com
RUN pip install --no-cache-dir tensorflow-hub tensorflow-io tensorflow-addons --no-deps \
                    --index-url=http://ftp.daumkakao.com/pypi/simple \
                      --trusted-host=ftp.daumkakao.com

