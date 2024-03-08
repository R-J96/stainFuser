FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.8.0

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 openslide-tools libvips -y

RUN pip install --no-cache-dir --upgrade pip 

RUN pip install --no-cache-dir torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /tmp/

RUN pip install --no-cache-dir diffusers["torch"]==0.26.3 \
    pip install --no-cache-dir -U albumentations==1.3.0 --no-binary qudida,albumentations \
    pip install --no-cache-dir --no-dependencies tiatoolbox==1.4.1
