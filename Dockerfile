FROM tensorflow/tensorflow:1.15.5-py3

WORKDIR /opt

# Install darknet.
RUN apt-get update && \
	apt-get install -y \
	git \
	wget \
	ffmpeg \
	libsm6 \
	libxext6 && \
	git clone https://github.com/pjreddie/darknet && \
	cd darknet && \
	make


# Install darkflow and YOLOv2 weights.
RUN pip install -U pip && pip install --no-cache-dir Cython opencv-python && \
	git clone https://github.com/thtrieu/darkflow && \
	cd darkflow && \
	pip install -e .&& \
	cd /opt && \
	# Download yolov2 weight file.
	mkdir bin && \
	wget -O /opt/bin/yolov2.weights https://pjreddie.com/media/files/yolov2.weights && \
	cp /opt/darknet/cfg/yolov2.cfg  /opt/darkflow/cfg/yolov2.cfg && \
	cp /opt/darknet/data/coco.names /opt/labels.txt
