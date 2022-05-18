ARG VERSION
FROM tensorflow/tensorflow:${VERSION}

WORKDIR /opt

# Install darknet.
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
	rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update && \
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
	mkdir /opt/bin && \
	mkdir -p /opt/data/ocr/ && \
	wget -O /opt/bin/yolov2.weights https://pjreddie.com/media/files/yolov2.weights && \
	# Download pre-trained OCR model.
	wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.cfg     -P data/ocr/ && \
	wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.names   -P data/ocr/ && \
	wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.data    -P data/ocr/ && \
	wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.weights -P data/ocr/ && \
	cp /opt/darknet/cfg/yolov2.cfg  /opt/darkflow/cfg/yolov2.cfg && \
	cp /opt/darknet/data/coco.names /opt/labels.txt

