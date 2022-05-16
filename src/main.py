from tensorflow.keras.optimizers import Adam

from gen import CCPDDataGen
from loss import loss
from models import build_model
from settings import INPUT_SIZE


def _main():
    # options = {
    #     "model": "/opt/darkflow/cfg/yolov2.cfg",
    #     "load": "/opt/bin/yolov2.weights",
    #     "threshold": 0.1,
    # }
    # tfnet = TFNet(options)

    # img = cv2.imread("./data/cars.jpg")
    # result = tfnet.return_predict(img)

    wpodnet = build_model()

    wpodnet.compile(loss=loss, optimizer=Adam(0.01))
    train_gen = CCPDDataGen(
        data_dir="/opt/data/CCPD2019/ccpd_weather",
        batch_size=32,
        input_size=(INPUT_SIZE, INPUT_SIZE),
    )

    wpodnet.fit(train_gen, epochs=1)


if __name__ == "__main__":
    _main()
