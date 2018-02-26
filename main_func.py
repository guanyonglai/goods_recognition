import os

from detection_and_localization.detector import init_detector,det

wd = '/home/tfl/workspace/project/YI/goods_recognition/data/'

if __name__ == "__main__":

    im_path = os.path.join(wd,'frames_00298.jpg')

    cfg_file_name = 'yolo-voc-608'
    weights_file_name = 'yolo-voc-608_40000'
    # --init detector
    net, meta = init_detector(model_cfg_name=cfg_file_name, model_weights_name=weights_file_name)
    # --detect and localization
    res = det(im_path,net,meta)

    if len(res)==0:
        print('No goods detected!')
    # --recognition
    print('end')





