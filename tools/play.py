import argparse
import torch
import os
from tqdm import tqdm

import _init_paths
import models
import datasets
from config import config
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args) # Update default config based on parsed console arguments

    return args

def main():
    args = parse_args()
    model = eval('models.'+config.MODEL.NAME +'.get_seg_model')(config)
    
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)
    
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    if 'play' in config.DATASET.TEST_SET:
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(testloader)):
                image, label, _, name, *border_padding = batch

                pred = test_dataset.inference(
                    config,
                    model,
                    image)

                sv_path = os.path.join(config.DATASET.ROOT,config.OUTPUT_DIR)
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
    
if __name__ == '__main__':
    main()