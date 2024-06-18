from __future__ import print_function, division
import argparse
import os
import torch.backends.cudnn as cudnn
import time
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
from skimage import io

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-gcs', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', help='data path')
parser.add_argument('--testlist', help='testing list')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

args.dataset = "vkitti2"
# args.datapath = "/home/degbo/Desktop/SEDNet/datasets/vkitti/"
# args.testlist = "./filenames/vkitti2_test_morning.txt"
args.datapath = "/media/degbo/T7 Shield/uncertainty/Dataset-1/Processed/MVS/group1/dense/Point_clouds/temp_folder_cluster/2021-04-23_13-18-38_S2223314_DxO_res/"
args.testlist = "./filenames/UseGeo.txt"
args.loadckpt = "/home/degbo/Desktop/SEDNet/checkpoints/vkitti2/sednet-gwc-3std-lr1e-4/checkpoint_000025.ckpt"


# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False, args)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test():
    os.makedirs('./predictions/disp_0', exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        disp_est_np[sample['mask']] = 0
        gray_np = tensor2numpy(sample["left"])
        uncert_est_np = tensor2numpy(test_sample_uncertainty(sample))
        uncert_est_np[sample['mask']] = 0
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_est,gray, uncert_est, top_pad, right_pad, fn in zip(disp_est_np,gray_np, uncert_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            uncert_est = np.array(uncert_est[top_pad:, :-right_pad], dtype=np.float32)
            gray = np.array(gray[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join("predictions", fn.split('/')[-1])
            fn1 = fn.replace(".tif","_disp.tif")
            print("saving to", fn1, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            io.imsave(fn1, disp_est)
            fn2 = fn.replace(".tif","_uncert.tif")
            io.imsave(fn2, abs(uncert_est))
            fn3 = fn.replace(".tif","_gray.tif")
            io.imsave(fn3, gray)
            tmp = 1


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    output = model(sample['left'].cuda(), sample['right'].cuda())
    disp_ests = output['disp']
    return disp_ests[-1]

@make_nograd_func
def test_sample_uncertainty(sample):
    model.eval()
    output = model(sample['left'].cuda(), sample['right'].cuda())
    uncert_ests = output['uncert']
    return uncert_ests[-1]


if __name__ == '__main__':
    test()
