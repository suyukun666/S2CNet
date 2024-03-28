import os
import yaml
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import argparse
from scipy.stats import spearmanr, pearsonr
from model.ssc import SSC
from data.croppingdataset import CropDataset


parser = argparse.ArgumentParser(description='User generated content based image cropping')
parser.add_argument('--cfg', required=True, help='config path')
parser.add_argument('--gpu', default='0', help='config path')
parser.add_argument('--pretrained', default=None, type=str, help='Checkpoint state_dict file to load training from')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cuda = True if torch.cuda.is_available() else False
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    filepath = os.path.join(os.getcwd(), args.cfg)
    with open(filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2namespace({**config, **vars(args)})

    dataset_dir = os.path.join(cfg.dataset_root, cfg.dataset)
    data_loader_test = data.DataLoader(CropDataset(image_size=cfg.image_size, dataset_dir=dataset_dir, set='test'),
                                       batch_size=cfg.val_batch, num_workers=cfg.num_workers, shuffle=False)
    net = SSC(cfg)
    if args.pretrained:
        print('loading pretrained weight')
        net.load_state_dict(torch.load(args.pretrained))
    if cuda:
        net = torch.nn.DataParallel(net,device_ids=[0])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        net = net.cuda()
    srcc, acc5, acc10 = test(net, data_loader_test, cfg.dataset)


def compute_acc(gt_scores, pr_scores):
    assert (len(gt_scores) == len(pr_scores)), '{} vs. {}'.format(len(gt_scores), len(pr_scores))
    sample_cnt = 0
    acc4_5  = [0 for i in range(4)]
    acc4_10 = [0 for i in range(4)]
    for i in range(len(gt_scores)):
        gts, preds  = gt_scores[i], pr_scores[i]
        id_gt = sorted(range(len(gts)), key=lambda j : gts[j], reverse=True)
        id_pr = sorted(range(len(preds)), key=lambda j : preds[j], reverse=True)
        for k in range(4):
            temp_acc4_5  = 0.
            temp_acc4_10 = 0.
            for j in range(k+1):
                if gts[id_pr[j]] >= gts[id_gt[4]]:
                    temp_acc4_5 += 1.0
                if gts[id_pr[j]] >= gts[id_gt[9]]:
                    temp_acc4_10 += 1.0
            acc4_5[k]  += (temp_acc4_5 / (k+1.0))
            acc4_10[k] += ((temp_acc4_10) / (k+1.0))
        sample_cnt += 1
    acc4_5  = [i / sample_cnt for i in acc4_5]
    acc4_10 = [i / sample_cnt for i in acc4_10]
    avg_acc4_5  = sum(acc4_5)  / len(acc4_5)
    avg_acc4_10 = sum(acc4_10) / len(acc4_10)
    return avg_acc4_5, avg_acc4_10


def test(network, test_loader, dataset_name):
    network.eval()

    print('\n')
    print('='*5, f'Evaluating on {dataset_name} dataset', '='*5)
    srcc_list = []
    gt_scores = []
    pr_scores = []
    count = 0
    with torch.no_grad():
        for id, sample in enumerate(test_loader):
            image = sample['image']
            bboxs = sample['bbox']
            MOS = sample['MOS']
            rcnn_bboxes = sample['rcnn_bbox']

            count += image.shape[0]
            
            MOS_numpy = []
            for i in range(len(MOS)):
                MOS_numpy.append(MOS[i].reshape(-1).cpu().numpy())
            
            MOS = MOS_numpy
            roi = []
            rcnn_roi = []

            for idx in range(0,len(bboxs['xmin'])):
                roi.append((0, bboxs['xmin'][idx],bboxs['ymin'][idx],bboxs['xmax'][idx],bboxs['ymax'][idx]))
            for idx in range(len(rcnn_bboxes['xmin'])):
                rcnn_roi.append((0, rcnn_bboxes['xmin'][idx],rcnn_bboxes['ymin'][idx],rcnn_bboxes['xmax'][idx],rcnn_bboxes['ymax'][idx]))

            if cuda:
                image = Variable(image.cuda())
                roi = Variable(torch.Tensor(roi))
                rcnn_roi = Variable(torch.Tensor(rcnn_roi))
            else:
                image = Variable(image)
                roi = Variable(roi)
                rcnn_roi = Variable(rcnn_roi)

            pre_scores = network(image, roi, rcnn_roi)
            pre_scores = pre_scores.cpu().detach().numpy().reshape(-1)
            srcc_list.append(spearmanr(MOS, pre_scores)[0])
            gt_scores.append(MOS)
            pr_scores.append(pre_scores)
    srcc = sum(srcc_list) / len(srcc_list)
    acc5, acc10 = compute_acc(gt_scores, pr_scores)
    print('Test on GAICD {} images, SRCC={:.3f}, acc5={:.3f}, acc10={:.3f}'.format(count, srcc, acc5, acc10))

    return srcc, acc5, acc10


if __name__ == '__main__':
    main()
    