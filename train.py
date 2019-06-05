# adapted from https://raw.githubusercontent.com/zijundeng/pytorch-semantic-segmentation/master/train/cityscapes-psp_net/train.py
import datetime
import os

import numpy as np
import torchvision.transforms as standard_transforms
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import cityscapes
from model.dfanet import DFANet
from utils.misc import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

ckpt_path = './checkpoints/'

args = {
    'train_batch_size': 48,
    'lr': 2e-1,
    'lr_decay': 0.9,
    'max_iter': 40000,
    'longer_size': 2048,
    'crop_size': 1024,
    'stride_rate': 2 / 3.,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'snapshot': '',
    'print_freq': 10,
    'val_freq': 400
}


def main():
    net = DFANet(n_classes=cityscapes.num_classes)

    if len(args['snapshot']) == 0:
        # net.load_state_dict(torch.load(os.path.join(ckpt_path, 'xx.pth')))
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'iter': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0,
                               'fwavacc': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'iter': int(split_snapshot[3]),
                               'val_loss': float(split_snapshot[5]), 'acc': float(split_snapshot[7]),
                               'acc_cls': float(split_snapshot[9]),'mean_iu': float(split_snapshot[11]),
                               'fwavacc': float(split_snapshot[13])}
    net.cuda().train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomSized(args['longer_size']),
        joint_transforms.RandomHorizontallyFlip()
    ])
    sliding_crop = joint_transforms.SlidingCrop(args['crop_size'], args['stride_rate'], cityscapes.ignore_label)
    train_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()

    train_set = cityscapes.CityScapes('fine', 'train', joint_transform=train_joint_transform, sliding_crop=sliding_crop,
                                      transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
    val_set = cityscapes.CityScapes('fine', 'val', transform=val_input_transform, sliding_crop=sliding_crop,
                                    target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=cityscapes.ignore_label).cuda()

    optimizer = optim.SGD(params=net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, 'opt_' + args['snapshot'])))

    check_mkdir(ckpt_path)
    open(os.path.join(ckpt_path, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    train(train_loader, net, criterion, optimizer, curr_epoch, args, val_loader)


def train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader):
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']

            inputs, gts, _ = data
            assert len(inputs.size()) == 5 and len(gts.size()) == 4
            inputs.transpose_(0, 1)
            gts.transpose_(0, 1)

            assert inputs.size()[3:] == gts.size()[2:]
            slice_batch_pixel_size = inputs.size(1) * inputs.size(3) * inputs.size(4)

            for inputs_slice, gts_slice in zip(inputs, gts):
                inputs_slice = Variable(inputs_slice).cuda()
                gts_slice = Variable(gts_slice).cuda()

                optimizer.zero_grad()
                outputs, aux = net(inputs_slice)
                assert outputs.size()[2:] == gts_slice.size()[1:]
                assert outputs.size()[1] == cityscapes.num_classes

                main_loss = criterion(outputs, gts_slice)
                aux_loss = criterion(aux, gts_slice)
                loss = main_loss + 0.4 * aux_loss
                loss.backward()
                optimizer.step()

                train_main_loss.update(main_loss.data[0], slice_batch_pixel_size)
                train_aux_loss.update(aux_loss.data[0], slice_batch_pixel_size)

            curr_iter += 1

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [train aux loss %.5f]. [lr %.10f]' % (
                    curr_epoch, i + 1, len(train_loader), train_main_loss.avg, train_aux_loss.avg,
                    optimizer.param_groups[1]['lr']))
            if curr_iter >= train_args['max_iter']:
                return
            if curr_iter % train_args['val_freq'] == 0:
                validate(val_loader, net, criterion, optimizer, curr_epoch, i + 1, train_args)
        curr_epoch += 1


def validate(val_loader, net, criterion, optimizer, epoch, iter_num, train_args):
    # the following code is written assuming that batch size is 1
    net.eval()

    val_loss = AverageMeter()

    gts_all = np.zeros((len(val_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
    predictions_all = np.zeros((len(val_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
    for vi, data in enumerate(val_loader):
        input, gt, slices_info = data
        assert len(input.size()) == 5 and len(gt.size()) == 4 and len(slices_info.size()) == 3
        input.transpose_(0, 1)
        gt.transpose_(0, 1)
        slices_info.squeeze_(0)
        assert input.size()[3:] == gt.size()[2:]

        count = torch.zeros(args['longer_size'] / 2, args['longer_size']).cuda()
        output = torch.zeros(cityscapes.num_classes, args['longer_size'] / 2, args['longer_size']).cuda()

        slice_batch_pixel_size = input.size(1) * input.size(3) * input.size(4)

        for input_slice, gt_slice, info in zip(input, gt, slices_info):
            input_slice = Variable(input_slice).cuda()
            gt_slice = Variable(gt_slice).cuda()

            output_slice = net(input_slice)
            assert output_slice.size()[2:] == gt_slice.size()[1:]
            assert output_slice.size()[1] == cityscapes.num_classes
            output[:, info[0]: info[1], info[2]: info[3]] += output_slice[0, :, :info[4], :info[5]].data
            gts_all[vi, info[0]: info[1], info[2]: info[3]] += gt_slice[0, :info[4], :info[5]].data.cpu().numpy()

            count[info[0]: info[1], info[2]: info[3]] += 1

            val_loss.update(criterion(output_slice, gt_slice).data[0], slice_batch_pixel_size)

        output /= count
        gts_all[vi, :, :] /= count.cpu().numpy().astype(int)
        predictions_all[vi, :, :] = output.max(0)[1].squeeze_(0).cpu().numpy()

        print('validating: %d / %d' % (vi + 1, len(val_loader)))

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)
    if val_loss.avg < train_args['best_record']['val_loss']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['iter'] = iter_num
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        torch.save(net.state_dict(), os.path.join(ckpt_path, 'DFANet_best.pth.tar'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, 'opt_' + 'DFANet_best.pth.tar'))
    torch.save(net.state_dict(), os.path.join(ckpt_path, 'DFANet.pth.tar'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, 'opt_' + 'DFANet.pth.tar'))

    print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [iter %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, iter_num, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d], '
          '[iter %d]' % (train_args['best_record']['val_loss'], train_args['best_record']['acc'],
                         train_args['best_record']['acc_cls'], train_args['best_record']['mean_iu'],
                         train_args['best_record']['fwavacc'], train_args['best_record']['epoch'],
                         train_args['best_record']['iter']))

    print('-----------------------------------------------------------------------------------------------------------')

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()