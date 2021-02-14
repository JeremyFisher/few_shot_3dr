from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import scipy.io as sio
import functools
import PIL
import logging
import time
import math
import sys, os
import json

from few_shot_3dr.base.train import iou_voxel, iou_shapelayer, my_iou_voxel, my_iou_sdf
from few_shot_3dr.base.ResNet import *
from few_shot_3dr.base.DatasetLoader import *
from few_shot_3dr.base.DatasetCollector import *
from ipdb import set_trace
from scipy.stats import logistic

def vis_slice(sdf_slice):
    from mayavi import mlab
    mlab.surf(sdf_slice, warp_scale='auto')

def calc_iou_np(targets, pred):
    targets = targets.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    bs = targets.shape[0]

    iou_list = []
    for i in range(bs):
        c_intersection = np.logical_and(targets[i], pred[i])
        c_union = np.logical_or(targets[i], pred[i])
        c_iou = (1.0*c_intersection.sum())/c_union.sum()
        iou_list.append(c_iou)
    return iou_list

def update_iou_dic(c_iou, iou_dic, example_ids, ids2cat):
    bs = c_iou.shape[0]
    for i in range(bs):
        example_id = example_ids[i]
        cat_id = ids2cat[example_id]
        iou_dic[cat_id].append(c_iou[i].item())
    return iou_dic


def calc_iou_gpu(targets, pred):
    bs = targets.shape[0]
    c_intersection = targets*pred
    c_intersection = c_intersection.view(bs, -1)
    c_intersection = c_intersection.sum(axis=1)

    c_union = targets+pred
    c_union = c_union.to(bool) # remove 2
    c_union = c_union.view(bs,-1)
    c_union = c_union.sum(axis=1)

    c_iou = c_intersection.to(float) / c_union.to(float)

    return c_iou    



if __name__ == '__main__':


    
    logging.basicConfig(level=logging.INFO)
    logging.info(sys.argv) # nice to have in log files

    name2net        = {'resnet': ResNet}
    net_default     = 'resnet'

    name2dataset    = {'ShapeNet':ShapeNet3DR2N2Collector}
    dataset_default = 'ShapeNet'

    parser = argparse.ArgumentParser(description='Train a  Network')

    # general options
    parser.add_argument('--title',     type=str,            default='few_shot_3dr', help='Title in logs, filename')
    parser.add_argument('--no_cuda',   action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu',       type=int,            default=0,     help='GPU ID if cuda is available and enabled')
    parser.add_argument('--batchsize', type=int,            default=16,    help='input batch size for training (default: 128)') # 
    parser.add_argument('--nthreads',  type=int,            default=1,     help='number of threads for loader') #
    parser.add_argument('--save_inter', type=int,           default=10,    help='Saving interval in epochs (default: 10)')

    # options for dataset
    parser.add_argument('--dataset',          type=str,            default=dataset_default, help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--set',              type=str,            default='test',         help='Validation or test set. (default: val) or train. input samples', choices=['train', 'val', 'test']) # 

    parser.add_argument('--basedir',          type=str,            default='/media/jeremy/data/few_shot_3dr/data/',       help='Base directory for dataset.')

    # options for network
    parser.add_argument('--file',  type=str, default=None, help='Savegame')
    parser.add_argument('--net',   type=str, default=net_default, help=('Network architecture [%s]' % ','.join(name2net.keys())))
    parser.add_argument('--ncomp', type=int, default=1,   help='Number of nested shape layers (default: 1)')

    # other options
    parser.add_argument('--label_type',  type=str, default='vox', help='Type of representation: vox(voxels), sdf or mesh')

    parser.add_argument('--vis_pred', action='store_true', default=False, help='if True, will only print predictions')
    parser.add_argument('--gen_report', type=str, default='no', help='if not <no> then generate iou, imgs or all the above')
    parser.add_argument('--test_ids', type=str, default='0 5 10', help='if not <no> then generate iou, loss, imgs or all the above')    
    parser.add_argument('--cat_id',  type=str, default='02958343', help='cat_id, default is cars 02958343') # 
    parser.add_argument('--path_to_res',  type=str, default='/media/3dr/data/few_shot_3dr/results/', help='path to output results')
    parser.add_argument('--path_to_prep_shapenet',  type=str, default='/media/jeremy/data/few_shot_3dr', help='path to prep shapenet')
    parser.add_argument('--path_to_data',  type=str, default='/media/jeremy/data/few_shot_3dr/data', help='path to output results')


    parser.add_argument('--side',  type=int, default=32, help='Output resolution [if dataset has multiple resolutions.] (default: 128)') #
    parser.add_argument('--p_norm',  type=int, default=1, help='p_norm for paper loss, default =2') 

    parser.add_argument('--no_save',  type=str, default='True', help='calculates iou and prints it')
    parser.add_argument('--k_shot',  type=str, default='full', help='few shot learning. Full means all shot. 0 means just base')
    parser.add_argument('--rand_mode', default='False', type=str,   help='False, or base')
    parser.add_argument('--film_type',  type=str, default='dec_3d', help='which part of the network is filmed? options: dec_3d, enc_2d, both')
    parser.add_argument('--dec_3d_channels',  type=int, default=1, help='chooses nr of channels')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.no_save == 'True':
        args.no_save = True
    else:
        args.no_save = False
 
    assert args.gen_report in ['iou', 'loss', 'imgs', 'all', 'no']

    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
    torch.cuda.set_device(device)
    torch.manual_seed(1)

    if args.file == None:
        args.file = '/home/jeremy/Desktop/DL/few_shot_3dr/results/ShapeNet_10.pth.tar'# FIX
    else:
        pass
    assert os.path.exists(args.file)

    args.no_cuda = True

    if args.vis_pred:
        args.batchsize=1



    savegame = torch.load(args.file, map_location=device)

    args.side = savegame['side']


    try:
        logging.info('Initializing dataset "%s"' % args.dataset)
        Collector = ShapeNet3DR2N2Collector(base_dir=args.basedir,cat_id = args.cat_id,
            representation=args.label_type, side=args.side, p_norm=args.p_norm, k_shot = args.k_shot,
            rand_mode = args.rand_mode)
    except KeyError:
        logging.error('A dataset named "%s" is not available.' % args.dataset)
        exit(1)

    logging.info('Initializing dataset loader')

    #set_trace()

    if args.set == 'val':
        samples = Collector.val()
    elif args.set == 'test':
        samples = Collector.test()
        if args.vis_pred:
            np.random.shuffle(samples)
    elif args.set == 'train':
        samples = Collector.train()


   
    if 1==12:
        np.random.shuffle(samples)
        samples = samples[:1000]

    
    num_samples = len(samples)
    logging.info('Found %d test samples.' % num_samples)
    test_loader = torch.utils.data.DataLoader(DatasetLoader(samples, args.side, args.ncomp, \
        input_transform=transforms.Compose([transforms.ToTensor()])), \
        batch_size=args.batchsize, shuffle=False, num_workers=args.nthreads, \
        pin_memory=True)

  

    samples = []
    net = name2net[args.net](\
            num_input_channels=3, 
            num_initial_channels=savegame['ninf'],
            num_inner_channels=savegame['ngf'],
            num_penultimate_channels=savegame['noutf'], 
            num_output_channels=6*args.ncomp,
            input_resolution=128, 
            output_resolution=savegame['side'],
            num_downsampling=savegame['down'],
            bottleneck_dim = 128,
            num_blocks=savegame['block'],
            film_type=args.film_type,
            dec_3d_channels=args.dec_3d_channels
            ).to(device)
    logging.info(net)
    net.load_state_dict(savegame['state_dict'])
    
    net.eval()

    agg_iou   = 0.
    count     = 0
    results   = torch.zeros(args.batchsize*100, 6, 128,128).to(device)

    ctr_tej=0
    real_ctr = 0
    vox_threshold = 0.4
    iou_list = []
     
    path_to_json = os.path.join(args.basedir, 'ids_dic', 'ids2cat_vox_32.json')
    assert os.path.exists(path_to_json)
    with open(path_to_json, 'r') as fin:
        ids2cat = json.load(fin)

    cat_list = list(set(ids2cat.values()))

    iou_dic = {}
    for c_cat_id in cat_list:
        iou_dic[c_cat_id] = []
   
    t1 = time.time()


    with torch.no_grad():
        for batch_idx, (inputs, targets, cat_nrs, example_ids) in enumerate(test_loader):

            if args.vis_pred:              
                if batch_idx % 24 != 10: # there are 24 views, take only 1
                    continue        
                inputs  = inputs.to(device, non_blocking=True)
                cat_nrs = cat_nrs.to(device, non_blocking=True)
                pred = net(inputs, priors)
                inputs = inputs.cpu()
                pred = pred.cpu()                   
                cat_nrs = cat_nrs.cpu()

                example_nr=0
                c_view = inputs[example_nr].numpy()
                c_label = targets[example_nr].numpy()
                c_cat_nr = cat_nrs[example_nr].numpy()
                
                vis_view(c_view)
                if args.label_type == 'vox':
                    c_pred = logistic.cdf(pred[example_nr].numpy()) > vox_threshold
                    vis_voxels(c_label, color=(0,1,0))
                    vis_voxels(c_pred)

                elif args.label_type == 'sdf':
                    c_pred = pred[example_nr].numpy()
                    vis_sdf(c_label, color=(0,1,0))
                    vis_sdf(c_pred)                    
                elif args.label_type == 'chamfer':
                    path_to_man = os.path.join(args.path_to_prep_shapenet,
                        args.cat_id, 'meshes',
                        example_ids[example_nr],
                        'sim_manifold_50000.obj')
                    c_label_man = trimesh.load_mesh(path_to_man)
                    if type(c_label_man) is list:
                        c_label_man = merge_mesh(c_label_man)
                    vis_mesh(c_label_man.vertices, c_label_man.faces)
                    
                    path_to_sdf = os.path.join(args.path_to_data,
                        'ShapeNetSDF'+str(args.side), args.cat_id,
                        example_ids[example_nr],
                        'sdf_manifold_50000_grid_size_'+str(args.side)+'.npy')
                    c_label_sdf = np.load(path_to_sdf)\
                        .reshape((args.side,)*3)
                    vis_sdf(c_label_sdf, color=(1,0,0))
                    
                    c_pred = pred[example_nr].numpy()
                    vis_sdf(c_pred)                    

                else:
                    print('unknown label type')
                    exit()
                show()
                print('Should I stop? ')
                user_response = input()
                if user_response == 'y':
                    exit()
                else:
                    continue
            elif args.no_save:
                if args.label_type == 'vox':
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    cat_nrs = cat_nrs.to(device, non_blocking=True)

                    pred = net(inputs, cat_nrs)
                    pred = torch.sigmoid(pred) > vox_threshold
                    targets = targets.float()
                    pred = pred.float()
                    c_iou = calc_iou_gpu(targets, pred)

                    iou_dic = update_iou_dic(c_iou, iou_dic, example_ids, ids2cat)



            else: 

                if args.label_type == 'vox':
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    cat_nrs = cat_nrs.to(device, non_blocking=True)
                    pred = net(inputs, cat_nrs)
                    pred = torch.sigmoid(pred) > vox_threshold

                    targets = targets.float()
                    pred = pred.float()


                    targets = targets.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    inputs = inputs.detach().cpu().numpy()
                    for random_nr in range(pred.shape[0]):
                        example_id = example_ids[random_nr]
                        random_cat_id = ids2cat[example_id]
                        path_to_out_gt = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', args.k_shot,
                                random_cat_id, example_id,
                            example_id+'_vox_gt_'+str(args.side)+'_pad_1_view'+str(random_nr)+'.npy')
                        cmd = 'mkdir -p '+os.path.dirname(path_to_out_gt)
                        os.system(cmd)
                        np.save(path_to_out_gt, targets[random_nr])

                        path_to_out_pred = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', args.k_shot,
                            random_cat_id, example_id,                           
                            example_id+'_vox_pred_'+str(args.side)+'_pad_1_view'+str(random_nr)+'.npy')
                        np.save(path_to_out_pred, pred[random_nr])

                        path_to_out_view = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', args.k_shot,
                            random_cat_id, example_id,
                            example_id+'_view_'+str(random_nr)+'.npy')
                        np.save(path_to_out_view, inputs[random_nr])




                elif args.label_type == 'sdf':
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    pred = net(inputs)
                    targets = targets.float()
                    pred = pred.float()
                    targets = targets.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    inputs = inputs.detach().cpu().numpy()

                    for random_nr in range(targets.shape[0]):
                        example_id = example_ids[random_nr]
                        path_to_out_gt = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                            example_id+'_sdf_gt_'+str(args.side)+'_pad_1_view_'+str(random_nr)+'.npy')
                        cmd = 'mkdir -p '+os.path.dirname(path_to_out_gt)
                        os.system(cmd)
                        np.save(path_to_out_gt, targets[random_nr])

                        path_to_out_pred = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                            example_id+'_sdf_pred_'+str(args.side)+'_pad_1_view_'+str(random_nr)+'.npy')
                        np.save(path_to_out_pred, pred[random_nr])

                        path_to_out_view = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                            example_id+'_view'+str(random_nr)+'.npy')
                        np.save(path_to_out_view, inputs[random_nr])
                   

                else:
                    print('unknown label type')
                    exit()               
                if args.gen_report == 'iou' or args.gen_report == 'all':
                    if args.label_type == 'vox':
                        iou, bs = my_iou_voxel(pred, targets)
                        iou_list += iou.cpu().tolist()
                    elif args.label_type == 'sdf':
                        iou, bs = my_iou_sdf(pred, targets)
                        iou_list += iou.cpu().tolist()                       
                    else:
                        print('unkown label type')
                        exit()
                   


    t2 = time.time()
    print('took seconds for whole dataset: ', t2-t1)

    print('BASE CATEGORIES')
    print('cat id       ~ mean  ~ min   ~ max   ~ median    ~ var')
    try:
        print('plane:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['02691156']), np.min(iou_dic['02691156']),
            np.max(iou_dic['02691156']), np.median(iou_dic['02691156']),
            np.var(iou_dic['02691156'])))
    except:
        pass
    try:
        print('car  :       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['02958343']), np.min(iou_dic['02958343']),
            np.max(iou_dic['02958343']), np.median(iou_dic['02958343']),
            np.var(iou_dic['02958343'])))
    except:
        pass
    try:
        print('chair:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['03001627']), np.min(iou_dic['03001627']),
            np.max(iou_dic['03001627']), np.median(iou_dic['03001627']),
            np.var(iou_dic['03001627'])))
    except:
        pass
    try:
        print('monitor:     ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['03211117']), np.min(iou_dic['03211117']),
            np.max(iou_dic['03211117']), np.median(iou_dic['03211117']),
            np.var(iou_dic['03211117'])))
    except:
        pass
    try:
        print('speaker:     ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['03691459']), np.min(iou_dic['03691459']),
            np.max(iou_dic['03691459']), np.median(iou_dic['03691459']),
            np.var(iou_dic['03691459'])))
    except:
        pass
    try:
        print('phone:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['04401088']), np.min(iou_dic['04401088']),
            np.max(iou_dic['04401088']), np.median(iou_dic['04401088']),
            np.var(iou_dic['04401088'])))
    except:
        pass
    try:
        print('table:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['04379243']), np.min(iou_dic['04379243']),
            np.max(iou_dic['04379243']), np.median(iou_dic['04379243']),
            np.var(iou_dic['04379243'])))
    except:
        pass

    print('NOVEL CATEGORIES')
    print('cat id       ~ mean  ~ min   ~ max   ~ median    ~ var')
    try:
        print('bench:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['02828884']), np.min(iou_dic['02828884']),
            np.max(iou_dic['02828884']), np.median(iou_dic['02828884']),
            np.var(iou_dic['02828884'])))
    except:
        pass
    try:
        print('cabinet:     ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['02933112']), np.min(iou_dic['02933112']),
            np.max(iou_dic['02933112']), np.median(iou_dic['02933112']),
            np.var(iou_dic['02933112'])))
    except:
        pass
    try:
        print('lamp:        ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['03636649']), np.min(iou_dic['03636649']),
            np.max(iou_dic['03636649']), np.median(iou_dic['03636649']),
            np.var(iou_dic['03636649'])))
    except:
        pass
    try:
        print('rifle:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['04090263']), np.min(iou_dic['04090263']),
            np.max(iou_dic['04090263']), np.median(iou_dic['04090263']),
            np.var(iou_dic['04090263'])))
    except:
        pass
    try:
        print('couch:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['04256520']), np.min(iou_dic['04256520']),
            np.max(iou_dic['04256520']), np.median(iou_dic['04256520']),
            np.var(iou_dic['04256520'])))
    except:
        pass
    try:
        print('vessel:      ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['04530566']), np.min(iou_dic['04530566']),
            np.max(iou_dic['04530566']), np.median(iou_dic['04530566']),
            np.var(iou_dic['04530566'])))
    except:
        pass
    print('EXTRA NOVEL CATEGORIES')
    print('cat id       ~ mean  ~ min   ~ max   ~ median    ~ var')
    try:
        print('knife:       ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['03624134']), np.min(iou_dic['03624134']),
            np.max(iou_dic['03624134']), np.median(iou_dic['03624134']),
            np.var(iou_dic['03624134'])))
    except:
        pass
    try:
        print('guitar:      ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['03467517']), np.min(iou_dic['03467517']),
            np.max(iou_dic['03467517']), np.median(iou_dic['03467517']),
            np.var(iou_dic['03467517'])))
    except:
        pass
    try:
        print('laptop:      ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['03642806']), np.min(iou_dic['03642806']),
            np.max(iou_dic['03642806']), np.median(iou_dic['03642806']),
            np.var(iou_dic['03642806'])))
    except:
        pass
    try:
        print('bathtub:     ~ %0.3f ~ %0.3f ~ %0.3f ~  %0.3f    ~ %0.3f'\
            %(np.mean(iou_dic['02808440']), np.min(iou_dic['02808440']),
            np.max(iou_dic['02808440']), np.median(iou_dic['02808440']),
            np.var(iou_dic['02808440'])))
    except:
        pass











    iou_json = json.dumps(iou_dic)
    path_to_out = args.file[:-4]+'_rand_'+args.rand_mode+'_iou.txt'
    path_to_out = os.path.join(os.path.dirname(path_to_out), args.k_shot,
        os.path.basename(path_to_out))
    cmd = 'mkdir -p '+os.path.dirname(path_to_out)
    os.system(cmd)
    with open(path_to_out, 'w') as fout:
        fout.write(iou_json)
    pass


