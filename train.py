from __future__ import division

from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from dataset.cocodataset import *
from dataset.vocdataset import *
import os
import argparse
import yaml
import random
from utils.vis_bbox import vis_bbox
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
from utils.eval_tool import eval_detection_voc

VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'person',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    parser.add_argument('--eval_interval', type=int,
                        default=4000, help='interval between evaluations')
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard', help='tensorboard path for logging', type=str, default=None)
    return parser.parse_args()

def evals(model, test_num=100):
    model.eval()

    augmentation = {'LRFLIP': False, 'JITTER': 0, 'RANDOM_PLACING': False,
                        'HUE': 0, 'SATURATION': 0, 'EXPOSURE': 0, 'RANDOM_DISTORT': False}
    dataset = VOCDataset(model_type=['YOLO'],
                          data_dir='../../VOCdevkit/VOC2007',
                          name='test',
                          img_size=416,
                          augmentation=augmentation)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8)

    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii,(img, gt_label, info_img, id_) in tqdm(enumerate(dataloader)):
        # if ii >= test_num:
        #     break
        Tensor = torch.cuda.FloatTensor
        # print(img.shape)
        img = Variable(img.type(Tensor))
        outputs = model(img)
        outputs = postprocess(
            outputs, 80, 0.05, 0.3)
        if outputs[0] is None:
            continue

        bboxes = list()
        classes = list()
        scores =list()


        outputs_=outputs[0].cpu()
        bboxes = np.zeros((1,len(outputs_),4))
        classes = np.zeros((1,len(outputs_)))
        scores = np.zeros((1,len(outputs_)))
        index=0

        # 经过预处理后的图像是归一化后的图像大小
        image=np.transpose(img[0].cpu(), (1,2,0))*255
        image=image.numpy().astype(int)
        image=image[:,:,::-1]

        #先写后读，为了获得opencv的图片格式，否则不能正常绘图
        cv2.imwrite('result/' + str(id_[0]) + '.jpg', image)
        image=cv2.imread('result/' + str(id_[0]) + '.jpg')


        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs_:
            if x1==0 and x2==0:
                continue

            bboxes[0,index,:] = np.abs(np.array([x1, y1, x2, y2]))
            classes[0,index]  = int(cls_pred)
            scores[0,index] = conf.cpu().numpy()
            index+=1

            # abs是防止负值，round四舍五入
            xmin = np.abs(int(round(float(x1))))
            ymin = np.abs(int(round(float(y1))))
            xmax = np.abs(int(round(float(x2))))
            ymax = np.abs(int(round(float(y2))))

            # 绘图 rectangle(image,(左上角坐标)，(右下角坐标),颜色,粗细)
            cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (0, 0, 255), 5)
            cv2.putText(image, VOC_BBOX_LABEL_NAMES[int(cls_pred)], (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
            cv2.putText(image, str(conf.cpu().numpy()), (xmin + 30, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)

        gt_label_=gt_label[0]
        gt_bbox = np.zeros((1,len(gt_label_),4))
        gt_class = np.zeros((1,len(gt_label_)))
        index=0
        for cls, x1, y1, x2, y2 in gt_label_:
            if x1==0 and x2==0:
                continue

            # abs是防止负值，round四舍五入
            # 标签的格式【x,y,w,h】 其中x，y为bounding box中心点坐标
            xmin = np.abs(int(round(float(x1*416-x2*416/2.0))))
            ymin = np.abs(int(round(float(y1*416-y2*416/2.0))))
            xmax = np.abs(int(round(float(x2*416))))+xmin
            ymax = np.abs(int(round(float(y2*416))))+ymin

            gt_bbox[0,index,:]=[xmin,ymin,xmax,ymax]
            gt_class[0,index]=cls
            index+=1
            cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (255, 0, 255), 5)
            cv2.putText(image, VOC_BBOX_LABEL_NAMES[int(cls)], (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 255), 1)

        cv2.imwrite('result/' + str(id_[0]) + '.jpg', image)

        #去除全零部分
        gt_bbox = gt_bbox[:,0:index,:]
        gt_class = gt_class[:,0:index]

        # 评价函数需要的输入格式
        # bbox: list(list([n,4]))
        # label:list(list([n]))
        # score:list(list([n]))
        gt_bboxes+=(list(gt_bbox))
        gt_labels+=(list(gt_class))

        pred_bboxes+=(list(bboxes))
        pred_labels+=(list(classes))
        pred_scores+=(list(scores))


    # 这个评价函数是返回ap 和map值 其中传入的pred_bboxes格式为3维的数组的list格式，
    # 也就是说每个list都是一个3维数组(有batch的考量) bbox: list(list([n,4]))
    # 其他的同理                                 label:list(list([n]))
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)

    model.train()
    return result

def draw(model,datatype,imgsize):
    model.eval()
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()
    id_list_file = os.path.join(
            '../../VOCdevkit/VOC2007', 'ImageSets/Main/{0}.txt'.format('test'))
    ids = [id_.strip() for id_ in open(id_list_file)]
    for i in tqdm(range(len(ids))):
        if datatype=='voc':
            img_file = os.path.join('../../VOCdevkit/VOC2007', 'JPEGImages', ids[i] + '.jpg')
        else:
            img_file = os.path.join('COCO', 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)
        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))

        img, info_img_t = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = Variable(img.type(torch.cuda.FloatTensor))
        outputs = model(img)

        outputs = postprocess(outputs, 80, 0.5, 0.5)
        # imgs.shape : torch.Size([1, 3, 608, 608])
        # outputs[0].shape :torch.Size([3, 7])
        # targets.shape :torch.Size([1, 50, 5])
        # print(outputs)
        if outputs[0] is not None:
            bboxes = list()
            classes = list()
            colors = list()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

                cls_id = coco_class_ids[int(cls_pred)]
                box = yolobox2label([y1, x1, y2, x2], info_img_t)
                bboxes.append(box)
                classes.append(cls_id)
                colors.append(coco_class_colors[int(cls_pred)])
            vis_bbox(
                img_raw, bboxes, label=classes, label_names=coco_class_names,
                instance_colors=colors, linewidth=2)
            plt.savefig('draw/'+ids[i]+'.jpg')
    model.train()

def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    print("successfully loaded config file: ", cfg)

    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision
    datatype = cfg['TRAIN']['DATATYPE']
    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Learning rate setup
    def burnin_schedule(i):
        if i < burn_in:
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # Initiate model
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    elif args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        print("using cuda")
        model = model.cuda()

    if args.tfboard:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard)

    model.train()
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()
    imgsize = cfg['TRAIN']['IMGSIZE']
    if datatype=='voc':
        dataset = VOCDataset(model_type=cfg['MODEL']['TYPE'],
                              data_dir='../../VOCdevkit/VOC2007',
                              img_size=imgsize,
                              augmentation=cfg['AUGMENTATION'],
                              debug=args.debug)
        print('load voc dataset successfully')
    else:
        dataset = COCODataset(model_type=cfg['MODEL']['TYPE'],
                      data_dir='COCO/',
                      img_size=imgsize,
                      augmentation=cfg['AUGMENTATION'],
                      debug=args.debug)
        print('load COCO dataset successfully')

        evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                                     data_dir='COCO/',
                                     img_size=cfg['TEST']['IMGSIZE'],
                                     confthre=cfg['TEST']['CONFTHRE'],
                                     nmsthre=cfg['TEST']['NMSTHRE'])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
    dataiterator = iter(dataloader)


    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # optimizer setup
    # set weight decay only on conv.weight
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{'params': value, 'weight_decay': decay
                        * batch_size * subdivision}]
        else:
            params += [{'params': value, 'weight_decay': 0.0}]
    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size * subdivision)

    iter_state = 0

    if args.checkpoint:
        if 'optimizer_state_dict' in state.keys():
            optimizer.load_state_dict(state['optimizer_state_dict'])
            iter_state = state['iter'] + 1
    #学习率控制 Sets the learning rate of each parameter group to the initial lr times a given function. When last_epoch=-1, sets initial lr as lr.
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # result=evals(model)
    # print(result)

    # start training loop
    # print('args.eval_interval',args.eval_interval)
    for iter_i in range(iter_state, iter_size + 1):
        if iter_i % (args.eval_interval*2) == 0 and iter_i > 0:
            if datatype=='voc':
                result=evals(model)
                print(result)
            else:
                ap50_95, ap50 = evaluator.evaluate(model)
                print(ap50_95, ap50)
                model.train()
                if args.tfboard:
                    tblogger.add_scalar('val/COCOAP50', ap50, iter_i)
                    tblogger.add_scalar('val/COCOAP50_95', ap50_95, iter_i)

        if iter_i % (40000) == 0 and iter_i > 0:
            draw(model,datatype,imgsize)
        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, info_img, id_ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, info_img, id_ = next(dataiterator)  # load a batch
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            loss = model(imgs, targets)
            loss.backward()

        optimizer.step()
        scheduler.step()


        if iter_i % 10 == 0:
            # logging

            current_lr = scheduler.get_lr()[0] * batch_size * subdivision
            print('[Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, total %f, imgsize %d]'
                  % (iter_i, iter_size, current_lr,
                     model.loss_dict['xy'], model.loss_dict['wh'],
                     model.loss_dict['conf'], model.loss_dict['cls'],
                     model.loss_dict['l2'], imgsize))

            if args.tfboard:
                tblogger.add_scalar('train/total_loss',
                                    model.loss_dict['l2'], iter_i)

            # random resizing
            # 变输入大小,利用了yolov3网络的全卷积，使得模型不受图像大小的改变而影响参数。
            if random_resize:
                imgsize = (random.randint(0, 9) % 10 + 10) * 32
                dataset.img_shape = (imgsize, imgsize)
                dataset.img_size = imgsize
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
                dataiterator = iter(dataloader)

        if iter_i % 100 == 0:
            model.eval()
            if datatype=='voc':
                img_file = os.path.join('../../VOCdevkit/VOC2007', 'JPEGImages', id_[0] + '.jpg')
            else:
                img_file = os.path.join('COCO', 'train2017',
                                        '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)
            img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
            # print(img_raw.shape)
            # print(img_raw)
            # print(imgs)
            img, info_img_t = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
            img = np.transpose(img / 255., (2, 0, 1))
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = Variable(img.type(torch.cuda.FloatTensor))
            outputs = model(img)
            #outputs.shape : torch.Size([1, 12348, 85])
            outputs = postprocess(outputs, 80, 0.5, 0.5)
            # imgs.shape : torch.Size([1, 3, 608, 608])
            # outputs[0].shape :torch.Size([3, 7])
            # targets.shape :torch.Size([1, 50, 5])
            # print(outputs)
            if outputs[0] is not None:
                bboxes = list()
                classes = list()
                colors = list()
                # print(info_img_t)
                info_img=tuple(info_img)
                # print(info_img)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

                    cls_id = coco_class_ids[int(cls_pred)]
                    # print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
                    # print('\t+ Label: %s, Conf: %.5f' %
                    #       (coco_class_names[cls_id], cls_conf.item()))
                    # print([y1, x1, y2, x2])
                    box = yolobox2label([y1, x1, y2, x2], info_img_t)
                    bboxes.append(box)
                    classes.append(cls_id)
                    colors.append(coco_class_colors[int(cls_pred)])
                vis_bbox(
                    img_raw, bboxes, label=classes, label_names=coco_class_names,
                    instance_colors=colors, linewidth=2)
                plt.savefig('output/'+str(iter_i)+'.jpg')
            model.train()

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            torch.save({'iter': iter_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        },
                       os.path.join(args.checkpoint_dir, "snapshot" + str(iter_i) + ".ckpt"))
    if args.tfboard:
        tblogger.close()





if __name__ == '__main__':
    main()
