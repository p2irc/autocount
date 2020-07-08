"""

Unsupervised segmentation by backpropagation method from https://github.com/kanezaki/pytorch-unsupervised-segmentation
reproduced here with necessary changes under the MIT license. Please cite:

Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), 2018.

"""

import numpy as np
import cv2
import argparse
import os
from joblib import dump

from skimage import segmentation

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float,
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
# Added for AutoCount
parser.add_argument('--target', metavar='DIRNAME',
                    help='target directory for counting')
parser.add_argument('--output', metavar='FILENAME',
                    help='output file for segmentations')
parser.add_argument('--filetype', default='jpg', type=str,
                    help='png or jpg for images')
args = parser.parse_args()


def write_activation_images(output, im_target):
    print('Relevant channels:')
    print(np.unique(im_target))

    print('Outputting channel images...')

    for i in np.unique(im_target):
        c = output[i, :, :]
        c = c / np.max(c) * 255.
        cv2.imwrite('output-{0}.png'.format(i), c)


def get_image_filenames(dir_path, file_type):
    all_image_filenames = []

    if file_type == 'png' or file_type == 'PNG':
        ext = '.png'
    else:
        ext = '.jpg'

    for root, dirnames, filenames in os.walk(dir_path):

        for filename in [filename for filename in filenames if filename.endswith(ext) or filename.endswith(ext.upper())]:
            all_image_filenames.append(os.path.join(dir_path, filename))

    return all_image_filenames


def slic(im):
    # slic
    labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
    labels = labels.reshape(im.shape[0] * im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    return l_inds


# Match images and labels
all_image_paths = get_image_filenames(args.target, file_type=args.filetype)
sample_index = 0

num_samples = len(all_image_paths)
print('Num samples: {0}'.format(num_samples))


# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# train
model = MyNet( 3 )
if use_cuda:
    model.cuda()
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))

data = None
target = None

for batch_idx in range(args.maxIter):
    # Get next sample
    if use_cuda:
        torch.cuda.empty_cache()

    del data
    del target

    sample_filename = all_image_paths[sample_index]

    sample_index = sample_index + 1 if sample_index < num_samples - 1 else 0

    # Load it up
    im = cv2.imread(sample_filename)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    l_inds = slic(im)

    # forwarding
    optimizer.zero_grad()

    output = model( data )[0]

    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()

    nLabels = len(np.unique(im_target))

    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
       
        cv2.imshow( "output", im_target_rgb )
        cv2.waitKey(10)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[ l_inds[ i ] ]
        u_labels_per_sp = np.unique( labels_per_sp )
        hist = np.zeros( len(u_labels_per_sp) )
        for j in range(len(hist)):
            hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
        im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
    target = torch.from_numpy( im_target )

    if use_cuda:
        target = target.cuda()
    target = Variable( target )
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# User input
output = model(data)[0]
output = nn.functional.softmax(output, dim=2)
output = output.data.cpu().numpy()

write_activation_images(output, target.cpu().numpy())

print('Channel images output to working directory.')
selected_channel = input("Which channel contains the organs? ")
selected_channel = int(selected_channel)

print('Selected channel {0}'.format(selected_channel))

# Save the images in memory

image_dict = {}

for filename in all_image_paths:
    if use_cuda:
        torch.cuda.empty_cache()

    del data

    im = cv2.imread(filename)

    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    
    output = model(data)[0]

    output = output.permute( 1, 2, 0 )

    output = nn.functional.softmax(output, dim=2)

    im_target = output.data

    if use_cuda:
        im_target = im_target.cpu().numpy()
    else:
        im_target = im_target.numpy()

    image_dict[filename] = im_target[:, :, selected_channel]

dump(image_dict, args.output)

print('Output file written.')
