## AutoCount

This repository contains a Python 3.7 implementation for the AutoCount unsupervised object counting system. Please cite:

Ubbens, J. R., Ayalew, T. W., Shirtliffe, S. J., Josuttes, A., Pozniak, C. J. & Stavness, I. (2020). AutoCount: Unsupervised Segmentation and Counting of Plant Organs in Field Images. In the European Conference on Computer Vision (ECCV) Workshops, 2020. August 2020.

The first script uses a modified implementation of the segmentation technique proposed in Asako Kanezaki. *Unsupervised Image Segmentation by Backpropagation.* IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.

### Usage

The first script (`step1.py`) performs unsupervised segmentation on a chosen image from the dataset (specified by the `input` argument). Example:

`python step1.py --target ./dataset --output dataset_softmax --nChannel 32 --minLabels 8`

All of the arguments from the [author's implementation](https://github.com/kanezaki/pytorch-unsupervised-segmentation) are preserved. The `--target` and `--output` arguments are added. Following the training of the segmentation network, a file specified by `--output` containing probability density maps for all of the images in the `--target` folder is written.

The second step (`step2.py`) uses this output file to find an optimal thresholding and segmentation. The first argument is the name of the input file and the second argument is the number of threads (use as many as possible for your system).

`python step2.py dataset_softmax 16`

After the optimization step, the filenames and corresponding predicted object counts are output.
