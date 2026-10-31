# Full-Stokes recovery

`imgproc.py` reconstructs RGB/BGR S0–S3 from RAW sensor mosaics using a measured
analyzer calibration tensor. The Basler processing pipeline supplies the tensor
from the dataset; `data/pcalib_tensor.npy` provides the standalone default.
