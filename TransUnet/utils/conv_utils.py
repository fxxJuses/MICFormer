import numpy as np

def conv_output_dim(input_dim, kernel_size, stride, padding, dilation, output_padding=0):
  '''
  Calculates the output dimension of a convolution operation
  Parameters:
    input_dim (int): spacial d imension along which convolution takes place
    num_kernels (int): number of kernels
    kernel_size (int): size of the kernel
    stride (int): stride 
    padding (int): padding
    dilation (int): dilation 
    output_padding (int): output padding (ignored) 

  Returns:
    output_dim (int): new dimension at the output of the convolution operation
  '''
  output_dim = int(np.floor((input_dim + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1))
  return output_dim


def transpose_conv_output_dim(input_dim, kernel_size, stride, padding, dilation, output_padding=0):
  '''
  Calculates the output dimension of a transpose convolution operation
  Parameters:
    input_dim (int): spacial d imension along which convolution takes place
    num_kernels (int): number of kernels
    kernel_size (int): size of the kernel
    stride (int): stride 
    padding (int): padding
    dilation (int): dilation 
    output_padding (int): ouput padding

  Returns:
    output_dim (int): new dimension at the output of the convolution operation
  '''
  output_dim = int((input_dim-1)*stride-2*padding+dilation*(kernel_size-1)+ output_padding + 1)
  return output_dim


def conv3d_output_dim(input_dim, num_kernels, kernel_size, stride, padding, dilation, type='normal', output_padding=0):
  '''
  Calculates the output dimensions of a 3d convolution operation
  Parameters:
    input_dim (Tuple[int, int, int , int, int]): (N, C_in, D_in, H_in, W_in)
    num_kernels (int): number of kernels
    kernel_size (Tuple[int, int, int] or int): size of the kernel along the depth, height and width dimensions respectively
    stride (Tuple[int, int, int] or int): stride along the depth, height and width dimensions respectively
    padding (Tuple[int, int, int] or int): padding along the depth, height and width dimensions respectively
    dilation (Tuple[int, int, int] or int): dilation along the depth, height and width dimensions respectively
    type (str): either 'normal' or 'transpose'. 

  Returns:
    output_dim (Tuple[int, int, int , int, int]): (N, C_out, D_out, H_out, W_out)
  '''
  if not isinstance(kernel_size, tuple):
    assert isinstance(kernel_size, int), "kernel_size must be either int or Tuple[int, int, int]"
    kernel_size = (kernel_size, kernel_size, kernel_size)

  if not isinstance(padding, tuple):
    assert isinstance(padding, int), "padding must be either int or Tuple[int, int, int]"
    padding = (padding, padding, padding)

  if not isinstance(dilation, tuple):
    assert isinstance(dilation, int), "dilation must be either int or Tuple[int, int, int]"
    dilation = (dilation, dilation, dilation)

  if not isinstance(stride, tuple):
    assert isinstance(stride, int), "stride must be either int or Tuple[int, int, int]"
    stride = (stride, stride, stride)

  assert isinstance(stride[0], int) and isinstance(stride[1], int) and isinstance(stride[2], int), "must be either int or Tuple[int, int, int]"
  
  if type == 'normal':
    dim_compute = conv_output_dim
  elif type == 'transpose':
    dim_compute = transpose_conv_output_dim
  else:
    raise NotImplementedError(f"{type} is not implemented")

  spacial_dims = [dim_compute(input_dim[2+i], kernel_size[i], stride[i], padding[i], dilation[i], output_padding=output_padding) for i in range(3)]
  output_dim = (input_dim[0], num_kernels, spacial_dims[0], spacial_dims[1], spacial_dims[2])
  return output_dim