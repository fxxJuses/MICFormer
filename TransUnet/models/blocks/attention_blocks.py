import numpy as np
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute3D
from models.blocks.conv_blocks import Conv3DDropoutNormActivation
from models.blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample


class PatchifyVisionMultiheadAttention(nn.Module):
    def __init__(
            self,
            skip_dim, 
            decoder_dim, 
            patch_size=8, 
            embed_size=64, 
            num_heads=8,
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            upscale_attention=TransposeConv3dUpsample,
            dropout=0,
            ):
        super(PatchifyVisionMultiheadAttention, self).__init__()
        '''
        Multiheaded attention block that can be used to enhance nn-Unet. 
        The query is the skip connection while the key and value are come 
        from the decoder path.

        Parameters:
            skip_dim (Tuple[int, int, int, int, int]): (N, C_skip, D_skip, H_skip, W_skip) dimension of the skip connection input
            decoder_dim (Tuple[int, int, int, int, int]): (N, C_enc, D_enc, H_enc, W_enc) dimension of the decoder path input
            patch_size (int or Tuple[int,int,int]): size of the patch in the embedding
            embed_size (int): Total dimension of the model
            num_heads (int): number of attention heads
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            upscale_attention (blocks.conv.downsampling.Downscale): upsampling scheme for attention output
            dropout (float): dropout added to the layer
        '''

        self.patch_embed_skip = Conv3DDropoutNormActivation(
                                                skip_dim[1], 
                                                embed_size, 
                                                patch_size, 
                                                stride=patch_size,  
                                                activation=activation, 
                                                normalization=normalization,
                                                dropout=dropout,
                                            )
        
        self.patch_embed_decoder = Conv3DDropoutNormActivation(
                                                decoder_dim[1], 
                                                embed_size, 
                                                patch_size, 
                                                stride=patch_size,  
                                                activation=activation, 
                                                normalization=normalization,
                                                dropout=dropout,
                                            )
        
        self.vision_attention = VisionMultiheadAttention(
                                                embed_size=embed_size, 
                                                num_heads=num_heads
                                            )
    
        dimension_att = np.array(skip_dim)
        dimension_att[2:] = dimension_att[2:] / np.array(patch_size)

        self.upscale_attention = upscale_attention(
                                                tuple(dimension_att), 
                                                skip_dim, 
                                                embed_size, 
                                                skip_dim[1]
                                            )
        
        self.normalization = normalization(skip_dim[1])
        

    def forward(
            self, 
            skip_path, 
            decoder_path, 
            visualize=False
            ):
        '''
        Parameters:
            skip_path (torch.Tensor): shape is (batch_size, num_skip_channels, skip_depth, skip_height, skip_width)
            decoder_path (torch.Tensor): shape is (batch_size, num_decoder_channels, decoder_depth, decoder_height, decoder_width)
            visualize (bool): whether or not to return attention weights in visualization format

        Returns:
            output (torch.Tensor): (batch_size, num_skip_channels, skip_depth, skip_height, skip_width)
            attention_weights_avg (torch.tensor): attention weigths with shape (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        '''
        #save skip path
        skip_connection = skip_path

        #embed the patches in skip path
        skip_path = self.patch_embed_skip(skip_path)

        #embed the patches in decoder path
        decoder_path = self.patch_embed_decoder(decoder_path)

        #multiheaded attention
        output, attention_weights_avg = self.vision_attention(skip_path, decoder_path, visualize)

        #upscale the attention output to original size
        output = self.upscale_attention(output)

        #add
        output = output + skip_connection

        #normalize
        output = self.normalization(output)

        return output, attention_weights_avg




        

class VisionMultiheadAttention(nn.Module):
    def __init__(
            self, 
            embed_size=64, 
            num_heads=8
            ):
        '''
        Multiheaded attention block that can be used to enhance nn-Unet. 
        The query is the skip connection while the key and value are come 
        from the decoder path.

        Parameters:
            embed_size (int): Total dimension of the model
            num_heads (int): number of attention heads
        '''
        super(VisionMultiheadAttention, self).__init__()

        #positional encoders
        self.query_pos_encoder = PositionalEncodingPermute3D(embed_size)
        self.key_val_pos_encoder = PositionalEncodingPermute3D(embed_size)

        #multihead attention
        self.multihead_attention_block = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)


    def forward(
            self, 
            skip_path, 
            decoder_path, 
            visualize=False
            ):
        '''
        Parameters:
            skip_path (torch.Tensor): shape is (batch_size, num_skip_channels, skip_depth, skip_height, skip_width)
            decoder_path (torch.Tensor): shape is (batch_size, num_decoder_channels, decoder_depth, decoder_height, decoder_width)
            visualize (bool): whether or not to return attention weights in visualization format

        Returns:
            output (torch.Tensor): (batch_size, num_skip_channels, skip_depth, skip_height, skip_width)
            attention_weights_avg (torch.tensor): attention weigths with shape (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        '''
        #remember skip shape (N, C_skip, D_skip, H_skip, W_skip)
        shape_skip = list(skip_path.shape)

        #remember skip shape (N, C_dec, D_dec, H_dec, W_dec)
        shape_decoder = list(decoder_path.shape)

        #add positional encodings to the skip and decoder paths 
        skip_path = skip_path + self.query_pos_encoder(skip_path)
        decoder_path = decoder_path + self.key_val_pos_encoder(decoder_path)

        #flatten the physical dimensions for both the skip path and decoder path
        skip_path = torch.flatten(skip_path, start_dim=2, end_dim=- 1)
        decoder_path = torch.flatten(decoder_path, start_dim=2, end_dim=- 1)

        #invert the length (height + width + depth) dimension with the channels
        skip_path = torch.transpose(skip_path, 1,2)
        decoder_path = torch.transpose(decoder_path, 1,2)

        #attention mechanism
        output, attention_weights_avg = self.multihead_attention_block(skip_path, decoder_path, decoder_path)

        #invert the length (height + width + depth) dimension with the channels
        output = torch.transpose(output, 1,2)

        #reshape output and attention weights
        output = torch.reshape(output, shape_skip)

        #if visualize is set to true then reshape attention weights
        attention_weights_avg = self._reshape_attention(attention_weights_avg, shape_skip, shape_decoder)if visualize == True else None
        
        return output, attention_weights_avg


    def _reshape_attention(
            self, 
            attention_weights_avg, 
            shape_skip, 
            shape_decoder
            ):
        '''
        function that reshapes the attention from (N, L_skip, L_decoder) to (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        Parameters:
            attention_weights_avg (torch.tensor): attention weigths with shape (N, L_skip, L_decoder)
            shape_skip (list[int]): shape of the skip connection (query)
            shape_decoder (list[int]): shape of the decoder context (key and value)

        Returns:
            attention_weights_avg (torch.tensor): attention weigths with shape (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        '''
        #detach from the graph
        attention_weights_avg = attention_weights_avg.detach()

        # (N, L_skip, L_decoder) -> (N, L_decoder, L_skip)
        attention_weights_avg = torch.transpose(attention_weights_avg, -1, -2)
        attention_shape = list(attention_weights_avg.shape)

        #(N, L_decoder, L_skip) -> (N, L_decoder, D_skip, H_skip, W_skip)
        attention_weights_avg = torch.reshape(attention_weights_avg, attention_shape[:-1] + shape_skip[2:])

        #(N, L_decoder, D_skip, H_skip, W_skip) -> (N, D_skip, H_skip, W_skip, L_decoder)
        attention_weights_avg = torch.permute(attention_weights_avg, (0,2,3,4,1))
        attention_shape = list(attention_weights_avg.shape)

        #(N, D_skip, H_skip, W_skip, L_decoder) -> (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        attention_weights_avg = torch.reshape(attention_weights_avg, attention_shape[:-1] + shape_decoder[2:])

        return attention_weights_avg