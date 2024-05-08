import torch.nn as nn
from .base_layers import *
from ..DCEIFlow.DCEIFlow import DCEIFlow
from ..ERAFT.eraft import ERAFT
from ..utils.flow_utils import FrameWarp

class CistaNet(nn.Module):
     def __init__(self, base_channels=64, depth=5, num_bins=5):
          super(CistaNet, self).__init__()
          '''
               CISTA network for events-to-video reconstruction
          '''
          self.model_mode = model_mode
          self.num_bins = num_bins
          self.depth = depth

          
          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1, groups=1) #We_new
          
           
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1)
          
          self.P0 = ConvLayer(in_channels=base_channels, out_channels=2*base_channels, kernel_size=3,\
               stride=1, padding=1, activation=None, norm=None)#64

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False) 
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])
          

          self.Dg = ConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
               activation='relu', norm=None) 


          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation=None) #activation='relu'
          
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()
          self.prev_rec = None
     
     def reset_states(self):
     #    self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
        self.prev_rec = None

     def forward(self, events):
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
          Outputs:
               dict {'image': rec_I}: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame

          '''
          if self.prev_rec is None:
               self.prev_rec = torch.zeros(events.shape[0], 1, events.shape[2], events.shape[3],
                                         device=events.device)
          

          x_E = self.We(events)
          x_I = self.Wi(self.prev_rec)
          x1 = connect_cat(x_E, x_I) 

          x1 = self.W0(x1) 

          z = self.P0(x1)

          tmp = z.clone()
          for i in range(self.depth):
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z   

          rec_I = self.Dg(z)

          rec_I = self.upsamp_conv(rec_I)
          rec_I = self.sigmoid(self.final_conv(rec_I))
          

          output_dict = {'image': rec_I} 
          self.prev_rec = rec_I.detach()
          
          return output_dict



class CistaTCNet(nn.Module):
     def __init__(self, base_channels=64, depth=5, num_bins=5):
          super(CistaTCNet, self).__init__()
          '''
               CISTA-TC network for events-to-video reconstruction
          '''
          self.num_bins = num_bins
          self.depth = depth
          self.num_states = 2 

          self.one_conv_for_prev = ConvLayer(in_channels=2*base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          self.one_conv_for_cur = ConvLayer(in_channels=2*base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          alpha = nn.Parameter(torch.Tensor([0.001*np.random.rand(2*base_channels, 1,1)]))
          self.alpha = nn.ParameterList([ alpha for i in range(self.depth)])

               
          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1)
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1)

          self.P0 = ConvLayer(in_channels=base_channels, out_channels=2*base_channels, kernel_size=3,\
               stride=1, padding=1)#64

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False)
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])

          self.Dg = RecurrentConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
                    activation='relu') 

          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation=None, norm=None)
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()
          
          self.states = None
          self.prev_rec = None
     
     def reset_states(self):
     #    self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
        self.states = None
        self.prev_rec = None
        
     def calc_attention_feature(self, img1, img2, prev_attention_state):# , prev_attention_state):
          # TSA
          S1 = self.sim_layers(img1)
          S2 = self.sim_layers(img2)
          feat1 =  self.one_conv1(S1)
          feat2 = self.one_conv2(S2)
          attention_map = torch.sigmoid(torch.mul(feat1,feat2)) #attention state
          # return attention_map
          if prev_attention_state is None:
               prev_attention_state = torch.ones_like(attention_map)
          attention1 = torch.mul(S1, prev_attention_state)
          attention2 = torch.mul(S2, attention_map)
          return attention1, attention2, attention_map

     def forward(self, events):
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
          Outputs:
               rec_I: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame
          '''
          # input event tensor Ek, Ik-1, Ik,
          if self.prev_rec is None:
               self.prev_rec = torch.zeros(events.shape[0], 1, events.shape[2], events.shape[3],
                                         device=events.device)
          
          if self.states is None:
               self.states = [None]*self.num_states
          states = [] 


          x_E = self.We(events)
          x_I = self.Wi(self.prev_rec)

          x1 = self.W0(connect_cat(x_E, x_I) ) 
          z = self.P0(x1)
          tmp = z
          if self.states[0] is None:
              self.states[0] = torch.zeros_like(z)
          
          one_ch_prev_z = self.one_conv_for_prev(self.states[0])
          for i in range(self.depth):
               one_ch_cur_z = self.one_conv_for_cur(tmp)
               attention_map = torch.sigmoid(torch.mul(one_ch_prev_z, one_ch_cur_z))
               temporal_z = attention_map*torch.mul((self.states[0]-tmp), self.alpha[i])
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z + temporal_z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z      

          states.append(z)
          
          rec_I, state = self.Dg(z, self.states[-1])
          states.append(state)

          rec_I = self.upsamp_conv(rec_I)
          rec_I = self.final_conv(rec_I)
          rec_I = self.sigmoid(rec_I)
          
          self.states = states

          output_dict = {'image': rec_I} 
          self.prev_rec = rec_I.detach()
          
          return output_dict
     


class CistaLSTCNet(nn.Module):
     def __init__(self, base_channels=64, depth=5, num_bins=5):
          super(CistaLSTCNet, self).__init__()
          '''
               CISTA-LSTC network for events-to-video reconstruction
          '''
          self.model_mode = model_mode
          self.num_bins = num_bins
          self.depth = depth

          self.num_states = 3 
          
          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1, groups=1) #We_new
          
           
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1)
          
          
          self.P0 = ConvLSTC(x_size=base_channels, z_size=2*base_channels, output_size=2*base_channels, kernel_size=3) 

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False) 
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])
          
          
          self.Dg = RecurrentConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
               activation='relu')

          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation='relu') #activation='relu'
          
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()
          self.states = None
          self.prev_rec = None
     
     def reset_states(self):
     #    self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
        self.states = None
        self.prev_rec = None

     def forward(self, events): #, prev_image, prev_states
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
          Outputs:
               rec_I: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame
          '''
          if self.prev_rec is None:
               self.prev_rec = torch.zeros(events.shape[0], 1, events.shape[2], events.shape[3],
                                         device=events.device)
          
          if self.states is None:
               self.states = [None]*self.num_states
          states = [] 

          x_E = self.We(events)
          x_I = self.Wi(self.prev_rec)
          x1 = connect_cat(x_E, x_I) 

          x1 = self.W0(x1) 

          if self.model_mode in ['cista']:
               z = self.P0(x1)
          else:
               z, state = self.P0(x1, self.states[-2], self.states[0] if self.states[0] is not None else None)
               states.append(state)
          tmp = z.clone()
          for i in range(self.depth):
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z   
          states.append(z)

          if self.model_mode == 'cista':
               rec_I = self.Dg(z)
          else:
               rec_I, state = self.Dg(z, self.states[-1])
               states.append(state)
          
          rec_I = self.upsamp_conv(rec_I)
 
          rec_I = self.sigmoid(self.final_conv(rec_I))
          
          self.states = states

          output_dict = {'image': rec_I} 
          self.prev_rec = rec_I.detach()
          
          return output_dict



class BaseFlowRec(nn.Module):
     def __init__(self):
          super(BaseFlowRec, self).__init__()
          self.num_bins = 5 #args.num_bins #['num_bins']
          self.frame_warp = FrameWarp(mode='forward') # kwargs['warp_mode']
          self.fix_net_name = None
          self.scale_factor = 0.5
   
          self.cista_net = CistaLSTCNet(num_bins=self.num_bins)
          self.event_flownet = None #TODO
          
     def fix_params(self, net_name):
          self.fix_net_name = net_name
          if net_name == 'rec':
               for param in self.cista_net.parameters():
                    param.requires_grad = False
               for param in self.event_flownet.parameters():
                    param.requires_grad = True 
               self.event_flownet.train()
          elif net_name == 'flow':
               for param in self.event_flownet.parameters():
                    param.requires_grad = False
               for param in self.cista_net.parameters():
                    param.requires_grad = True
               self.event_flownet.eval() #freeze_bn()
               self.cista_net.train()
          else:
               assert net_name in ['flow', 'rec']
   
     def reset_states(self):
        self.cista_net.reset_states()


# single GPU
class DCEIFlowCistaNet(BaseFlowRec):
     '''CISTA-EIFlow: CISTA-LSTC + DCEIFlow'''
     def __init__(self): #args
          super(DCEIFlowCistaNet, self).__init__() #args
          self.event_flownet = DCEIFlow(num_bins=self.num_bins) #.to('cuda:1')

     def forward(self, batch_data): #, gt_prev_frame=None, gt_frame=None, gt_flow=None):
          '''
          batch_data: event_voxel, rec_img0; 
                    optional variables for training: event_voxel_bw (optional, but must along with gt_img1), gt_img1
          '''
          # Initialise I_prev
          if self.cista_net.prev_rec is None:
               self.cista_net.prev_rec = torch.zeros(batch_data['event_voxel'].shape[0], 1, batch_data['event_voxel'].shape[2], batch_data['event_voxel'].shape[3],
                                         device=batch_data['event_voxel'].device)
          
          # Estimate flow
          batch_flow = self.event_flownet(event_voxel=batch_data['event_voxel'], image1=self.cista_net.prev_rec)
          flow_final = batch_flow['flow_final']
          
          if self.fix_net_name == 'flow':
               flow_final = flow_final.detach()
               flow_final.requires_grad = False

          # warp input I_prev and Z
          if not flow_final.any():
               warped_I = self.cista_net.prev_rec 
          else:
               self.cista_net.prev_rec = self.frame_warp.warp_frame(self.cista_net.prev_rec, flow_final)
               if self.cista_net.states is not None:
                    downsampled_flow = nn.functional.interpolate(flow_final, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                    self.cista_net.states[1] = self.frame_warp.warp_frame(self.cista_net.states[1], downsampled_flow)
              
          # Reconstruct image
          output_dict = self.cista_net(batch_data['event_voxel'])
          output_dict['flow'] = batch_flow['flow_final'].detach()

          return output_dict #(image, flow)


# single GPU
class ERAFTCistaNet(BaseFlowRec):
     '''CISTA-ERAFT: CISTA-LSTC + ERAFT'''
     def __init__(self):
          super(ERAFTCistaNet, self).__init__()
          self.event_flownet = ERAFT(self.num_bins)

     def forward(self, batch_data):
          '''
          batch_data: event_voxel_old, event_voxel (a pair of event voxel grids)
          '''
          if self.cista_net.prev_rec is None:
               self.cista_net.prev_rec = torch.zeros(batch_data['event_voxel'].shape[0], 1, batch_data['event_voxel'].shape[2], batch_data['event_voxel'].shape[3],
                                         device=batch_data['event_voxel'].device)
          batch_flow = self.event_flownet(image1=batch_data['event_voxel_old'], image2=batch_data['event_voxel'])
          flow_final = batch_flow['flow_final']
          
          # valid_mask = (abs(flow_final)>0.1) & (abs(flow_final)<100)
          # flow_final *= valid_mask
          
          if self.fix_net_name == 'flow':
               flow_final = flow_final.detach()
               flow_final.requires_grad = False
          
          if not flow_final.any():
               warped_I = self.cista_net.prev_rec 
          else:
               self.cista_net.prev_rec = self.frame_warp.warp_frame(self.cista_net.prev_rec, flow_final)
               if self.cista_net.states is not None:
                    downsampled_flow = nn.functional.interpolate(flow_final, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                    self.cista_net.states[1] = self.frame_warp.warp_frame(self.cista_net.states[1], downsampled_flow)
               
          output_dict = self.cista_net(batch_data['event_voxel'])
          output_dict['flow'] = batch_flow['flow_final'].detach()

          return output_dict
