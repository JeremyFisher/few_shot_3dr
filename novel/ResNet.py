import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import log, floor

from ipdb import set_trace

NR_OF_CLASSES = 21

class BatchNorm_film(nn.Module):
    def __init__(self,num_features, nr_of_classes, eps=1e-05, momentum=0.1, affine=True,
            track_running_stats=True):
        super().__init__()
        self.bn =  nn.BatchNorm2d(num_features, eps, momentum, affine,
                track_running_stats)


    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.bn(x)
        return out,class_nr

class Reflection_film(nn.Module):
    def __init__(self,padding, nr_of_classes):
        super().__init__()
        self.reflection =  nn.ReflectionPad2d(padding)

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.reflection(x)
        return out,class_nr

class Conv2d_film(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nr_of_classes,
            stride=1, padding=0,dilation=1,groups=1, bias=True, padding_mode='zeros'):
        super().__init__()

        self.conv2d =  nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                padding,dilation,groups, bias, padding_mode)



    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.conv2d(x)
        return out,class_nr

class Relu_film(nn.Module):
    def __init__(self, nr_of_classes, inplace=False):
        super().__init__()
        self.relu =  nn.ReLU(inplace=inplace)

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.relu(x)
        return out,class_nr





class InitialLayer(nn.Module):
    def __init__(self, num_input_channels, num_initial_channels, kernel_size=3, padding=0,
            nr_of_classes=NR_OF_CLASSES):

        super().__init__()
        self.refl1 = nn.ReflectionPad2d(1)
        self.refl2 = nn.ReflectionPad2d(1)




        self.conv1 = nn.Conv2d(num_input_channels, num_initial_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(num_initial_channels, num_initial_channels, kernel_size=3, padding=0)
        self.relu = nn.ReLU(True)
        self.bn =  ConditionalBatchNorm2d(num_initial_channels, nr_of_classes)
   
    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.refl1(x) # nn.ReflectionPad2d(1)
        out = self.conv1(out) # nn.Conv2d(num_input_channels, num_initial_channels, kernel_size=3, padding=0)
        out = self.bn(out, class_nr) # nn.BatchNorm2d(num_initial_channels)
        out = self.relu(out) # relu
        out = self.refl2(out) # nn.ReflectionPad2d(1)


        out = self.conv2(out) # nn.Conv2d(num_initial_channels, num_initial_channels, kernel_size=3, padding=0)
        return out, class_nr  



class FirstModel(nn.Module):
    def __init__(self, num_features, nr_of_classes=NR_OF_CLASSES):

        super().__init__()
        self.bn = ConditionalBatchNorm2d(num_features, nr_of_classes)

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.bn(x, class_nr)
        return out, class_nr

  


class ResSameBlockFilm(nn.Module):
    def __init__(self, dim, nr_of_classes=NR_OF_CLASSES):

        super(ResSameBlockFilm, self).__init__()
        self.bn1 = ConditionalBatchNorm2d(dim, nr_of_classes) # nn.BatchNorm2d(dim, True)
        self.bn2 = ConditionalBatchNorm2d(dim, nr_of_classes) # nn.BatchNorm2d(dim, True)

        self.relu1 = nn.ReLU(True) # nn.ReLU(True)
        self.relu2 = nn.ReLU(True) # nn.ReLU(True)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1) # nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1) # nn.Conv2d(dim, dim, kernel_size=3, padding=1)



    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.bn1(x, class_nr) # nn.BatchNorm2d(dim, True)

        out = self.relu1(out) # nn.ReLU(True)


        out = self.conv1(out) # nn.Conv2d(dim, dim, kernel_size=3, padding=1)


        out = self.bn2(x, class_nr) # nn.BatchNorm2d(dim, True)       
        out = self.relu2(out) # nn.ReLU(True)

        out = self.conv2(out) # nn.Conv2d(dim, dim, kernel_size=3, padding=1)])


        return x + out, class_nr
    pass    

class ResSameBlock(nn.Module):
    
    def __init__(self, dim):
        super(ResSameBlock, self).__init__()

        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True), \
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),            
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)])

    def forward(self, x):
        return x + self.model(x)
    pass    

class ResSameBlock_film(nn.Module):
    
    def __init__(self, dim):
        super(ResSameBlock_film, self).__init__()

        self.model = nn.Sequential(*[BatchNorm_film(num_features=dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=True, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=dim, out_channels=dim,
                    kernel_size=3, padding=1, nr_of_classes=NR_OF_CLASSES),
            BatchNorm_film(num_features=dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=True, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=dim, out_channels=dim,
                    kernel_size=3, padding=1, nr_of_classes=NR_OF_CLASSES)])

        self.model666 = nn.Sequential(*[nn.BatchNorm2d(dim, True), \
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),            
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)])


    def forward(self, tuple_lolz):
#        set_trace()

        x, class_nr = tuple_lolz

        out1= self.model((x, class_nr))[0]
        out2= self.model666(x)

        return x + self.model((x, class_nr))[0], class_nr
    pass    

class ResSameBlock_film2(nn.Module):
   
    def __init__(self, dim):
        super(ResSameBlock_film2, self).__init__()

        self.model = nn.Sequential(*[ConditionalBatchNorm2d(num_features=dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=True, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=dim, out_channels=dim,
                    kernel_size=3, padding=1, nr_of_classes=NR_OF_CLASSES),
            ConditionalBatchNorm2d(num_features=dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=True, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=dim, out_channels=dim,
                    kernel_size=3, padding=1, nr_of_classes=NR_OF_CLASSES)])

    def forward(self, tuple_lolz):
#        set_trace()

        x, class_nr = tuple_lolz

        return x + self.model((x, class_nr))[0], class_nr
    pass    


class ResDownBlockFilm(nn.Module):
    def __init__(self, dim, num_down, nr_of_classes=NR_OF_CLASSES):

        super(ResDownBlockFilm, self).__init__()
        self.num_down = num_down
        
        self.bn1 = ConditionalBatchNorm2d(dim, nr_of_classes) # nn.BatchNorm2d(dim, True)
        self.relu1 = nn.ReLU(True) # nn.ReLU(False) # TODO BUG????
        self.relu2 = nn.ReLU(True)  # nn.ReLU(True)
        self.conv1 = nn.Conv2d(dim, num_down+dim, kernel_size=3, padding=1, stride=2) # nn.Conv2d(dim, num_down+dim, kernel_size=3, padding=1, stride=2)
        self.bn2 = ConditionalBatchNorm2d(num_down+dim, nr_of_classes) # nn.BatchNorm2d(num_down+dim, True)
        self.conv2 = nn.Conv2d(num_down+dim, num_down+dim, kernel_size=3, padding=1)
        pass

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        xu = x[:,:,::2,::2]
        bs,_,h,w = xu.size()
        out = self.bn1(x) # TODO SN_CHECK
        out = self.relu1(out) # nn.ReLU(False)
        out = self.conv1(out) # nn.Conv2d(dim, num_down+dim, kernel_size=3, padding=1, stride=2)
        out = self.bn2(out) # TODO SN_CHECK
        out = self.relu2(out) # nn.ReLU(True)
        out = self.conv2(out) # nn.Conv2d(num_down+dim, num_down+dim, kernel_size=3, padding=1)
        return torch.cat([xu, x.new_zeros(bs, self.num_down, h, w, requires_grad=False)],1) + out, class_nr
    pass


class ResDownBlock(nn.Module):
    
    def __init__(self, dim, num_down):
        super(ResDownBlock, self).__init__()
        self.num_down = num_down
        
        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True), \
            nn.ReLU(False), 
            nn.Conv2d(dim, num_down+dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_down+dim, True),
            nn.ReLU(True),
            nn.Conv2d(num_down+dim, num_down+dim, kernel_size=3, padding=1)])
        pass

    def forward(self, x):
        xu = x[:,:,::2,::2]
        bs,_,h,w = xu.size()
        return torch.cat([xu, x.new_zeros(bs, self.num_down, h, w, requires_grad=False)],1) + self.model(x)
    pass

class ResDownBlock_film(nn.Module):
    
    def __init__(self, dim, num_down):
        super(ResDownBlock_film, self).__init__()
        self.num_down = num_down
 

        self.model = nn.Sequential(*[BatchNorm_film(num_features=dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=False, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=dim, out_channels=num_down+dim,
                    kernel_size=3, padding=1, stride=2, nr_of_classes=NR_OF_CLASSES),
            BatchNorm_film(num_features=num_down+dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=True, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=num_down+dim, out_channels=num_down+dim,
                    kernel_size=3, padding=1, nr_of_classes=NR_OF_CLASSES)])

        pass

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz

        xu = x[:,:,::2,::2]
        bs,_,h,w = xu.size()


        return torch.cat([xu, x.new_zeros(bs, self.num_down, h, w, requires_grad=False)],1) + self.model((x,class_nr))[0],\
            class_nr
    pass

class ResDownBlock_film2(nn.Module):
    
    def __init__(self, dim, num_down):
        super(ResDownBlock_film2, self).__init__()
        self.num_down = num_down
 

        self.model = nn.Sequential(*[ConditionalBatchNorm2d(num_features=dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=False, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=dim, out_channels=num_down+dim,
                    kernel_size=3, padding=1, stride=2, nr_of_classes=NR_OF_CLASSES),
            ConditionalBatchNorm2d(num_features=num_down+dim, nr_of_classes=NR_OF_CLASSES),
            Relu_film(inplace=True, nr_of_classes=NR_OF_CLASSES),
            Conv2d_film(in_channels=num_down+dim, out_channels=num_down+dim,
                    kernel_size=3, padding=1, nr_of_classes=NR_OF_CLASSES)])

        pass

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz

        xu = x[:,:,::2,::2]
        bs,_,h,w = xu.size()


        return torch.cat([xu, x.new_zeros(bs, self.num_down, h, w, requires_grad=False)],1) + self.model((x,class_nr))[0],\
            class_nr
    pass
  

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, nr_of_classes):
        super().__init__()
#        set_trace()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(nr_of_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_id = tuple_lolz
        out = self.bn(x)
        gamma, beta = self.embed(class_id).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out, class_id

class BsLayer(nn.Module):
    def __init__(self, nr_channels_prev, nr_channels, kernel_size_list,
            nr_of_classes):
        super().__init__()
        self.conv = nn.ConvTranspose3d(nr_channels_prev, nr_channels,
                    kernel_size_list,
                    stride=1, padding=0, output_padding=0, groups=1, bias=True,
                    dilation=1)
        self.bn =  ConditionalBatchNorm3d(nr_channels, nr_of_classes)
        self.relu = nn.ReLU(True)

    def forward(self, tuple_lolz):
#        set_trace()
        x, class_nr = tuple_lolz
        out = self.conv(x)
        out = self.bn(out,class_nr)
        out = self.relu(out)
        return out,class_nr

class ConditionalBatchNorm3d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1,
                affine=False, track_running_stats=True)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, class_id):
#        set_trace()
        out = self.bn(x)
        gamma, beta = self.embed(class_id).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1, 1) * out + beta.view(-1, self.num_features,1,1,1)
        return out

class ResNet(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, num_penultimate_channels, \
        input_resolution, output_resolution, num_initial_channels=16, num_inner_channels=64, \
        num_downsampling=3, num_blocks=6, bottleneck_dim=1024, film_type='dec_3d',
        dec_3d_channels=1):

#        set_trace()

        assert num_blocks >= 0

        super(ResNet, self).__init__()

        relu = nn.ReLU(True)

        if film_type == 'enc_2d' or film_type == 'all':

            model = [ConditionalBatchNorm2d(num_features=num_input_channels, nr_of_classes=NR_OF_CLASSES)]

            # additional down and upsampling blocks to account for difference in input/output resolution
            num_additional_down   = int(log(input_resolution / output_resolution,2)) if output_resolution <= input_resolution else 0
            num_additional_up     = int(log(output_resolution / input_resolution,2)) if output_resolution >  input_resolution else 0

            # number of channels to add during downsampling
            num_channels_down     = int(floor(float(num_inner_channels - num_initial_channels)/(num_downsampling+num_additional_down)))

            # adjust number of initial channels
            num_initial_channels += (num_inner_channels-num_initial_channels) % num_channels_down

            model += [Reflection_film(padding=1, nr_of_classes=NR_OF_CLASSES),
                Conv2d_film(in_channels=num_input_channels, out_channels=num_initial_channels,
                    kernel_size=3, padding=0, nr_of_classes=NR_OF_CLASSES),
                ConditionalBatchNorm2d(num_features=num_initial_channels, nr_of_classes=NR_OF_CLASSES),
                Relu_film(inplace=True, nr_of_classes=NR_OF_CLASSES)]

            model += [Reflection_film(padding=1, nr_of_classes=NR_OF_CLASSES),
                Conv2d_film(in_channels=num_initial_channels, out_channels=num_initial_channels,
                    kernel_size=3, padding=0, nr_of_classes=NR_OF_CLASSES)]



            # downsampling
            for i in range(num_downsampling+num_additional_down):                        
                model += [ResDownBlock_film2(num_initial_channels, num_channels_down)] # FILM
                model += [ResSameBlock_film2(num_initial_channels+num_channels_down)] # FILM
                num_initial_channels += num_channels_down
                pass

            # inner blocks at constant resolution
            for i in range(num_blocks):
                model += [ResSameBlock_film2(num_initial_channels)]
                pass
            model_encoder = model[:20].copy() # model_encoder is bsx512x1x1
            self.model_enc = nn.Sequential(*model_encoder)
       


    
        if 1==1: # just spit out encoder
            # BOTTLENECK
            if dec_3d_channels == 1:
                tmp_init_depth = 8 # nr of channels 
            elif dec_3d_channels == 2:
                tmp_init_depth = 32 # nr of channels 
            elif dec_3d_channels == 3:
                tmp_init_depth = 64 # nr of channels 
            elif dec_3d_channels == 4:
                tmp_init_depth = 128 # nr of channels 
          

            tmp_n_final = 6 # we will first resize to 6x6x6 with 8 channels

            enc_last_dim = num_initial_channels
            model_bottleneck = [nn.Linear(128, bottleneck_dim)] # 

            model_bottleneck += [nn.Linear(bottleneck_dim, tmp_init_depth*tmp_n_final**3)] 
            self.model_bottleneck = nn.Sequential(*model_bottleneck)
            model_cBN_decoder = []

            kernel_size_list = [3,3, 3, 5, 5, 7, 7]
            if dec_3d_channels == 1:
                channels_list =    [8,8, 16, 32, 16, 8, 4, 4] # 1
            elif dec_3d_channels == 2:
                channels_list =  [ 32,  32,  64, 128,  64,  32,  16,  4] # 2 
            elif dec_3d_channels == 3:
                channels_list =  [ 64,  64, 128, 256, 128,  64,  32,  4] # 3
            elif dec_3d_channels == 4:
                channels_list = [128, 128, 128, 128, 128, 128,  64,  4] # 4

            
            for deconv_iter in range(len(kernel_size_list)):
                if film_type == 'dec_3d' or film_type == 'all':
                    model_cBN_decoder += [BsLayer(channels_list[deconv_iter],
                        channels_list[deconv_iter+1], kernel_size_list[deconv_iter], NR_OF_CLASSES)]

                else:
                    model_cBN_decoder += [nn.ConvTranspose3d(channels_list[deconv_iter],
                        channels_list[deconv_iter+1], kernel_size_list[deconv_iter],
                        stride=1, padding=0, output_padding=0, groups=1, bias=True,
                        dilation=1),
                        nn.BatchNorm3d(channels_list[deconv_iter+1], eps=1e-05, 
                            momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(True)]

                
            self.model_cBN_decoder = nn.Sequential(*model_cBN_decoder)

            model_decoder = [nn.ConvTranspose3d(channels_list[-1],
                    1, 1,
                    stride=1, padding=0, output_padding=0, groups=1, bias=True,
                    dilation=1)]
            self.model_decoder = nn.Sequential(*model_decoder)



            pre_bottleneck = [nn.Linear(512, 128), nn.ReLU(True)] # to make encoder same dim as above
            self.pre_bottleneck = nn.Sequential(*pre_bottleneck)
            self.film_type = film_type
            self.dec_3d_channels = dec_3d_channels


            return 
            
        
    def forward(self, input2d, cat_nrs):


        if self.film_type == 'enc_2d' or self.film_type == 'all' or self.film_type=='enc_debug'\
                or self.film_type == 'None' or self.film_type=='dec_3d':
            x = self.model_enc((input2d, cat_nrs))


            x = x[0] # drop class nr, should be 32,512,1,1
        x = x.view(-1, x.shape[1])        
        x = self.pre_bottleneck(x)
        x = self.model_bottleneck(x)
        # resize x to -1, 6, 6, 6, 8
        if self.dec_3d_channels == 1:
            tmp_init_depth = 8 # nr of channels 
        elif self.dec_3d_channels == 2:
            tmp_init_depth = 32 # nr of channels
        elif self.dec_3d_channels == 3:
            tmp_init_depth = 64 # nr of channels
        elif self.dec_3d_channels == 4:
            tmp_init_depth = 128 # nr of channels

        tmp_n_final = 6 # we will first resize to 6x6x6 with 8 channels
        x = x.view((-1,)+(tmp_init_depth,) + (tmp_n_final,)*3 )
        # DECODER
        if self.film_type == 'dec_3d' or self.film_type == 'all':
            x = self.model_cBN_decoder((x, cat_nrs))
            x = x[0] # drop the class_nr
        else:
            x = self.model_cBN_decoder(x)

        x = self.model_decoder(x)
        x = x[:,0,:,:,:] # remove the '1' in bsxcxWxDxH, c=1
        return x
    pass




class ResUpBlock(nn.Module): #

    def __init__(self, dim, num_up):
        super(ResUpBlock, self).__init__()
        
        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True),\
            nn.ReLU(False),
            nn.ConvTranspose2d(dim, -num_up+dim, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(-num_up+dim, True),
            nn.ReLU(True),
            nn.Conv2d(-num_up+dim, -num_up+dim, kernel_size=3, padding=1)])

        self.project = nn.Conv2d(dim,dim-num_up,kernel_size=1)
        pass

    def forward(self, x):        
        xu = F.interpolate(x, scale_factor=2, mode='nearest')
        bs,_,h,w = xu.size()
        return self.project(xu) + self.model(x)
    pass




