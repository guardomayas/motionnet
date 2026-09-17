from motionnet.models import makeRaisedCosBasis
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math




class MotionCNN(nn.Module):
    def __init__(self,
                 fps =              None, 
                 rf_size_1=         None,
                 rf_size_2=         None,
                 num_subunits_1=    None,
                 num_subunits_2=    None,
                 out_gain=          None,
                 initial_threshold= None,
                 initial_gain=      None,
                 input_noise_std =  None,
                 layer_noise_std =  None
                 ):

        super().__init__()

        self.rf_size_1 = rf_size_1
        self.rf_size_2 = rf_size_2
        
        self.num_subunits_1 = num_subunits_1
        self.num_subunits_2 = num_subunits_2
        
        # self.mean_function = mean_function
        self.nonlinearity  = 'relu'

        # Noise / regularization
        self.input_noise_std = input_noise_std
        self.layer_noise_std = layer_noise_std     # additive, FIXED scale

        ## First Layer
        self.spatial_kernel = nn.Parameter(
                            torch.randn(
                                num_subunits_1,
                                1,
                                rf_size_1,
                                rf_size_1,
                            )
                        )
        dt = 1 / fps                      # not a hardcoded 1/60
        nbasis = 8 #;     %number of stimulus basis functions

        RFstart = 15e-3 #;%0?
        RFend   = 130e-3#;%150e-3 or 180e-3;
        b = 0.02

        t, B_orth, B = makeRaisedCosBasis(nbasis, dt, [RFstart, RFend], b, zflag=0)
        D = B.shape[0]
        
        print(f"support: {D} frames = {t[-1]*1e3:.0f} ms")
        print("rank:", B_orth.shape[1], "of", nbasis)
        print("condition number:", np.linalg.cond(B))# 16

        self.register_buffer("tbasis", torch.as_tensor(B, dtype=torch.float32))
        self.max_delay = self.tbasis.shape[0]        # support comes from the basis
        self.register_buffer("tbasis", torch.tensor(B, dtype=torch.float32))   # (D, nb), lag 0 first
        self.temporal_w = nn.Parameter(0.1 * torch.randn(num_subunits_1, B.shape[1]))
        
        # self.temporal_kernel = nn.Parameter(
        #                     torch.randn(
        #                         num_subunits_1,
        #                         1,
        #                         max_delay,
        #                     )
        #                 )
        self.threshold = nn.Parameter(
            torch.full(
                (num_subunits_1,),
                float(initial_threshold),
            )
        )
        self.threshold2 = nn.Parameter(
                    torch.full(
                        (num_subunits_2,),
                        float(initial_threshold),
                    )
                )
        self.readout    = nn.Parameter(0.1 * torch.randn(2, num_subunits_2))   # rows: v_x, v_y
        self.out_gain   = out_gain                                             # fixed float, NOT a Parameter
        #
        initial_gain = torch.tensor(float(initial_gain))

        raw_initial_gain = torch.log(
            torch.expm1(initial_gain)
        )

        self.raw_gain = nn.Parameter(
            raw_initial_gain.repeat(num_subunits_1)
        )

        # =======================================================
        # Second nonlinear spatial layer
        # =======================================================
        
        self.layer2 = nn.Conv2d(
            in_channels=num_subunits_1,
            out_channels=num_subunits_2,
            kernel_size=rf_size_2,
            padding=0, #check best convolution operation
            bias=False, #Should we?
        )
        self.raw_gain2 = nn.Parameter(torch.full((num_subunits_2,), math.log(math.e - 1)))  # softplus -> 1


        with torch.no_grad():
            self.layer2.weight -= self.layer2.weight.mean(dim=(2, 3), keepdim=True)
            # ===========================================================
    # First-stage parameters
    # ===========================================================    
    
    @property
    def gain(self):
        """
        Positive response gain for each first-layer channel.

        Shape:
            (num_subunits_1,)
        """
        return F.softplus(self.raw_gain)

    
    def add_input_noise(self, movie):
        if self.input_noise_std > 0:
            movie = movie + self.input_noise_std * torch.randn_like(movie)

            # Optional if your movie is normalized to [0, 1]
            # movie = movie.clamp(0.0, 1.0)

        return movie


    def forward(self, movie):
    
        movie = self.add_input_noise(movie)

        B, T, C, H, W = movie.shape

        # -------------------------------------------------------
        # 1. Spatial conv per frame
        # -------------------------------------------------------
        x = movie.reshape(B * T, C, H, W)
        spatial_weight = self.spatial_kernel
        spatial_norm = torch.linalg.vector_norm(
                                spatial_weight,
                                ord=2,
                                dim=(1, 2, 3),
                                keepdim=True,
                            )
        spatial_weight = spatial_weight / (spatial_norm + 1e-8)
        x = F.conv2d(
                x,
                spatial_weight,
                bias=None,
                padding=0
            )

        _, N1, H1, W1 = x.shape

        x = x.reshape(B, T, N1, H1, W1)

        # -------------------------------------------------------
        # 2. Temporal conv at each spatial location
        # -------------------------------------------------------
        x = x.permute(0, 3, 4, 2, 1)              # (B, H1, W1, N1, T)
        x = x.reshape(B * H1 * W1, N1, T)         # (B*H1*W1, N1, T)

        # causal temporal padding
        x = F.pad(x, (self.max_delay - 1, 0))
        phi = self.temporal_w @ self.tbasis.T                                  # (N1, D)
        phi = phi / (phi.norm(dim=1, keepdim=True) + 1e-8)
        temporal_weight = phi.flip(-1)[:, None, :]
        
        x = F.conv1d(
            x,
            temporal_weight,
            bias=None,
            padding=0,
            groups=self.num_subunits_1,
        )

        x = x.reshape(B, H1, W1, N1, T)
        x = x.permute(0, 4, 3, 1, 2)              # (B, T, N1, H1, W1)

        threshold = self.threshold[None, None, :, None, None]
        
        gain = self.gain[
            None,
            None,
            :,
            None,
            None,
        ]
        if self.nonlinearity == "relu":
            x = gain * F.relu(x - threshold)
        if self.nonlinearity == 'softplus': # can inclued recitified powers later!
            x = gain * F.softplus(x- threshold)
        
        self._subunit_rate = x.mean()            # stash for the penalty
        self._l1_std = x.detach().std(dim=(0, 1, 3, 4))    # per channel, before noise
        if self.layer_noise_std > 0: 
            x = x + self.layer_noise_std * torch.randn_like(x) 
    

        # -------------------------------------------------------
        # 3. Second spatial conv
        # -------------------------------------------------------
        x = x.reshape(B * T, N1, H1, W1)
        # DS should emerge here
        K = self.layer2.weight
        K = K / (torch.linalg.vector_norm(K, dim=(1, 2, 3), keepdim=True) + 1e-8)
        x = F.softplus(self.raw_gain2)[None, :, None, None] * F.relu(
                F.conv2d(x, K) - self.threshold2[None, :, None, None])
        
        self.layer2_rate = x.mean()
        
        # if self.layer_noise_std > 0:#This is Avg out in current set up. For it to matter we need local readout
        #             x = x + self.layer_noise_std * torch.randn_like(x) 
        
        ## Add divisive normalization here or on layer 1 or both
        #TODO:
        _, N2, H2, W2 = x.shape
        
        # -------------------------------------------------------
        # 4. Pooling over space + linear readout
        # -------------------------------------------------------
        x_pool = x.mean(dim=(-1, -2))              # (B*T, N2)        
        self._pool = x_pool.detach()
        W = self.readout / self.readout.norm(dim=1, keepdim=True)             # unit rows
        U = self.out_gain * x_pool @ W.T                                       # (B*T, 2)
        return U.reshape(B, T, 2)
    