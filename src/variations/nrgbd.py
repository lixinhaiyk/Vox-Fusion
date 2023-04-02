import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.embedding_size = mapping_size

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)
    
    
BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')
class HashEmbedder(nn.Module):
    def __init__(self, n_levels=2, n_features_per_level=2,\
                log2_hashmap_size=1, base_resolution=1, finest_resolution=3):
        super(HashEmbedder, self).__init__()
        self.total_time = 0
        self.cnt = 0
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        self.embedding_size = (self.n_levels * self.n_features_per_level)*3

        #self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))
        self.b = torch.tensor(self.finest_resolution).float()

        self.embeddings = nn.ModuleList([nn.Embedding(2,2) for i in range(3)]+[nn.Embedding(4,2) for i in range(3)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    
    def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
        xyz = xyz.squeeze()
        '''
        xyz: 3D coordinates of samples. B x 3
        bounding_box: min and max x,y,z coordinates of object bbox
        resolution: number of voxels per axis
        '''
        box_min, box_max = bounding_box

        if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
            # print("ALERT: some points are outside bounding box. Clipping them!")
            #pdb.set_trace()
            xyz = torch.clamp(xyz, min=box_min, max=box_max)

        grid_size = (box_max-box_min)/resolution
        
        bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
        voxel_min_vertex = bottom_left_idx*grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0],device='cuda:0')*grid_size

        # hashed_voxel_indices = [] # B x 8 ... 000,001,010,011,100,101,110,111
        # for i in [0, 1]:
        #     for j in [0, 1]:
        #         for k in [0, 1]:
        #             vertex_idx = bottom_left_idx + torch.tensor([i,j,k])
        #             # vertex = bottom_left + torch.tensor([i,j,k])*grid_size
        #             hashed_voxel_indices.append(hash(vertex_idx, log2_hashmap_size))

        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
        #hashed_voxel_indices = voxel_indices.mean(-1)
        #hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)
        #hashed_voxel_indices = hash2(voxel_indices, log2_hashmap_size)
        hashed_voxel_indices = voxel_indices.clamp(0,resolution).int()
        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        x = x.squeeze()
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        begin_time = time.time()
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = self.get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            for j in range(hashed_voxel_indices.shape[-1]):
                t_voxl_indices = hashed_voxel_indices[...,j]
                voxel_embedds = self.embeddings[i*3+j](t_voxl_indices)
                x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
                x_embedded_all.append(x_embedded)

        res = torch.cat(x_embedded_all, dim=-1)
        # end_time = time.time()
        # self.total_time += (end_time-begin_time)
        # self.cnt = self.cnt +1
        # print("fourier {}",self.total_time/self.cnt)
        
        return res 

class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, in_dim, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs
        self.embedding_size = multires*in_dim*2 + in_dim

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class Same(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.embedding_size = in_dim

    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self,
                 depth=8,
                 width=256,
                 in_dim=3,
                 sdf_dim=128,
                 skips=[4],
                 multires=6,
                 embedder='sip',
                 local_coord=False,
                 **kwargs):
        """
        """
        super().__init__()
        self.D = depth
        self.W = width
        self.skips = skips
        if embedder == 'nerf':
            self.pe = Nerf_positional_embedding(in_dim, multires)
        elif embedder == 'none':
            self.pe = Same(in_dim)
        elif embedder == 'gaussian':
            self.pe = GaussianFourierFeatureTransform(in_dim)
        elif embedder == 'sip':
            self.pe = HashEmbedder()
        else:
            raise NotImplementedError("unknown positional encoder")

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.pe.embedding_size, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + self.pe.embedding_size, width) for i in range(depth-1)])
        self.sdf_out = nn.Linear(width, 1+sdf_dim)
        self.color_out = nn.Sequential(
            nn.Linear(sdf_dim+self.pe.embedding_size, width),
            nn.ReLU(),
            nn.Linear(width, 3),
            nn.Sigmoid())
        # self.output_linear = nn.Linear(width, 4)

    def get_values(self, x):
        x = self.pe(x)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # outputs = self.output_linear(h)
        # outputs[:, :3] = torch.sigmoid(outputs[:, :3])
        sdf_out = self.sdf_out(h)
        sdf = sdf_out[:, :1]
        sdf_feat = sdf_out[:, 1:]

        h = torch.cat([sdf_feat, x], dim=-1)
        rgb = self.color_out(h)
        outputs = torch.cat([rgb, sdf], dim=-1)

        return outputs

    def get_sdf(self, inputs):
        return self.get_values(inputs['emb'])[:, 3]

    def forward(self, inputs):
        outputs = self.get_values(inputs['emb'])

        return {
            'color': outputs[:, :3],
            'sdf': outputs[:, 3]
        }
