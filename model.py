from einops import einsum
import torch
import torch.nn as nn

class PolyProbe(nn.Module):
    def __init__(self, in_features, out_features, max_order, ranks, d_prob=0.0, linear_init=None, train_linear=False, train_mode='', term_drop=True):
        super().__init__()
        print(f'PolyProbe: in_features={in_features}, out_features={out_features}, max_order={max_order}')
        self.layer_type = 'Poly_CP'
        self.train_mode = train_mode
        self.out_features = out_features
        self.d_prob = d_prob
        self.ranks = ranks
        self.max_order = max_order
        self.in_features = in_features
        self.term_drop = term_drop
        
        self.lam = nn.ParameterList([nn.Parameter(torch.randn(ranks[o])*0.02, requires_grad=True) for o in range(max_order-1)])

        # 0th and 1st order terms
        self.W = [
            nn.Parameter(torch.zeros([out_features]), requires_grad=bool(train_linear)), # constant bias term
            nn.Parameter(torch.nn.Linear(in_features, out_features).weight, requires_grad=bool(train_linear)), # linear term
        ]

        self.HO = [] # store the higher-order terms
        for order in range(max_order-1):
            self.HO.append(nn.Parameter(torch.nn.Linear(in_features, ranks[order]).weight, requires_grad=True))

        self.W = nn.ParameterList(self.W)
        self.HO = nn.ParameterList(self.HO)
        
        if linear_init is not None:
            self.W[0].data = torch.Tensor(linear_init[0], device=self.W[0].device)
            self.W[1].data = torch.Tensor(linear_init[1], device=self.W[1].device)
            print(f'Using {"(trainable)" if train_linear else "(frozen)"} linear probe initialization for PolyLayer!')

    def forward(self, x, test_time_order=None):
        # linear term
        y = einsum(self.W[1], x, 'o i, ... i -> ... o') + self.W[0]

        if self.term_drop and self.training:
            dout_mask = (torch.rand((x.shape[0]), device=x.device) > self.d_prob).float() if self.training else torch.ones((x.shape[0]), device=x.device)
            y = y * dout_mask.view(-1, 1)
        
        ys = [y]

        # loop over higher-orders
        for n in range(min(test_time_order, self.max_order)-1):
            order = n+2
            inner = einsum(x, self.HO[n], '... i, r i -> ... r') ** (order) # contract input with factor matrix, raise to power 'order'
            yn = einsum(inner, self.lam[n], '... r, r -> ...').unsqueeze(-1) # sum over the rank dimension

            # optional dropout for previous terms
            if self.term_drop and self.training and order < test_time_order:
                dout_mask = (torch.rand((x.shape[0]), device=x.device) > self.d_prob).float() if self.training else torch.ones((x.shape[0]), device=x.device)
                yn = yn * dout_mask.view(-1, 1)

            y = y + yn
            ys.append(y)

        return ys

################################################################
# enter: baselines etc below
################################################################

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)
        self.layer_type = 'Linear'

    # kwargs absorbs the order flags (e.g. test-time order)
    def forward(self, x: torch.Tensor, **kwargs):
        y = self.net(x)
        return [y]  # we're wrapping the [output] in a list with one element just to match the interface of the other probes


class BilinearProbe(nn.Module):
    """
    Bilinear probe with symmetric decomposition, and factorized forward pass

    Note here: this corrresponds to the 2nd order term from the polynomial expansion alone
    """
    def __init__(self, in_features, out_features, rank, **kwargs):
        super().__init__()
        #print(f'BilinearProbe: in_features={in_features}, out_features={out_features}, rank={rank}')
        self.layer_type = 'Bilinear'

        self.symmetric = True

        if not self.symmetric:
            # CP factors
            Wi1 = nn.Parameter(torch.nn.Linear(in_features, rank).weight)
            Wi2 = nn.Parameter(torch.nn.Linear(in_features, rank).weight)
            Wo = nn.Parameter(torch.nn.Linear(rank, out_features).weight)
            self.W = nn.ParameterList([Wi1, Wi2, Wo])
        else:
            W = nn.Parameter(torch.nn.Linear(in_features, rank).weight)
            self.lam = nn.Parameter(torch.randn(rank)*0.02, requires_grad=True)
            self.W = nn.ParameterList([W])

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=True)
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
    # kwargs absorbs the order flags (e.g. test-time order)
    def forward(self, x, testing=False, **kwargs):
        if self.symmetric:
            y1 = einsum(x, self.W[0], '... i, r i -> ... r')**2
            y = einsum(y1, self.lam, '... r, r -> ...').unsqueeze(-1) + self.bias # contract the rank dimension with the output mode
        else:
            y1 = einsum(x, self.W[0], '... i, r i -> ... r')
            y2 = einsum(x, self.W[1], '... i, r i -> ... r')
            y = einsum(y1, y2, self.W[2], '... r, ... r, o r -> ... o') + self.bias

        return [y]  # we're wrapping the [output] in a list with one element just to match the interface of the other probes


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_prob=0.0, **kwargs):
        super().__init__()
        self.layer_type = 'MLP'
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=d_prob) if d_prob > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y = self.net(x)
        return [y]  # we're wrapping the [output] in a list with one element just to match the interface of the other probes


class EEMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, num_layers, output_dim=1, increasing_depth=False, d_prob=0.0, **kwargs):
        super().__init__()
        self.layer_type = 'EEMLP'

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=d_prob) if d_prob > 0 else nn.Identity()
            )
            for (in_dim, hidden_dim) in zip([input_dim]+hidden_dims[:-1], hidden_dims)
        ])
        self.side_branches = nn.ModuleList([
            nn.Linear(input_dim, output_dim),  # branch 0
            *[nn.Linear(h, output_dim) for h in hidden_dims]
        ])

    def forward(self, x: torch.Tensor, test_time_order: int = None) -> torch.Tensor:
        if test_time_order is None:
            # by default, use all layers
            test_time_order = len(self.W)

        ys = [self.side_branches[0](x)]
        
        for i in range(test_time_order-1):
            x = self.layers[i](x)
            ys.append(self.side_branches[i+1](x))
            
        return ys
