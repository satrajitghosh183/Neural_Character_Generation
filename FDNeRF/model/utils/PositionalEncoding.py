import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# for performance analysis
import torch.autograd.profiler as profiler 

class PositionalEncoding(nn.Module):
    """
    NeRF positional encoding
    """
    
    def __init__(self, num_freqs = 6, d_in = 3, freq_factor = np.pi, include_input = True, log_sampling = True):
        """_summary_

        Args:
            num_freqs (int, optional): _description_. Defaults to 6.
            d_in (int, optional): _description_. Defaults to 3.
            freq_factor (_type_, optional): _description_. Defaults to np.pi.
            include_input (bool, optional): _description_. Defaults to True.
            log_sampling (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        
        self.num_freqs = num_freqs
        self.d_in = d_in
        
        # add log sampling option
        if log_sampling:
            self.freqs = freq_factor * 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freqs = freq_factor * torch.linspace(1, 2**num_freqs, num_freqs)
        
        self.include_input = include_input
        
        self.d_out = num_freqs * d_in
        
        if include_input:
            self.d_out += d_in
        
        # non-learnable parameters saved together with model
        # dimensions: [1, num_freqs*2, 1], used for broadcasting
        self.register_buffer('_freqs', torch.repeat_interleave(self.freqs, 2).view(1, -1, 1))
        
        _phases = torch.zeros(2*self.num_freqs)
        _phases[1::2] = 0.5 * np.pi # odd, pi/2
        self.register_buffer('_phases', _phases.view(1, -1, 1))
        
    def forward(self, x):
        with profiler.record_function('positional_enc'):
            
            # (batch_size, d_in) -> (batch_size, 1, d_in) -> (batch_size, num_freqs*2, d_in)
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs)) # check addcmul manual for broadcasting mechanism
            embed = embed.view(x.shape[0], -1)
            
            if self.include_input:
                embed = torch.cat([x, embed], dim = -1)
            return embed
    
    # The use of config to be determined, may re-implement with parser
    @classmethod
    def from_conf(cls, conf, d_in = 3):
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
            conf.get_bool("log_sampling", True)
        )
        
    
        
        
if __name__ == "__main__":
    # test
    pos_enc = PositionalEncoding(d_in=1, num_freqs=3, include_input = False, log_sampling = True)
    
    # x = torch.randn(2, 3)
    x = torch.tensor([[1,1,1]])
    
    y = pos_enc(x)
    
    print(y.shape)
    print(y)
    
    # test with input included
    pos_enc = PositionalEncoding(d_in=1, num_freqs=3, include_input = True, log_sampling = True)
    
    x = torch.tensor([[1,1,1]])
    
    y = pos_enc(x)
    
    print(y.shape)
    print(y)