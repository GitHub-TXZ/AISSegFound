import torch.nn as nn 
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
from typing import List,Optional,Union,Tuple,Sequence

class ConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Sequence] = 3,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 1, groups=1, drop=0.):
        super(ConvNormNonlin, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm3d(num_features=out_channels)
        self.nonlin = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = self.nonlin(self.norm(x))
        return x


class ProjectConvBlock(nn.Module):
    def __init__(self, dim, drop=0.):
        super().__init__()
        self.d_conv = ConvNormNonlin(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim,
                                      drop=drop)
        self.h_conv = ConvNormNonlin(dim, dim, kernel_size=(3, 1, 3), padding=(1, 0, 1), groups=dim,
                                      drop=drop)
        self.w_conv = ConvNormNonlin(dim, dim, kernel_size=(3, 3, 1), padding=(1, 1, 0), groups=dim,
                                      drop=drop)
        self.pwconv = ConvNormNonlin(dim * 3, dim, kernel_size=1, stride=1, padding=0, drop=drop)

    def forward(self, x):
        x_d = self.d_conv(x)
        x_h = self.h_conv(x)
        x_w = self.w_conv(x)
        x = torch.cat([x_d, x_h, x_w], dim=1)
        x = self.pwconv(x)
        return x




class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, raw_size:Optional[Union[None,Tuple,List]],multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(2).unsqueeze(3).unsqueeze(4))
        zeros = torch.zeros(raw_size, requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class Conv_lora(nn.Module):
    def __init__(self,interplote_ratio=4,in_channels=4) -> None:
        super().__init__()
        self.interplote_ratio = interplote_ratio
        self.conv = nn.Conv3d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1) # ConvLoRA 原始的 Regular Conv
        # self.conv = ProjectConvBlock(in_channels) # 师兄的MPConv
    
    def forward(self,x):
        # x = x.permute(0,3,1,2).contiguous()
        B,C,D,H,W = x.shape

        size0 = []
        for i in range(3, 0, -1):
            size0.append(self.interplote_ratio * x.shape[-i])
        
        x = F.interpolate(x,size=tuple(size0),mode="trilinear")
        x = self.conv(x)
        x = F.interpolate(x, size=(D, H, W), mode="trilinear")

        # x = x.permute(0,2,3,1)

        return x 

class MoEConv(nn.Module):
    def __init__(self, patch_size, img_size, d, scales:List[int], M=4, K=1, noisy_gating=True):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
        """
        super(MoEConv, self).__init__()

        assert len(scales) == M 
        self.patch_size = patch_size  # D,H,W
        self.img_size = img_size
        self.M = M
        self.k = K
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # global average pooling

        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        # 构建Experts
        self.experts = nn.ModuleList([Conv_lora(interplote_ratio=scales[i],in_channels=d) for i in range(self.M)])

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        D_, H_, W_ = [img_size // patch_size for img_size, patch_size in zip(self.img_size, self.patch_size)] #超像素大小
        feats = feats.view(-1, D_, H_, W_, feats.shape[-1]).contiguous()
        feats = feats.permute(0,4,1,2,3).contiguous()
        batch_size = feats.shape[0]
        
        raw_shape = feats.shape
        feats_S = self.gap(feats).view(batch_size, -1)

        clean_logits = feats_S @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.M, gates)
        expert_inputs = dispatcher.dispatch(feats)
        gates = dispatcher.expert_to_gates()

        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.M)]

        y = dispatcher.combine(expert_outputs,raw_size=raw_shape)
    
        y = y.permute(0,2,3,4,1).contiguous()
        
        y = y.view(batch_size, D_ * H_ * W_, y.shape[-1]).contiguous()
        return y
        # return y, loss

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


if __name__ == "__main__":
    # t = torch.randn(8, 4, 16, 16, 4) #  nnUNet: B, D, H, W, C
    t = torch.randn(3, 8*16*16, 4) 
    model = MoEConv(patch_size=[4,16,16],img_size=[32,256,256], d=4, scales=[1,2,4,8],K=2,M=4)
    # gate, loss = model(t)
    gate = model(t)
    print(gate.shape)
    # 计算下model的参数量（M）和计算量（FLOPs），用thop
    from thop import profile
    flops, params = profile(model, inputs=(t,))   
    print(f"FLOPs: {flops / 1e9:.2f}G")
    print(f"Number of parameters: {params}")
    # print(f"Number of parameters: {params / 1e6:.2f}M")
    
  
  
    # print(loss)
    
    