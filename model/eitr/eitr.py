from .u_trans import mls_tpa


class EITR(mls_tpa):
    def __init__(self, eitr_kwargs):
        super().__init__(eitr_kwargs['num_bins'], eitr_kwargs['norm'])

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
                 N x 1 x H x W
        """
        out = self.func(event_tensor)
        return {'image': out}
