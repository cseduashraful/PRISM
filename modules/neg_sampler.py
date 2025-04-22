import torch

class NegLinkSamplerDest:

    def __init__(self, dst_nodes):
        self.dst_nodes = dst_nodes.tolist()  # Convert to list if necessary

    def sample(self, pos_dst):
        # Sample negative destination nodes using torch
        negdst = torch.tensor(
            [self.dst_nodes[i] for i in torch.randint(0, len(self.dst_nodes), (len(pos_dst),))],
            dtype=pos_dst.dtype
        )

        # Find indices where sampled nodes match positive destination nodes
        same_value_indices = (negdst == pos_dst).nonzero(as_tuple=True)

        if len(same_value_indices[0]) > 0:
            # Resample for indices where there are matches
            newdst = self.sample(pos_dst[same_value_indices])
            negdst[same_value_indices] = newdst

        return negdst
