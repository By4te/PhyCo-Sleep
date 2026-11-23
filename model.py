import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_directional_conflict(e_v, e_w, eps=1e-8):
    alpha_v = e_v + 1.0
    alpha_w = e_w + 1.0
    S_v = alpha_v.sum(dim=1, keepdim=True)
    S_w = alpha_w.sum(dim=1, keepdim=True)

    b_v = e_v / (S_v + eps)
    b_w = e_w / (S_w + eps)
    u_v = e_v.new_full((e_v.size(0), 1), e_v.size(1)) / (S_v + eps)
    u_w = e_w.new_full((e_w.size(0), 1), e_w.size(1)) / (S_w + eps)

    b_hat_v = b_v / (1.0 - u_v + eps)
    b_hat_w = b_w / (1.0 - u_w + eps)

    C_vw = 0.5 * torch.sum(torch.abs(b_hat_v - b_hat_w), dim=1, keepdim=True)
    return C_vw.clamp(0.0, 1.0)


class PhyCo(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(PhyCo, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)]
        )

    def forward(self, X):
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])

        C = {}
        for v in range(self.num_views):
            for w in range(self.num_views):
                if v == w:
                    continue
                C[(v, w)] = compute_directional_conflict(evidences[v], evidences[w])

        weights = {}
        for v in range(self.num_views):
            if self.num_views == 1:
                weights[v] = torch.ones_like(evidences[v][:, :1])
                continue
            C_avg_v = sum(C[(v, w)] for w in range(self.num_views) if w != v) / (self.num_views - 1)
            weights[v] = 1.0 - C_avg_v

        e_plus = 0.0
        for v in range(self.num_views):
            e_plus = e_plus + weights[v] * evidences[v]

        alpha_plus = e_plus + 1.0
        S_plus = alpha_plus.sum(dim=1, keepdim=True)
        b_plus = (alpha_plus - 1.0) / (S_plus + 1e-8)
        u_plus = self.num_classes / (S_plus + 1e-8)

        evidence_a = e_plus
        return evidences, evidence_a


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h
