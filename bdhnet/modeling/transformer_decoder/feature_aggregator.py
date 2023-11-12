import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Optional
from .gcn import GCN

from visualizer import get_local


def calculate_instance_centers(binary_masks):
    """
    计算二值掩码中每个实例的中心坐标。

    Args:
        binary_masks (torch.Tensor): 二值掩码张量，形状为 [batch_size, num_instance, height, width]。

    Returns:
        tuple: 包含两个张量 (instance_centers_x, instance_centers_y)，形状为 [batch_size, num_instance]。
    """
    # 获取二值掩码的形状信息
    bsz, num_instance, height, width = binary_masks.shape

    # 创建坐标网格
    x_grid, y_grid = torch.meshgrid(torch.arange(width), torch.arange(height))
    x_grid, y_grid = x_grid.float().to(binary_masks.device), y_grid.float().to(binary_masks.device)

    # 将坐标网格复制为每个实例
    x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(bsz, num_instance, -1, -1)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(bsz, num_instance, -1, -1)

    # 计算每个实例的中心坐标
    instance_centers_x = torch.sum(x_grid * binary_masks, dim=[-2, -1]) / (torch.sum(binary_masks, dim=[-2, -1]) + 1e-5)
    instance_centers_y = torch.sum(y_grid * binary_masks, dim=[-2, -1]) / (torch.sum(binary_masks, dim=[-2, -1]) + 1e-5)

    # 组合成 [batch_size, nq, 2] 的张量
    mask_centers = torch.stack([instance_centers_x, instance_centers_y], dim=-1) / height

    return mask_centers


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskFeatureEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=6, dropout=1):
        super().__init__()

        self.downsample = MLP(in_dim, hidden_dim * 2, hidden_dim, 2)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, masks):
        b, n, h, w = masks.shape

        # 先自适应平均池化下采样
        # x = F.adaptive_avg_pool2d(masks, (h // 4, w // 4))
        x = F.adaptive_avg_pool2d(masks, output_size=(64, 64))

        # 将B,N维拼接成单个批量
        x = x.view(b, n, x.size(-2) * x.size(-1))
        x = self.downsample(x)

        # 添加Channel维,进行自注意力
        # x = x.permute(1, 0, 2)
        x = self.norm(x)
        #
        # x = x.unsqueeze(0)
        x, _ = self.attn(x, x, x)
        x += self.dropout(x)

        # 恢复B,N维
        x = x.squeeze(1).view(b, n, -1)

        # FFN
        x = self.ffn(x)

        return x


class SelfAttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0.1):
        super(SelfAttentionAggregator, self).__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(input_dim, output_dim)
        # TODO
        # self.multihead_attention = nn.MultiheadAttention(output_dim, num_heads)
        self.norm = nn.LayerNorm(output_dim)
        self.multihead_attention = nn.MultiheadAttention(200, num_heads)

        self.dropout = nn.Dropout(dropout)

    # @get_local("attn_weight")
    def forward(self, multi_feats):
        # input_feats的形状: [B, N, F]，其中B是批量大小，N是实例数量，F是特征维度
        # input_feats = input_feats.permute(1, 0, 2)
        multi_feats = self.linear(multi_feats)

        multi_feats = self.norm(multi_feats)

        # TODO by wenxu shi:
        multi_feats = multi_feats.permute(2, 0, 1)

        # multi_feats = multi_feats.unsqueeze(0)
        # 使用自注意力机制聚合特征
        attn_output, attn_weight = self.multihead_attention(multi_feats, multi_feats, multi_feats)
        attn_output += self.dropout(attn_output)
        # attn_output = attn_output.squeeze(0)

        # return attn_output
        return attn_output.permute(1, 2, 0)


class FeatureAggregator(nn.Module):
    def __init__(self, hidden_dim=296, out_dim=256):
        super().__init__()
        self.aggregator = SelfAttentionAggregator(hidden_dim, out_dim)
        self.spatial_rel_attn = GCN(out_dim, out_dim * 2)
        self.ffn = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LeakyReLU(), nn.Linear(out_dim, 256))
        self.ffn2 = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softmax(dim=-1))

    def forward(self, pred_height, pred_logits, pred_masks, mask_embed, pred_offsets, scale):
        mask_centers = calculate_instance_centers(pred_masks)
        multi_feats = torch.concat([pred_height, pred_logits, pred_offsets, mask_centers, mask_embed, scale], dim=-1)
        height_feat_intra = self.aggregator(multi_feats).squeeze(0)  # [n, nq, 256]
        height_feat = self.spatial_rel_attn(height_feat_intra)
        height_feat_res = self.ffn(height_feat.unsqueeze(0))

        return height_feat_res


if __name__ == "__main__":
    pred_logits = torch.randn(200, 2).cuda()
    pred_masks = torch.randn(200, 168, 168).cuda()
    pred_offsets = torch.randn(200, 2).cuda()

    model = FeatureAggregator(294, 256).cuda()
    y = model(pred_logits, pred_masks, pred_offsets)
    print(y.shape)
    print("Done!")
