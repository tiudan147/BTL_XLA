import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import segutils
import core.denseaffinity as dautils

identity_mapping = lambda x, *args, **kwargs: x


class ContrastiveConfig:
    def __init__(self, config=None):
        # Define the internal dictionary with default settings.
        if config is None:
            self._data = {
                'aug': {
                    'n_transformed_imgs': 2,
                    'blurkernelsize': [1],  # chooses one of this kernel sizes
                    'maxjitter': 0.0,
                    'maxangle': 0,  # rotation
                    # 'translate': (0,0),  # BE CAREFUL WITH TRANSLATE - if you apply it on the feature volume that has smaller spatial dims correspondences break
                    'maxscale': 1.0,  # 1.0 = No scaling
                    'maxshear': 20,
                    'randomhflip': False,
                    'apply_affine': True,
                    'debug': False
                },
                'model': {
                    'out_channels': 64,
                    'kernel_size': 1,
                    'prepend_relu': False,
                    'append_normalize': False,
                    'debug': False
                },
                'fitting': {
                    'lr': 1e-2,
                    'optimizer': torch.optim.SGD,
                    'num_epochs': 25,
                    'nce': {
                        'temperature': 0.5,
                        'debug': False
                    },
                    'normalize_after_fwd_pass': True,
                    'q_nceloss': True,
                    's_nceloss': True,
                    'protoloss': False,
                    'keepvarloss': True,
                    'symmetricloss': False,
                    'selfattentionloss': False,
                    'o_t_contr_proto_loss': True,
                    'debug': False
                },
                'featext': {
                    'l0': 3,  # the first resnet bottleneck id to consider (0,1,2,3,4,5...15)
                    'fit_every_episode': False
                }
            }
        else:
            self._data = config

    def __getattr__(self, key):
        # Try to get '_data' without causing a recursive call to __getattr__
        _data = super().__getattribute__('_data') if '_data' in self.__dict__ else None

        if _data is not None and key in _data:
            if isinstance(_data[key], dict):
                return ContrastiveConfig(_data[key])
            return _data[key]

        # If we're here, it means the key was not found in the data,
        # so we let Python raise the appropriate AttributeError.
        raise AttributeError(f"No setting named {key}")

    def __setattr__(self, key, value):
        # Prevent overwriting of the '_data' attribute by normal means
        if key == '_data':
            super().__setattr__(key, value)
        else:
            # Try to get '_data' without causing a recursive call to __getattr__
            _data = super().__getattribute__('_data') if '_data' in self.__dict__ else None

            if _data is not None:
                _data[key] = value
            else:
                # This situation should not normally occur, handle appropriately (e.g., log an error, raise exception)
                raise AttributeError("Unexpected")

    # Optional: Representation for better debugging.
    def __repr__(self):
        return str(self._data)

#------------------------------------------------
#            Nơi định nghĩa hàm loss Lq, Ls
#------------------------------------------------

def dense_info_nce_loss(original_features, transformed_features, config_nce):
    B, C, H, W = transformed_features.shape
    o_features = original_features.expand(B, C, H, W).permute(0, 2, 3, 1).view(B, H * W, C)
    t_features = transformed_features.permute(0, 2, 3, 1).view(B, H * W, C)

    # Calculate dot product between original and transformed feature vectors for positive pairs
    positive_logits = torch.einsum('bik,bik->bi', o_features, t_features) / config_nce.temperature

    # Calculate dot product between original features and all other transformed features for negative pairs
    all_logits = torch.einsum('bik,bjk->bij', o_features, t_features) / config_nce.temperature

    if config_nce.debug: print('pos/neg:', positive_logits.mean().detach(), all_logits.mean().detach())

    # Using the log-sum-exp trick
    max_logits = torch.max(all_logits, dim=-1, keepdim=True).values
    log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(all_logits - max_logits), dim=-1, keepdim=True))

    # Compute InfoNCE loss
    loss = - (positive_logits - log_sum_exp.squeeze())
    return loss.mean()  # [B=k*aug] or [B=k] -> scalar


def ssim(a, b):
    return torch.nn.CosineSimilarity()(a, b)

def augwise_proto(feat_vol, mask, k, aug):
    k, aug, c, h, w = k, aug, *feat_vol.shape[-3:]
    feature_vectors_augwise = torch.cat(feat_vol.view(k, aug, c, h * w).unbind(0), dim=-1)
    mask_augwise = torch.cat(segutils.downsample_mask(mask, h, w).view(k, aug, h * w).unbind(0), dim=-1)
    assert feature_vectors_augwise.shape == (aug, c, k * h * w) and mask_augwise.shape == (
    aug, k * h * w), "of transformed"

    fg_proto, bg_proto = segutils.fg_bg_proto(feature_vectors_augwise, mask_augwise)
    assert fg_proto.shape == bg_proto.shape == (aug, c)

    return fg_proto, bg_proto


def calc_q_pred_coarse_nodetach(qft, sft, s_mask, l0=3):
    bsz, c, hq, wq = qft.shape
    hs, ws = sft.shape[-2:]

    sft_row = torch.cat(sft.unbind(1), -1)  # bsz,k,c,h,w -> bsz,c,h,w*k
    smasks_downsampled = [segutils.downsample_mask(m, hs, ws) for m in s_mask.unbind(1)]
    smask_row = torch.cat(smasks_downsampled, -1)

    damat = dautils.buildDenseAffinityMat(qft, sft_row)
    filtered = dautils.filterDenseAffinityMap(damat, smask_row)
    q_pred_coarse = filtered.view(bsz, hq, wq)
    return q_pred_coarse


# input k*aug,c,h,w
def self_attention_loss(f_base, f_transformed, mask_base, mask_transformed, k, aug):
    c, h, w = f_base.shape[-3:]
    pseudoquery = torch.cat(f_base.view(k, aug, c, h, w).unbind(0), -1)  # shape aug,c,h,w*k
    pseudoquerymask = torch.cat(mask_base.view(k, aug, h, w).unbind(0), -1)  # shape aug,h,w*k
    pseudosupport = f_transformed.view(k, aug, c, h, w).transpose(0, 1)  # shape bsz,k,c,h,w
    pseudosupportmask = mask_transformed.view(k, aug, h, w).transpose(0, 1)  # shape bsz,k,h,w
    # display(segutils.tensor_table(q=pseudoquery, s=pseudosupport, m=pseudosupportmask))
    pred_map = calc_q_pred_coarse_nodetach(pseudoquery, pseudosupport, pseudosupportmask, l0=0)

    loss = torch.nn.BCELoss()(pred_map.float(), pseudoquerymask.float())
    return loss.mean()

#------------------------------------------------
#            Nơi định nghĩa hàm loss Lp
#------------------------------------------------

def ctrstive_prototype_loss(base, transformed, mask_base, mask_transformed, k, aug, tau_p=0.5):
    assert transformed.shape == base.shape, ".."
    b, c, h, w = base.shape
    assert b == k * aug, 'provide correct k and aug such that dim0=k*aug'

    # ==========================================
    # 1. TÍNH TOÁN PROTOTYPE TOÀN CỤC (Anchor tĩnh)
    # ==========================================
    fg_proto_o, bg_proto_o = augwise_proto(base, mask_base, k, aug)
    fg_proto_t, bg_proto_t = augwise_proto(transformed, mask_transformed, k, aug)

    # ==========================================
    # 2. GLOBAL LOSS (Duy trì cấu trúc vĩ mô)
    # ==========================================
    pos_fg = torch.exp(ssim(fg_proto_o, fg_proto_t) / tau_p)
    neg_bg_o = 0.5 * (torch.exp(ssim(fg_proto_o, bg_proto_t) / tau_p) + torch.exp(ssim(fg_proto_o, bg_proto_o) / tau_p))
    neg_bg_t = 0.5 * (torch.exp(ssim(fg_proto_t, bg_proto_o) / tau_p) + torch.exp(ssim(fg_proto_t, bg_proto_t) / tau_p))
    loss_global_fg = 0.5 * (-torch.log(pos_fg / (pos_fg + neg_bg_o)) - torch.log(pos_fg / (pos_fg + neg_bg_t))).mean()

    pos_bg = torch.exp(ssim(bg_proto_o, bg_proto_t) / tau_p)
    neg_bg_cross = 0.5 * (torch.exp(ssim(bg_proto_o, fg_proto_t) / tau_p) + torch.exp(ssim(bg_proto_t, fg_proto_o) / tau_p))
    loss_global_bg = -torch.log(pos_bg / (pos_bg + neg_bg_cross)).mean()

    # ==========================================
    # 3. SEMANTIC-GATED PIXEL-TO-PROTOTYPE (Lưỡi kiếm tinh gọn)
    # ==========================================
    tau_dense = 0.2
    tau_gate = 0.1  # Độ nhạy của Cổng Ngữ nghĩa
    m_pos = 0.15 
    m_neg = 0.05 
    
    base_norm = F.normalize(base, p=2, dim=1)
    trans_norm = F.normalize(transformed, p=2, dim=1)

    # Mỏ neo tĩnh
    fg_p_o = F.normalize(fg_proto_o, p=2, dim=-1).view(b, c, 1, 1).detach()
    bg_p_o = F.normalize(bg_proto_o, p=2, dim=-1).view(b, c, 1, 1).detach()
    fg_p_t = F.normalize(fg_proto_t, p=2, dim=-1).view(b, c, 1, 1).detach()
    bg_p_t = F.normalize(bg_proto_t, p=2, dim=-1).view(b, c, 1, 1).detach()

    soft_mask_o = segutils.downsample_mask(mask_base, h, w).unsqueeze(1)
    soft_mask_t = segutils.downsample_mask(mask_transformed, h, w).unsqueeze(1)
    
    # --- SEMANTIC GATE: Dùng Trí thông minh của mô hình để lọc rác ---
    # Ảnh gốc (O)
    sim_fg_o_raw = (base_norm * fg_p_t).sum(dim=1, keepdim=True)
    sim_bg_o_raw = (base_norm * bg_p_t).sum(dim=1, keepdim=True)
    gate_fg_o = torch.sigmoid((sim_fg_o_raw - sim_bg_o_raw) / tau_gate).detach()
    gate_bg_o = torch.sigmoid((sim_bg_o_raw - sim_fg_o_raw) / tau_gate).detach()
    
    # Ảnh biến đổi (T)
    sim_fg_t_raw = (trans_norm * fg_p_o).sum(dim=1, keepdim=True)
    sim_bg_t_raw = (trans_norm * bg_p_o).sum(dim=1, keepdim=True)
    gate_fg_t = torch.sigmoid((sim_fg_t_raw - sim_bg_t_raw) / tau_gate).detach()
    gate_bg_t = torch.sigmoid((sim_bg_t_raw - sim_fg_t_raw) / tau_gate).detach()

    # Áp dụng cổng thông minh vào Mask
    eff_mask_fg_o = soft_mask_o * gate_fg_o
    eff_mask_fg_t = soft_mask_t * gate_fg_t
    
    bg_mask_o = torch.clamp(1.0 - soft_mask_o, min=0.0, max=1.0)
    bg_mask_t = torch.clamp(1.0 - soft_mask_t, min=0.0, max=1.0)
    eff_mask_bg_o = bg_mask_o * gate_bg_o
    eff_mask_bg_t = bg_mask_t * gate_bg_t

    # --- A. CĂN CHỈNH TỪNG PIXEL TIỀN CẢNH (Với Margin CosFace) ---
    pos_fg_o = (sim_fg_o_raw - m_pos) / tau_dense
    neg_fg_o_1 = (sim_bg_o_raw + m_neg) / tau_dense
    neg_fg_o_2 = ((base_norm * bg_p_o).sum(dim=1, keepdim=True) + m_neg) / tau_dense
    loss_fg_map_o = torch.logsumexp(torch.stack([pos_fg_o, neg_fg_o_1, neg_fg_o_2], dim=-1), dim=-1) - pos_fg_o

    pos_fg_t = (sim_fg_t_raw - m_pos) / tau_dense
    neg_fg_t_1 = (sim_bg_t_raw + m_neg) / tau_dense
    neg_fg_t_2 = ((trans_norm * bg_p_t).sum(dim=1, keepdim=True) + m_neg) / tau_dense
    loss_fg_map_t = torch.logsumexp(torch.stack([pos_fg_t, neg_fg_t_1, neg_fg_t_2], dim=-1), dim=-1) - pos_fg_t

    loss_dense_fg = 0.5 * ((loss_fg_map_o * eff_mask_fg_o).sum() / (eff_mask_fg_o.sum() + 1e-6) +
                           (loss_fg_map_t * eff_mask_fg_t).sum() / (eff_mask_fg_t.sum() + 1e-6))

    # --- B. CĂN CHỈNH TỪNG PIXEL HẬU CẢNH ---
    pos_bg_o = (sim_bg_o_raw - m_pos) / tau_dense
    neg_bg_o_1 = (sim_fg_o_raw + m_neg) / tau_dense
    neg_bg_o_2 = ((base_norm * fg_p_o).sum(dim=1, keepdim=True) + m_neg) / tau_dense
    loss_bg_map_o = torch.logsumexp(torch.stack([pos_bg_o, neg_bg_o_1, neg_bg_o_2], dim=-1), dim=-1) - pos_bg_o

    pos_bg_t = (sim_bg_t_raw - m_pos) / tau_dense
    neg_bg_t_1 = (sim_fg_t_raw + m_neg) / tau_dense
    neg_bg_t_2 = ((trans_norm * fg_p_t).sum(dim=1, keepdim=True) + m_neg) / tau_dense
    loss_bg_map_t = torch.logsumexp(torch.stack([pos_bg_t, neg_bg_t_1, neg_bg_t_2], dim=-1), dim=-1) - pos_bg_t

    loss_dense_bg = 0.5 * ((loss_bg_map_o * eff_mask_bg_o).sum() / (eff_mask_bg_o.sum() + 1e-6) +
                           (loss_bg_map_t * eff_mask_bg_t).sum() / (eff_mask_bg_t.sum() + 1e-6))

    # Tỷ lệ tổng hợp: Gọn gàng và mạnh mẽ nhất
    return loss_global_fg + 0.2 * loss_global_bg + 0.5 * loss_dense_fg + 0.2 * loss_dense_bg

# -------------------hàm Lp ban đầu------------
# features of base, transformed: [b,c,h,w]
# if base features are aligned with transformed features, pass both same

# def ctrstive_prototype_loss(base, transformed, mask_base, mask_transformed, k, aug):
#     assert transformed.shape == base.shape, ".."
#     b, c, h, w = base.shape
#     assert b == k * aug, 'provide correct k and aug such that dim0=k*aug'
#     assert mask_base.shape == mask_transformed.shape == (b, h, w), ".."
#     fg_proto_o, bg_proto_o = augwise_proto(base, mask_base, k, aug)
#     fg_proto_t, bg_proto_t = augwise_proto(transformed, mask_transformed, k, aug)
#     # i: fg, b: bg
#     # p_b_i, p_b_j = segutils.fg_bg_proto(base.view(b,c,h*w), mask_base.view(b,h*w))
#     # p_t_i, p_t_j = segutils.fg_bg_proto(transformed.view(b,c,h*w), mask_transformed.view(b,h*w))
#     enumer = torch.exp(
#         ssim(fg_proto_o, fg_proto_t))  # 5vs5 (augvsaug), but in 5-shot: 25vs25, no, you want also augvsaug
#     denom = torch.exp(ssim(fg_proto_o, fg_proto_t)) + torch.exp(ssim(fg_proto_o, bg_proto_t))
#     assert enumer.shape == denom.shape == torch.Size([aug]), 'you want to calculate one prototype for each augmentation'
#     loss = -torch.log(enumer / denom)  # [bsz]
#     return loss.mean()


def opposite_proto_sim_in_aug(transformed_features, mapped_s_masks, k, aug):
    fg_proto_t, bg_proto_t = augwise_proto(transformed_features, mapped_s_masks, k, aug)
    fg_bg_sim_t = ssim(fg_proto_t, bg_proto_t)
    return fg_bg_sim_t.mean()


def proto_align_val_measure(original_features, transformed_features, mapped_s_masks, k, aug):
    fg_proto_o, _ = augwise_proto(original_features, mapped_s_masks, k, aug)
    fg_proto_t, _ = augwise_proto(transformed_features, mapped_s_masks, k, aug)
    fg_proto_sim = ssim(fg_proto_o, fg_proto_t)
    return fg_proto_sim.mean()


def atest():
    k, aug, c, h, w = 2, 5, 8, 20, 20
    f_base = torch.rand(k * aug, c, h, w).float()
    f_base.requires_grad = True
    f_transformed = torch.rand(k * aug, c, h, w).float()
    mask_base = torch.randint(0, 2, (k * aug, h, w)).float()
    mask_transformed = torch.randint(0, 2, (k * aug, h, w)).float()

    return self_attention_loss(f_base, f_transformed, mask_base, mask_transformed, k, aug)

def keep_var_loss(original_features, transformed_features):
    meandiff = original_features.mean((-2, -1)) - transformed_features.mean((-2, -1))
    vardiff = original_features.var((-2, -1)) - transformed_features.var((-2, -1))
    keepvarloss = torch.abs(meandiff).mean() + torch.abs(
        vardiff).mean()  # [k*aug,c] -> [scalar] or  [aug,c] -> [scalar]
    return keepvarloss

#-----------------------------------------------------
#         Vị trí mạng adapter
#---------------------------------------------------------

#-------------------NAF------------------------------

# class SimpleGate(nn.Module):
#     def forward(self, x):
#         # Cắt đôi Tensor theo chiều kênh
#         x1, x2 = x.chunk(2, dim=1)
#         # Nửa này làm mặt nạ Attention động cho nửa kia (0 tham số học, 0 làm mờ)
#         return x1 * x2

# class NAFBlock(nn.Module):
#     def __init__(self, channels=64):
#         super(NAFBlock, self).__init__()
        
#         # 1. LayerNorm: Vua ổn định, triệt tiêu nhiễu loạn của Few-Shot (Batch Size = 1)
#         self.norm = nn.LayerNorm(channels, eps=1e-6)
        
#         # 2. Conv1x1 mở rộng kênh gấp đôi (để chuẩn bị cho SimpleGate)
#         self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        
#         # 3. Depthwise 3x3: Lấy hình khối cho Lung/ISIC nhưng KHÔNG trộn kênh để bảo vệ Deepglobe
#         self.dw_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        
#         # 4. Cổng ma thuật NAF (Thay thế hoàn toàn Attention, ReLU, Sigmoid)
#         self.sg = SimpleGate()
        
#         # 5. Conv1x1: Gom đặc trưng sắc nét trả về
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

#     def forward(self, x):
#         identity = x
        
#         # Bước 1: Chuẩn hóa LayerNorm (Chuyển kênh về cuối để tính, rồi trả lại)
#         out = x.permute(0, 2, 3, 1)
#         out = self.norm(out)
#         out = out.permute(0, 3, 1, 2)
        
#         # Bước 2: Tích chập mở rộng và thu thập ngữ cảnh 3x3
#         out = self.conv1(out)
#         out = self.dw_conv(out)
        
#         # Bước 3: Phép nhân SimpleGate (Tuyệt kỹ phân tách nền/vật thể mà không cần Pooling)
#         out = self.sg(out)
        
#         # Bước 4: Khôi phục và cộng Residual
#         out = self.conv2(out)
        
#         return identity + out


#---------------------------LFC------------------------
class LCFAttention(nn.Module):
    def __init__(self, channels=64):
        super(LCFAttention, self).__init__()
        
        # 1. Depthwise 3x3: Chiều chuộng Lung/ISIC (Lấy hình khối, không làm mờ chéo)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        
        # 2. LayerNorm: Chiều chuộng Few-Shot (Miễn nhiễm hoàn toàn với Batch Size = 1)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        
        # 3. ECA Conv1D: Chiều chuộng Deepglobe (Đánh trọng số kênh không cần nén)
        self.eca_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        
        # Bước 1: Trích xuất không gian lân cận
        out = self.dw_conv(x)
        
        # Bước 2: Chuẩn hóa cực kỳ ổn định bằng LayerNorm
        out = out.permute(0, 2, 3, 1) # Đảo chiều B, C, H, W -> B, H, W, C
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2) # Đảo lại B, C, H, W
        
        # Bước 3: Trích xuất trọng số Kênh (ECA)
        v = self.eca_pool(out).squeeze(-1).transpose(-1, -2) # [B, 1, C]
        v = self.eca_conv(v)
        v = v.transpose(-1, -2).unsqueeze(-1) # [B, C, 1, 1]
        
        # Bước 4: Trộn trọng số và Cộng Residual giữ độ nét
        return identity + (out * self.sigmoid(v))


class ContrastiveFeatureTransformer(nn.Module):
    # def __init__(self, in_channels, config_model):
    #     super(ContrastiveFeatureTransformer, self).__init__()

    #     out_channels, kernel_size = config_model.out_channels, config_model.kernel_size
    #     # Add a convolutional layer and a batch normalization layer for learning
    #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
    #     self.bn = nn.BatchNorm2d(out_channels)
    #     self.linear = nn.Conv2d(out_channels, out_channels, 1)

    #     self.prepend_relu = config_model.prepend_relu
    #     self.append_normalize = config_model.append_normalize
    #     #self.debug = config_model.debug

    # def forward(self, x):
    #     if self.prepend_relu:
    #         x = nn.ReLU()(x)
    #     x = self.conv(x)
    #     x = self.bn(x)
    #     x = nn.ReLU()(x)
    #     x = self.linear(x)
    #     if self.append_normalize:
    #         x = F.normalize(x, p=2, dim=1)
    #     return x


#----------NFABlock---------------------------------


    # def __init__(self, in_channels, config_model):
    #     super(ContrastiveFeatureTransformer, self).__init__()
    #     out_channels = config_model.out_channels
        
    #     # Nén kênh từ ResNet
    #     self.project = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
    #         nn.BatchNorm2d(out_channels), 
    #         nn.ReLU(inplace=True)
    #     )
        
    #     # Khối NAFBlock - Đỉnh cao của sự đơn giản và bảo toàn đặc trưng
    #     self.naf_block = NAFBlock(channels=out_channels)
        
    #     # Trộn lại
    #     self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    #     self.prepend_relu = config_model.prepend_relu
    #     self.append_normalize = config_model.append_normalize

    # def forward(self, x):
    #     if self.prepend_relu:
    #         x = F.relu(x)
            
    #     x = self.project(x)
    #     x = self.naf_block(x)
    #     out = self.final_conv(x)
        
    #     if self.append_normalize:
    #         out = F.normalize(out, p=2, dim=1)
            
    #     return out

#-----------------LCF------------------------------


    def __init__(self, in_channels, config_model):
        super(ContrastiveFeatureTransformer, self).__init__()
        out_channels = config_model.out_channels
        
        # Nén kênh từ ResNet (Giữ nguyên BatchNorm mặc định của ResNet)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Khối LCF: Sự dung hòa tối thượng giữa Không gian, Kênh và Độ ổn định
        self.lcf_block = LCFAttention(channels=out_channels)
        
        # Trộn lại
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.prepend_relu = config_model.prepend_relu
        self.append_normalize = config_model.append_normalize

    def forward(self, x):
        if self.prepend_relu:
            x = F.relu(x)
            
        x = self.project(x)
        x = self.lcf_block(x)
        out = self.final_conv(x)
        
        if self.append_normalize:
            out = F.normalize(out, p=2, dim=1)
            
        return out


    # fits the model for one semantic class, therefore does not work with batches
    # mapped_qfeat_vol, aug_qfeat_vols: [aug,c,h,w]
    # mapped_sfeat_vol, aug_sfeat_vols: [k*aug,c,h,w]
    # augmented_smasks: [k*aug,h,w]
    def fit(self, mapped_qfeat_vol, aug_qfeat_vols, mapped_sfeat_vol, aug_sfeat_vols, augmented_smasks, config_fit):
        f_norm = F.normalize if config_fit.normalize_after_fwd_pass else identity_mapping
        optimizer = config_fit.optimizer(self.parameters(), lr=config_fit.lr)
        for epoch in range(config_fit.num_epochs):
            # Pass original and transformed image batches through the model

            # Q
            original_features = f_norm(self(mapped_qfeat_vol), p=2, dim=1)  # fwd pass non-augmented
            transformed_features = f_norm(self(aug_qfeat_vols), p=2, dim=1)  # fwd pass augmented

            qloss = dense_info_nce_loss(original_features, transformed_features,
                                        config_fit.nce) if config_fit.q_nceloss else 0
            if config_fit.keepvarloss:  # 1. idea: Let query and support have the same feature distribution (mean/var per channel)
                qloss += keep_var_loss(original_features, transformed_features)
            # S
            original_features = f_norm(self(mapped_sfeat_vol), p=2, dim=1)  # fwd pass non-augmented
            transformed_features = f_norm(self(aug_sfeat_vols), p=2, dim=1)  # fwd pass augmented

            sloss = dense_info_nce_loss(original_features, transformed_features,
                                        config_fit.nce) if config_fit.s_nceloss else 0
            if config_fit.keepvarloss:
                sloss += keep_var_loss(original_features, transformed_features)

            # 2. class-aware loss: opposite classes should get opposite features
            # for prototype calculation, we want only one prototype per class
            # so we average over features of entire k
            # but calculate prototype for each augmentation individually [k*aug,c,h,w]->[aug,c,k*h*w]->[aug,c]
            kaug, c, h, w = transformed_features.shape
            aug = aug_qfeat_vols.shape[0]
            k = kaug // aug
            if config_fit.protoloss:
                assert not config_fit.o_t_contr_proto_loss, 'only one of the proto losses should be used'
                opposite_proto_sim = opposite_proto_sim_in_aug(transformed_features, augmented_smasks, k, aug)
                if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0): print(
                    'proto-sim intER-class transf<->transf', opposite_proto_sim.item())
                proto_loss = opposite_proto_sim
            elif config_fit.selfattentionloss:
                proto_loss = self_attention_loss(original_features, transformed_features, augmented_smasks,
                                                 augmented_smasks, k, aug)
                if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0): print(
                    'self-att non-transf<->transformed bce', proto_loss.item())
            elif config_fit.o_t_contr_proto_loss:
                o_t_contr_proto_loss = ctrstive_prototype_loss(original_features, transformed_features,
                                                               augmented_smasks, augmented_smasks, k, aug)
                if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0): print(
                    'proto-contr non-transf<->transformed', o_t_contr_proto_loss.item())
                proto_loss = o_t_contr_proto_loss
            else:
                proto_loss = 0

            if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0):
                proto_align_val = proto_align_val_measure(original_features, transformed_features, augmented_smasks, k,
                                                          aug)
                print('proto-sim intRA-class non-transf<->transformed (for validation)', proto_align_val.item())

            # 3. do not let only one image fit well - regularization
            q_s_loss_diff = torch.abs(qloss - sloss) if config_fit.symmetricloss else 0

#------------------------------------------------
#            Nơi tổng hợp loss
#------------------------------------------------

            # Aggregate loss
            # loss = qloss + sloss + q_s_loss_diff + proto_loss

            loss = qloss + sloss + q_s_loss_diff + (1.5 * proto_loss) # loss giúp tăng cho lung
            
            assert loss.isfinite().all(), f"invalid contrastive loss:{loss}"

            # Backpropagation and optimization
            if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0):
                def gradient_magnitude(loss_term):
                    optimizer.zero_grad()
                    loss_term.backward(retain_graph=True)
                    magn = torch.abs(self.conv.weight.grad.mean()) + torch.abs(self.linear.weight.grad.mean())
                    return magn

                q_loss_grad_magnitude = gradient_magnitude(qloss)
                s_loss_grad_magnitude = gradient_magnitude(sloss)
                proto_loss_grad_magnitude = gradient_magnitude(proto_loss)
                q_s_loss_diff_grad_magnitude = gradient_magnitude(q_s_loss_diff)
                display(segutils.tensor_table(q_loss_grad_magnitude=q_loss_grad_magnitude,
                                              s_loss_grad_magnitude=s_loss_grad_magnitude,
                                              proto_loss_grad_magnitude=proto_loss_grad_magnitude,
                                              q_s_loss_diff_grad_magnitude=q_s_loss_diff_grad_magnitude))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config_fit.debug and epoch % 10 == 0: print('loss', loss.detach())


import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import affine
from torchvision.transforms import GaussianBlur, ColorJitter


class AffineProxy:
    def __init__(self, angle, translate, scale, shear):
        self.affine_params = {
            'angle': angle,
            'translate': translate,
            'scale': scale,
            'shear': shear
        }

    def apply(self, img):
        return affine(img, angle=self.affine_params['angle'], translate=self.affine_params['translate'],
                      scale=self.affine_params['scale'], shear=self.affine_params['shear'])


# def affine_proxy(angle, translate, scale, shear):
#     def inner(img):
#         return affine(img, angle=angle, translate=translate, scale=scale, shear=shear)

#     return inner

class Augmen:
    def __init__(self, config_aug):
        self.config = config_aug
        self.blurs, self.jitters, self.affines = self.setup_augmentations()

    def copy_construct(self, blurs, jitters, affines, config_aug):
        self.config = config_aug
        self.blurs, self.jitters, self.affines = blurs, jitters, affines

    def setup_augmentations(self):
        blurkernelsize = self.config.blurkernelsize
        maxjitter = self.config.maxjitter

        maxangle = self.config.maxangle
        translate = (0, 0)
        maxscale = self.config.maxscale
        maxshear = self.config.maxshear

        blurs = []
        jitters = []
        affine_trans = []
        for i in range(self.config.n_transformed_imgs):
            # Randomize kernel size for GaussianBlur
            kernel_size = np.random.choice(torch.tensor(blurkernelsize), (1,)).item()
            blur = GaussianBlur(kernel_size)
            blurs.append(blur)

            # Randomize values for ColorJitter
            brightness_val = torch.rand(1).item() * maxjitter  # up to <maxjitter> change
            contrast_val = torch.rand(1).item() * maxjitter
            saturation_val = torch.rand(1).item() * maxjitter
            jitter = ColorJitter(brightness=brightness_val, contrast=contrast_val, saturation=saturation_val)
            jitters.append(jitter)

            # Random values for each iteration
            angle = torch.randint(-maxangle, maxangle + 1, (1,)).item()
            shear = [torch.randint(-maxshear, maxshear + 1, (1,)).item() for _ in range(2)]
            scale = torch.rand(1).item() * (1 - maxscale) + maxscale
            affine_trans.append(AffineProxy(angle=angle, translate=translate, scale=scale, shear=shear))

        return (blurs, jitters, affine_trans)  # tuple of lists

    def augment(self, original_image, orignal_mask):
        transformed_imgs = []
        transformed_masks = []
        for blur, jitter, affine_trans in zip(self.blurs, self.jitters, self.affines):
            # Apply non-geometric transformations
            t_img = blur(original_image)
            t_img = jitter(t_img)
            t_mask = orignal_mask.clone()

            if self.config.apply_affine:
                t_img = affine_trans.apply(t_img)
                t_mask = affine_trans.apply(t_mask)

            transformed_imgs.append(t_img)
            transformed_masks.append(t_mask)
        return torch.stack(transformed_imgs, dim=1), torch.stack(transformed_masks, dim=1)

    # [bsz,ch,h,w] -> [bsz,aug,ch,h,w], where aug is the number of augmentated images
    def applyAffines(self, feat_vol):
        return torch.stack([trans.apply(feat_vol) for trans in self.affines], dim=1)


class CTrBuilder:
    # call init 1st, pass all config parameters (init a ContrastiveConfig object in your code)
    def __init__(self, config, augmentator=None):
        if augmentator is None:
            augmentator = Augmen(config.aug)
        self.augmentator = augmentator

        self.augimgs = self.AugImgStack(augmentator)

        self.hasfit = False
        self.config = config

    class AugImgStack():
        def __init__(self, augmentator):
            self.augmentator = augmentator
            self.q, self.s, self.s_mask = None, None, None

        def init(self, s_img):
            # c is color channels here, not feature channels
            bsz, k, aug, c, h, w = *s_img.shape[:2], self.augmentator.config.n_transformed_imgs, *s_img.shape[-3:]
            self.q = torch.empty(bsz, aug, c, h, w).to(s_img.device)
            self.s = torch.empty(bsz, k, aug, c, h, w).to(s_img.device)
            self.s_mask = torch.empty(bsz, k, aug, h, w).to(s_img.device)

        def show(self):
            bsz_, k_, aug_ = self.s.shape[:3]
            for b in range(bsz_):
                display('aug x queries', segutils.pilImageRow(*[segutils.norm(img) for img in self.q[b]]))
                for k in range(k_):
                    print('k=', k, ' aug x (s, smask):')
                    display(segutils.pilImageRow(*[segutils.norm(img) for img in self.s[b, k]]))
                    display(segutils.pilImageRow(*self.s_mask[b, k]))

    def showAugmented(self):
        self.augimgs.show()

    # 2nd call makeAugmented
    def makeAugmented(self, q_img, s_img, s_mask):
        # 2. Augmentation
        # 2.1 Apply transformations to images
        self.augimgs.init(s_img)
        self.augimgs.q, _ = self.augmentator.augment(q_img, s_mask)

        for k in range(s_img.shape[1]):
            s_aug_imgs, s_aug_masks = self.augmentator.augment(s_img[:, k], s_mask[:, k])
            self.augimgs.s[:, k] = s_aug_imgs
            self.augimgs.s_mask[:, k] = s_aug_masks
        if self.config.aug.debug: self.augimgs.show()

    # 3rd call build_and_fit
    def build_and_fit(self, q_feat, s_feat, q_feataug, s_feataug, s_maskaug=None):
        if s_maskaug is None: s_maskaug = self.augimgs.s_mask
        self.ctrs = self.buildContrastiveTransformers(q_feat, s_feat, q_feataug, s_feataug, s_maskaug)
        self.hasfit = True

    def buildContrastiveTransformers(self, qfeat_alllayers, sfeat_alllayers, query_feats_aug, support_feats_aug,
                                     supp_aug_mask, s_mask=None):
        contrastive_transformers = []
        l0 = self.config.featext.l0
        # [bsz,k,aug,h,w] -> [k*aug,h,w]
        s_aug_mask = supp_aug_mask.view(-1, *supp_aug_mask.shape[-2:])
        # iterate over feature layers
        for (qfeat, sfeat, qfeataug, sfeataug) in zip(qfeat_alllayers[l0:], sfeat_alllayers[l0:], query_feats_aug[l0:],
                                                      support_feats_aug[l0:]):
            bsz, k, aug, ch, h, w = sfeataug.shape
            # we fit it for exactly one class, so use no batches
            assert bsz == 1, "bsz should be 1"
            assert supp_aug_mask.shape[1] == sfeat.shape[
                1] == k, f'augmented support shot-dimension mismatch:{s_aug_mask.shape[1]=},{sfeat.shape[1]=},(bsz,k,aug,ch,h,w)={bsz, k, aug, ch, h, w}'
            assert supp_aug_mask.shape[2] == qfeataug.shape[1] == aug, 'augmented shot-dimension mismatch'
            # [bsz,c,h,w] -> [1,c,h,w]
            qfeat = qfeat.view(-1, *qfeat.shape[-3:])
            # [bsz,k,c,h,w] -> [k,c,h,w]
            sfeat = sfeat.view(-1, *sfeat.shape[-3:])
            # [bsz,aug,c,h,w] -> [aug,c,h,w]
            qfeataug = qfeataug.view(-1, *qfeataug.shape[-3:])
            # [bsz,k,aug,c,h,w] -> [k*aug,c,h,w]
            sfeataug = sfeataug.view(-1, *qfeataug.shape[-3:])

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            contrastive_head = ContrastiveFeatureTransformer(in_channels=ch, config_model=self.config.model).to(device)

            # 3. Feature volumes from untransformed image need to be geometrically mapped to allow for dense matching
            mapped_qfeat = self.augmentator.applyAffines(qfeat)
            assert mapped_qfeat.shape[1] == aug, "should be 1,aug,c,h,w"
            mapped_qfeat = mapped_qfeat.view(-1, *qfeat.shape[-3:])  # ->[aug,c,h,w]
            mapped_sfeat = self.augmentator.applyAffines(sfeat)
            assert mapped_sfeat.shape[1] == aug and mapped_sfeat.shape[0] == k, "should be k,aug,c,h,w"
            mapped_sfeat = mapped_sfeat.view(-1, *sfeat.shape[-3:])  # ->[k*aug,c,h,w]

            contrastive_head.fit(mapped_qfeat, qfeataug, mapped_sfeat, sfeataug,
                                 segutils.downsample_mask(s_aug_mask, h, w), self.config.fitting)

            contrastive_transformers.append(contrastive_head)
            # show how support image and its augmentations would produce a affinity map
            if s_mask != None:
                display(segutils.to_pil(segutils.norm(dautils.filterDenseAffinityMap(
                    dautils.buildDenseAffinityMat(contrastive_head(sfeat), contrastive_head(sfeataug[:1])),
                    segutils.downsample_mask(s_mask, h, w)).view(1, h, w))))
                display(segutils.to_pil(segutils.norm(dautils.filterDenseAffinityMap(
                    dautils.buildDenseAffinityMat(contrastive_head(qfeat), contrastive_head(sfeat)),
                    segutils.downsample_mask(s_mask, h, w)).view(1, h, w))))
        return contrastive_transformers

    # You have fitted the contrastive transformers, now apply the transform and then pass to the downstream DCAMA
    # you just need to append the empty layers you exluded ([:3]), they're also skipped in dcama
    # Obtain the result of the contrastive head, which will be the new query and support feat representation
    def getTaskAdaptedFeats(self, layerwise_feats):
        if (self.ctrs == None): print("error: call buildContrastiveTransformers() first")
        task_adapted_feats = []

        for idx in range(len(layerwise_feats)):
            if idx < self.config.featext.l0:
                task_adapted_feats.append(None)
            else:
                input_shape = layerwise_feats[idx].shape
                idxth_feat = layerwise_feats[idx].view(-1, *input_shape[-3:])
                forward_pass_res = self.ctrs[idx - self.config.featext.l0](idxth_feat)
                target_shape = *input_shape[:-3], *forward_pass_res.shape[
                                                   -3:]  # borrow channel dim from result, but bsz,k dims from input
                task_adapted_feats.append(forward_pass_res.view(target_shape))

        return task_adapted_feats


class FeatureMaker:
    def __init__(self, feat_extraction_method, class_ids, config=ContrastiveConfig()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.featextractor = feat_extraction_method
        self.c_trs = {ctr: CTrBuilder(config) for ctr in class_ids}
        self.config = config
        self.norm_bb_feats = False

    def extract_bb_feats(self, img):
        with torch.no_grad():
            return self.featextractor(img)

    def create_and_fit(self, c_tr, q_img, s_img, s_mask, q_feat, s_feat):
        if self.config.model.debug: print('contrastive adaption')
        c_tr.makeAugmented(q_img, s_img, s_mask)

        bsz, k, c, h, w = s_img.shape
        aug = c_tr.augmentator.config.n_transformed_imgs
        # [bsz,aug,c,h,w]->[bsz*aug,c,h,w] squeeze for forward pass
        q_feataug = self.extract_bb_feats(c_tr.augimgs.q.view(-1, c, h, w))  # returns layer-list
        # then restore
        q_feataug = [l.view(bsz, aug, *l.shape[1:]) for l in q_feataug]
        # [bsz,k,aug,c,h,w]->[bsz*k*aug,c,h,w]->[bsz,k,aug,c,h,w]
        s_feataug = self.extract_bb_feats(c_tr.augimgs.s.view(-1, c, h, w))
        s_feataug = [l.view(bsz, k, aug, *l.shape[1:]) for l in s_feataug]

        c_tr.build_and_fit(q_feat, s_feat, q_feataug, s_feataug)

    def taskAdapt(self, q_img, s_img, s_mask, class_id):
        ch_norm = lambda t: t / torch.linalg.norm(t, dim=1)
        q_feat = self.extract_bb_feats(q_img)
        bsz, k, c, h, w = s_img.shape
        s_feat = self.extract_bb_feats(s_img.view(-1, c, h, w))
        if self.norm_bb_feats:
            q_feat = [ch_norm(l) for l in q_feat]
            s_feat = [ch_norm(l) for l in q_feat]
        s_feat = [l.view(bsz, k, *l.shape[1:]) for l in s_feat]

        c_tr = self.c_trs[class_id]  # select the relevant ctr for this class

        if c_tr.hasfit is False or c_tr.config.featext.fit_every_episode:  # create and fit a contrastive transformer if not existing yet
            self.create_and_fit(c_tr, q_img, s_img, s_mask, q_feat, s_feat)

        q_feat_t, s_feat_t = c_tr.getTaskAdaptedFeats(q_feat), c_tr.getTaskAdaptedFeats(
            s_feat)  # tocheck: do they require_grad here?
        return q_feat_t, s_feat_t
