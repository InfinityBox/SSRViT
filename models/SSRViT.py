from timm.models.layers import trunc_normal_
from .layers import *


def get_block(block_type, **kargs):
    if block_type == 'mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type == 'ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type == 'tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_patch(size, lam):
    S= size[2]
    cut_rat = np.sqrt(1. - lam)
    idx = np.random.choice(S, int(S * cut_rat), replace=False)

    return idx


def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay == 'linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay == 'fix':
        # use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr


class SRViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes_i=1000, num_classes_t=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 p_emb='4_2', head_dim=None, skip_lam=1.0, order=None, mix_token=False, return_dense=False,
                 patch_shuffle=False, viz_mode=False, inference=False):
        super().__init__()
        self.num_classes_i = num_classes_i
        self.num_classes_t = num_classes_t
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim_cls = embed_dim if num_classes_i == 0 else num_classes_i
        self.output_dim_tk = embed_dim if num_classes_t == 0 else num_classes_t
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb == '4_2':
                patch_embed_fn = PatchEmbed4_2
            elif p_emb == '4_2_128':
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                              embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if order is None:
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    skip_lam=skip_lam)
                for i in range(depth)])
        else:
            # use given order to sequentially generate modules
            dpr = get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i],
                          dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                          skip_lam=skip_lam)
                for i in range(len(order))])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes_i) if num_classes_i > 0 else nn.Identity()
        self.aux_head = nn.Linear(embed_dim, self.num_classes_t) if num_classes_t > 0 else nn.Identity()

        self.return_dense = return_dense
        self.patch_shuffle = patch_shuffle
        self.mix_token = mix_token

        self.viz_mode = viz_mode
        self.inference = inference

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        # trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes_i, global_pool=''):
        self.num_classes_i = num_classes_i
        self.head = nn.Linear(self.embed_dim, num_classes_i) if num_classes_i > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_features(self, x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        patch_h, patch_w = x.shape[2], x.shape[3]

        # shuffle patches in each image
        if self.patch_shuffle and self.training:
            idx = torch.randperm(patch_h * patch_w)
            x = x.view(x.shape[0], x.shape[1], -1)[:, :, idx]
        else:
            idx = torch.arange(patch_h * patch_w)

        # token level mixtoken augmentation
        if self.mix_token and self.training:
            x = x.view(x.shape[0], x.shape[1], -1)
            lam = np.random.beta(self.beta, self.beta)
            # bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            idd = rand_patch(x.size(), lam)
            if len(idd) == 0:
                temp_x = x.clone()
            else:
                temp_x = x.clone()
                temp_x[:, :, idd] = x.flip(0)[:, :, idd]
            x = temp_x
        else:
            idd = None

        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(x.device).type_as(x).clone().detach()
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)  # (B,N,C)
        x_cls = self.avgpool(x.transpose(1, 2))  # (B,C,1)
        x_cls = torch.flatten(x_cls, 1)  # (B,C)

        pred_cls = self.head(x_cls)
        x_aux = self.aux_head(x)  # shape:(Batch_size, patch_num*patch_num, classes_prob)

        if self.return_dense:

            if not self.training:
                if self.viz_mode:
                    return pred_cls, x_aux
                else:
                    return pred_cls, x_aux, x.clone(), x_cls

            # recover the mixed part
            if self.mix_token and self.training:
                temp_x = x_aux.clone()
                temp_x[:, idd, :] = x_aux.flip(0)[:, idd, :]
                x_aux = temp_x

            # recover the shuffle images
            if self.patch_shuffle and self.training:
                _, id = idx.sort()
                x_aux = x_aux[:, id, :].view(x_aux.size())

            return pred_cls, x_aux, idd, idx, x_cls
        return pred_cls, x_cls


class SViT(nn.Module):
    def __init__(self, dim, depth=6, num_heads=8, n_classes=1, out=False):
        super().__init__()

        self.blocks = nn.ModuleList([AttBlock(dim=dim, num_heads=num_heads) for i in range(depth)])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, n_classes)
        self.n_classes = n_classes
        self.output_f = out

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x_cls = self.avgpool(x.transpose(1, 2))
        x_cls = torch.flatten(x_cls, 1)
        logits = self.head(x_cls)
        return logits
