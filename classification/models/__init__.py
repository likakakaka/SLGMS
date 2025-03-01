from .vmambav2.vmamba import DINO_VSSM as VSSMV2

from classification.models.vmambav2.csms6s import CrossScan, CrossMerge, Cross_Recombination_Scan, Cross_Recombination_Merge

def build_model(config):
    model = VSSMV2(
            patch_size=config.MODEL.VSSMV2.PATCH_SIZE,
            in_chans=config.MODEL.VSSMV2.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSMV2.DEPTHS,
            dims=config.MODEL.VSSMV2.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSMV2.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSMV2.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSMV2.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSMV2.SSM_DT_RANK == "auto" else int(config.MODEL.VSSMV2.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSMV2.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSMV2.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSMV2.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSMV2.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSMV2.SSM_INIT,
            forward_type=config.MODEL.VSSMV2.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSMV2.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSMV2.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSMV2.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSMV2.PATCH_NORM,
            norm_layer=config.MODEL.VSSMV2.NORM_LAYER,
            downsample_version=config.MODEL.VSSMV2.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSMV2.PATCHEMBED,
            gmlp=config.MODEL.VSSMV2.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            return_maps=True,
            is_change=True
        )
    return model


