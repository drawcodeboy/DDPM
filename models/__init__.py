from .DDPM.ddpm import DDPM

def load_model(**cfg):
    if cfg['name'] == 'DDPM':
        return DDPM(time_steps=cfg['time_steps'],
                    beta_schedule=cfg['beta_schedule'],
                    input_dim=cfg['input_dim'],
                    init_dim=cfg['init_dim'],
                    dim_mults=cfg['dim_mults'],
                    time_emb_theta=cfg['time_emb_theta'],
                    time_emb_dim=cfg['time_emb_dim'],
                    attn_emb_dim=cfg['attn_emb_dim'],
                    attn_heads=cfg['attn_heads'],
                    dropout_rate=cfg['dropout_rate'],
                    device=cfg['device'])