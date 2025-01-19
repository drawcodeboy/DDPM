from .DDPM.ddpm import DDPM

def load_model(**cfg):
    if cfg['name'] == 'DDPM':
        return DDPM(time_steps=cfg['time_steps'],
                    beta_schedule=cfg['beta_schedule'])