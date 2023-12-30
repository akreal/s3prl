from .expert import UpstreamExpert as _UpstreamExpert


def espnet_asr_custom(ckpt, *args, config=None, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

def espnet_asr_bloomzmms_tmi(*args, refresh=False, **kwargs):
    return espnet_asr_custom("/home/ak/data/bloomzmms-tmi/model.bin", "/home/ak/data/bloomzmms-tmi/config.yaml")
