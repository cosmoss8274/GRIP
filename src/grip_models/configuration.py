"""Default configuration for the base energy model."""

import json
from types import SimpleNamespace


def load_config(config_path):
    """Load config from JSON file and convert to object."""
    default_config = {
        "model": {
            "num_atom": 5,
            "num_recycle": 3,
            "embeddings": {
                "seq_info_dim": 21,
                "m_dim": 1024,
                "tf_dim": 768,
                "seq_channel": 128,
                "pair_channel": 64,
                "max_relative_feature": 32,
                "go_embed_dim": 64
            },
            "heads": {
                "seqhead": {
                    "enabled": True,
                },
                "pairhead": {
                    "enabled": True,
                },
                "losshead": {
                    "enabled": True,
                    "l1_weight": 0.01,
                    "l2_weight": 0.01
                }
            }
        },
        "loss": {
            "seq": {
                "config": {},
                "loss_fn": "seq_loss",
                "weight": 1.0
            },
            "mrf_pair": {
                "config": {
                    "pair_norm_weight": 0.01,
                    "site_norm_weight": 0.01
                },
                "loss_fn": "mrf_pair_loss",
                "weight": 1.0
            }
        },
        "data": {
            "batch_size": 1,
            "padding": 100,
            "n_channels_emb": 21,
            "n_channels_seq": 21,
            "n_channels_esm": 33,
            "go_embedding_dim": 64,
            "m": 1024,
            "z": 128
        }
    }

    if config_path:
        with open(config_path) as f:
            user_config = json.load(f)
            default_config = merge_dicts(default_config, user_config)

    def dict_to_namespace(d):
        if not isinstance(d, dict):
            return d  # Return non-dict objects as-is
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))  # Recursive call for nested dict
            else:
                setattr(namespace, key, value)
        return namespace


    config = dict_to_namespace(default_config)
    return config
