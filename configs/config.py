from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=[
        "configs/base_params.toml",
        "configs/data_prep.toml",
        "configs/model.toml",
    ]
)
