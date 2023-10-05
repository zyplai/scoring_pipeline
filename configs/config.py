from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=["configs/base_params.toml", "configs/names_mapper.toml"]
)
