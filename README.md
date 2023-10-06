## README ##

This is repo for AutoDevelopment of PD models.

### Structure ###

0. configs - configuration files used in dev
1. data_prep - data processing
2. features - feature engineering
3. models - model training
4. validation - validation of results and samples
5. utils - custom utility functions that are frequently used

### Getting Started ###
In the CLI run the following commands:
- `poetry install` - installs dependencices, virtual environment to work in
- `poetry run python commands.py run_scoring` - runs the whole scoring pipeline on a given data

### Useful Commands ###
- `make fmt` - formats everything according to PEP-8;
- `make clean` - removes __pycache__ files everywhere;
- `make test` - runs testing

### Notes & TODO ###
This is still WIP project with basic features. There are still the following tasks:
- Develop DVC to version the data and get it remotely;
- Develop pycaret pipeline;
- Integrate data and model validation (modules are ready, just need to add);
- Add Single Factoral Analysis (module is ready, just need to add);
- Prepare the module extensive report on all aspects of the model and EDA;
- Hyperparameters tuning

### Contacts ###
In case of any questions write in telegram: @kshurik
