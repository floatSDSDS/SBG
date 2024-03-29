## SBG

Source code of WWW 2022 Paper "Modeling User Behavior with Graph Convolution for Personalized Product Search".

- environment
```shell script
conda env create -f SBG/air.yml
```

- prepare data
  - download amazon data from https://jmcauley.ucsd.edu/data/amazon/
  - specify the input and output dir in `preprocessing/prepare_amazon.py`
  - run `preprocessing/prepare_amazon.py` for corresponding datasets
  - config dataset in `persearch.config.cfg_data` 

- reproduce the results
```shell script
cd SBG
python main.py -d amazon_software@ -e 200 -r 5
```


- configuration
    - model configuration in `persearch.config.cfg_model`
    - training data generator configuration in `persearch.config.cfg_gen`  
    - other command line interactions: `persearch.args`

- customize model
    - training data generator for the model
        - build Generator pipeline in `persearch.gen`
        - register gen in `persearch.config.cfg_gen`, add its name in your 
            `persearch.config.cfg_model` with the key `'generator'`
    - implement `forward, do_train, and f_loss`
    - doc refer to `persearch.model.Base`
        - example as `persearch.model.zam.ZAM`
    - register model in `persearch.config.cfg_model`
    - load it in `model/__init__.py`
    - write test in `exps.py`
    
- log
    - summary stored in `logs/<dataset>/<dataset_ver>/<arg.caption>/<timestamp>`
