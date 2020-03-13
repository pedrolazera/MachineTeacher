import configparser

def read_configuration_file(path: str):
    raise NotImplementedError
    config = configparser.ConfigParser()
    config.read(path)

    # verificar 3 componentes:
    ## (a) learner - nome e configuracoes
    ## (b) teacher - nome e configuracoes
    ## (c) dataset - path csv do arquivo