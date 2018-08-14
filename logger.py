import logging
import logging.config
import json


def setupLogging(config_filename="logConfig.json"):
    with open(config_filename, 'r') as logging_configuration_file:
        config = json.load(logging_configuration_file)
    logging.config.dictConfig(config)
