"""Module for screening clients."""
import logging # type: ignore
import argparse # type: ignore
import pandas as pd # type: ignore
import yaml # type: ignore

from parser import Parser # type: ignore
from matching import StringMatching # type: ignore

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVELS = "debug", "info", "warning", "error", "critical"


def load_config(config_path: str) -> dict:
    """Load params.yml and return dictionary with class input parameters
    Parameters
    ----------
    input_path: str
        Path to the YAML configuration file.

    Returns
    -------
    dictionary
        Dictionary containing configuration parameters.
    """

    try:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError as error:
        msg = f"Could not load config file - path: {config_path!r}"
        raise FileNotFoundError(msg) from error


def main(config_path: str) -> None:
    """ Main routine to start open source data parser process.

    Parameters
    ----------
    config_path: str
        Path to the YAML configuration file.
    """

    config = load_config(config_path)

    log_file = config.get('log_file')
    log_level = config.get('log_level')

    # TODO: when log level is missing in config.yml debug en info are not logged in .log file. Fix?
    if log_level:
        logging.basicConfig(
            level=log_level.upper(),
            format=LOG_FORMAT,
        )

    if log_file:
        logger = logging.getLogger()
        logger_formatter = logging.Formatter(LOG_FORMAT)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(logger_formatter)
        logger.addHandler(file_handler)

    if not config.get('open_source_data_path'):
        msg = "Open source data path is missing in the configuration file."
        logger.error(msg)
        raise TypeError(msg)
    elif not config.get('client_data_path'):
        msg = "Client data path is missing in the configuration file."
        logger.error(msg)
        raise TypeError(msg)
    else:
        open_source_data_path = config.get('open_source_data_path')
        client_data_path = config.get('client_data_path')

    open_source_parser = Parser(open_source_data_path)
    string_match = StringMatching(client_data_path)
        
    
    try: 
        for type_screening in config.get("type_screening"):
            if type_screening == "pep":
                pep_parsed = open_source_parser.pep_parser()
                pep_matched = string_match.match_client_data(pep_parsed, type_screening)
                pep_matched.to_csv("output/pep_matched.csv")
            elif type_screening == "sanction":
                sanction_parsed = open_source_parser.sanction_parser()
                sanction_matched = string_match.match_client_data(sanction_parsed, type_screening)
                sanction_matched.to_csv("output/sanction_matched.csv")   
            elif type_screening == "leaked papers":
                leaked_papers_parsed = open_source_parser.leaked_papers_parser()
                leaked_papers_matched = string_match.match_client_data(leaked_papers_parsed, type_screening, config.get("train_model"))
                leaked_papers_matched.to_csv("output/leaked_papers_matched.csv") 
            else:
                pass       
    except TypeError as error:
        msg = "input argument 'type screening' is missing or its format is incorrect."
        logger.error(msg)
        raise error(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Command line interface for open source screening.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="file location of the input parameters",
        # default="",
        required=True
    )
    args = parser.parse_args()

    main(args.config)
