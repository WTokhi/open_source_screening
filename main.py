"""Module for screening clients."""
import logging
import argparse
import pandas as pd
import yaml

from matcher import NameMatcher
from parser import Pep, Sanction, LeakedPapers

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

    log_file = config.get("log_file")
    log_level = config.get("log_level", "debug")

    # Set root handler to debug.
    if log_level:
        logging.basicConfig(
            level=logging.DEBUG, format=LOG_FORMAT,
        )

    if log_file:
        root_logger = logging.getLogger()

        # Change level of terminal logger.
        root_logger.handlers[0].setLevel(log_level.upper())

        # Add file handler.
        file_handler = logging.FileHandler(filename=log_file)
        logger_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(logger_formatter)
        root_logger.addHandler(file_handler)

    if not config.get("open_source_data_path"):
        msg = "Open source data path is missing in the configuration file."
        logger.error(msg)
        raise TypeError(msg)
    elif not config.get("client_data_path"):
        msg = "Client data path is missing in the configuration file."
        logger.error(msg)
        raise TypeError(msg)
    else:
        open_source_data_path = config.get("open_source_data_path")
        client_data_path = config.get("client_data_path")

    string_match = NameMatcher(client_data_path)

    try:
        for type_screening in config.get("type_screening"):
            if type_screening == "pep":
                pep_parser = Pep(open_source_data_path)
                pep_parsed = pep_parser.pep_parser()
                pep_matched = string_match.match_name(pep_parsed, type_screening)
                pep_matched.to_csv("output/pep_matched.csv")

            elif type_screening == "sanction":
                sanction_parser = Sanction(open_source_data_path)
                sanction_parsed = sanction_parser.sanction_parser()
                sanction_matched = string_match.match_name(
                    sanction_parsed, type_screening
                )
                sanction_matched.to_csv("output/sanction_matched.csv")

            elif type_screening == "leaked papers":
                leaked_papers_parser = LeakedPapers(open_source_data_path)
                leaked_papers_parsed = leaked_papers_parser.leaked_papers_parser()
                leaked_papers_matched = string_match.match_name(
                    leaked_papers_parsed, type_screening, config.get("train_model")
                )
                leaked_papers_matched.to_csv("output/leaked_papers_matched.csv")
            else:
                print("hello")
    except TypeError as error:
        msg = "input argument 'type screening' is missing or its format is incorrect."
        logger.error(msg)
        raise error(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Command line interface for open source screening."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="file location of the input parameters",
        # default="",
        required=True,
    )
    args = parser.parse_args()

    main(args.config)
