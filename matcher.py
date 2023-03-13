"""  Module for parsing the survey data as a pandas DataFrame"""
# system packages
import os
import json
import glob
import logging
import pickle

import datetime as dt

import pandas as pd
import numpy as np
import unicodedata
from fuzzywuzzy import fuzz
import string


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class NameMatcher:
    """ Class for matching client data with  pep list, sanction list and leaked papers.

    Parameters
    -----------
    client_data_path: str
        Path to the client data file.

    """

    STOP_WORDS = [
        "limited",
        "ltd",
        "incorporated",
        "inc",
        "sa",
        "corp",
        "group",
        "holdings",
        "investments",
        "finance",
    ]

    def __init__(self, client_data_path: str) -> None:

        self._log = logging.getLogger(__name__)

        self._log.info("----------The StringMatching class is initialized----------")

        self._client_data_path: str = client_data_path

        self._log.info(f"Received client data input path: {self._client_data_path!r}")

    def match_name(
        self,
        open_source_parsed: pd.DataFrame(),
        type_screening: str,
        train_model: bool = True,
    ) -> pd.DataFrame:
        """Parses all client csv files from the input path into a DataFrame.

        The main purpose here is to concatenate the open source files and parse the names and date of births. 
        That is normalize(nfkd), transliterate and casefold the names of all entities. The results are
        returned as a pandas.DataFrame.

        Parameters
        ----------
        input_path: str
            Path to the csv files, there could be more than one file.

        open_source_parsed: pd.DataFrame
            Parsed open source entities.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing clients where a match is found in the open source dataset.
        
        """

        self._log.info(
            f"----------{type_screening.capitalize()} string matching has started----------"
        )

        input_data: pd.DataFrame = pd.DataFrame()
        self._type_screening = type_screening
        self._train_model = train_model

        try:
            input_data = pd.read_csv(
                self._client_data_path,
                delimiter="|",
                encoding="utf-8",
                parse_dates=["client_dob"],
            )
        except FileNotFoundError as error:
            msg = f"No csv file found in {self._client_data_path!r}."
            self._log.error(msg)
            raise FileNotFoundError(msg) from error

        # TODO: implement try/excpet
        if self._type_screening == "leaked papers":
            return self._knn(input_data, open_source_parsed)
        else:
            return self._levenshtein(input_data, open_source_parsed)

    # TODO: Add docstrings
    def _levenshtein(
        self,
        df_client: pd.DataFrame(),
        df_open_source: pd.DataFrame(),
        limit: int = 2,
        threshold: int = 75,
    ) -> pd.DataFrame():
        """ Fuzzy match client name with names from open source with levenshtein distance.

            step 1. Match on date fo birth to reduce set.
            step 2. Match on client lastname and complete name from open source.
        
        Parameters
        ----------
        df_client: pd.DataFrame()
            Contains information with respect to clients. 
        
        df_open_source: pd.DataFrame()
            Could be pep list or sanction list.
        
        limit: int
            .....

        threshold: int
            .....

        Returns
        -------
        pd.DataFrame()
            ...... 
        """

        df_client = df_client.assign(year=lambda x: x.client_dob.dt.year)

        # "year" column is filled when "dob" is an int, else empty.
        df_open_source["year"] = pd.to_numeric(df_open_source["dob"], "coerce")

        df_open_source["dob"] = pd.to_datetime(df_open_source["dob"]).where(
            df_open_source["year"].isna(), other=pd.NaT
        )

        # DOB matching: Union- merge rows on year OR datetime
        merged = pd.concat(
            [
                df_client.merge(df_open_source, how="inner", on="year"),
                df_client.merge(
                    df_open_source.drop(columns=["year"]),
                    how="inner",
                    left_on="client_dob",
                    right_on="dob",
                ),
            ]
        ).drop_duplicates()

        # fuzzy match on lastname and normalized name with .ratio() method
        merged["match_percentage"] = merged.apply(
            lambda df: self._ratio(df["person_normalized"], df["client_lastname"]),
            axis="columns",
        )

        # Aggregate to find the n largest match percentages per client
        aggregated = (
            merged.query(f"match_percentage >= {threshold}")
            .drop(columns=["year"])
            .groupby(["client_name", "client_dob"])
            .apply(lambda grp: grp.nlargest(limit, "match_percentage"))
            .reset_index(drop=True)
        )

        aggregated = self._same_address(aggregated, df_client)

        columns = [
            "client_name_y",
            "client_lastname_y",
            "client_dob_y",
            "client_address",
            "client_rol",
            "date_start",
            "date_end",
            "match_percentage",
        ]

        aggregated = aggregated[columns]

        aggregated.rename(
            columns={
                "client_name_y": "client_name",
                "client_lastname_y": "client_lastname",
                "client_dob_y": "client_dob",
                "date_start": "start_date",
                "date_end": "end_date",
            },
            inplace=True,
        )

        nb_housemates = aggregated.loc[
            aggregated["client_rol"] == "housemate", :
        ].shape[0]

        self._log.debug(f"Number of potential hits: {aggregated.shape[0]}")
        self._log.debug(f"Number of housemates: {nb_housemates}")

        return aggregated

    def _knn(self, test: pd.DataFrame(), train: pd.DataFrame()) -> pd.DataFrame():

        self._log.info(
            f"Vectorizing the data - this could take a few minutes for large datasets."
        )

        vectorizer = TfidfVectorizer(
            # Decode the data
            encoding="utf-8",
            # Remove accents and perform NFKD character normalization
            strip_accents="unicode",
            # Preprocessing options.
            preprocessor=self._preprocessing,
            # Character ngrams with length 3
            analyzer="char_wb",
            ngram_range=(3, 3),
            # Frequency pruning
            max_df=1.0,
            min_df=1,
        )

        if self._train_model:
            self._log.info("Getting tfidf matrix.")

            # Train['name'] values are not unique because a name can have more than 1 address.
            train_tfidf = vectorizer.fit_transform(train["name"])

            self._log.info("Getting nearest neighbours.")
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(train_tfidf)

            with open("output/vectorizer.pickle", "wb") as f:
                pickle.dump(vectorizer, f)
            with open("output/nbrs.pickle", "wb") as f:
                pickle.dump(nbrs, f)
        else:

            with open("output/nbrs.pickle", "rb") as f:
                nbrs = pickle.load(f)
            with open("output/vectorizer.pickle", "rb") as f:
                vectorizer = pickle.load(f)

        self._log.info("Fitting test data.")
        test_tfidf = vectorizer.transform(test["client_name"])
        distances, indices = nbrs.kneighbors(test_tfidf)

        self._log.info("Finding matches...")
        matches = []
        for i, j in enumerate(indices):
            temp = [
                j[0],
                test["client_name"][i],
                test["client_address"][i],
                np.array(train["name"][j + 1])[0],
                np.array(train["address"][j + 1])[0],
                round(distances[i][0], 2),
            ]
            matches.append(temp)

        matches = pd.DataFrame(
            matches,
            columns=[
                "index",
                "client_name",
                "client_address",
                "lp_name",
                "lp_address",
                "name_match",
            ],
        )
        matches["address_match(higher is better)"] = matches.apply(
            lambda df: self._token_set_ratio(df["lp_address"], df["client_address"]),
            axis="columns",
        )

        matches = self._same_address(matches, test)
        matches.drop_duplicates(inplace=True)

        self._log.info("Done")

        return matches.sort_values(by="index", ascending=True)

    @staticmethod
    def _token_set_ratio(value_1: str, value_2: str) -> float:
        """ This function tokenizes, lowercases, removes punctuations and sorts the strings alphabetically and then joins the two strings with fuzz.ratio(). 
        
        Parameters:
        ----------
        value_1: string
            It is address
        
        value_2: string
            It is address
        
        Returns:
        -------
        Float:
            Returns a measure of similarity between 0 and 100 bases on Levenstein distance. 


        """
        return fuzz.token_set_ratio(value_1, value_2)

    @staticmethod
    def _ratio(target: str, source: str) -> int:
        """ Compare the lastname of the client with the different name tokens from the open source data.
        
        Parameters
        ----------
        target: string
            Is client name. 
        
        source: string
            Complete name from the open source data.

        Returns
        -------
        integer
            Returns a measure of the sequence similarity between 0 and 100. 
        """

        target = target.split()
        vector = []

        for _, token in enumerate(target):
            match = fuzz.ratio(source, token)
            vector.append(match)

        return max(vector)

    def _same_address(
        self, df_matched: pd.DataFrame, df_client: pd.DataFrame
    ) -> pd.DataFrame:
        """ Extract list of clients with the same address.
        
            For clients who have a string similarity score above 
            a certain threshold check whether other clients are also registered 
            at the same address.
        
        Parameters
        ----------
        df_matched: pandas.DataFrame
            Is a pandas.DataFrame where client information (last name and dob) is already matched with open source entities.
        
        df_client: pandas.DataFrame
            Is a pandas.DataFrame where client information is found such as address.
        
        Returns
        -------
        pandas.DataFrame
            Returns clients who are registered at the same address.
        """

        df_same_address = df_matched.merge(
            df_client,
            left_on=["client_address"],
            right_on=["client_address"],
            validate="m:m",
            suffixes=("", "_y"),
            how="left",
        )
        # Assign rol -> pep/sanctioned/leaked paper or housemate
        df_same_address["client_rol"] = np.where(
            df_same_address["client_name"] == df_same_address["client_name_y"],
            self._type_screening,
            "housemate",
        )
        df_same_address["name_match(higher is beter)"] = np.where(
            df_same_address["client_rol"] != "housemate",
            df_same_address["name_match"],
            np.nan,
        )

        return df_same_address

    # TODO: Add docstrings
    def _preprocessing(self, value):

        value = value.lower()
        value = self._strip_punctuation(value)
        value = self._remove_stop_words(value)
        return value

    # TODO: Add docstrings
    @staticmethod
    def _strip_punctuation(value):
        return "".join(c if c not in string.punctuation else "" for c in value)

    # TODO: Add docstrings
    def _remove_stop_words(self, value):
        return " ".join([word for word in value.split() if word not in self.STOP_WORDS])
