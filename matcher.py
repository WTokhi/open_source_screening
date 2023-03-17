"""  Module for matching names with different methods"""
import logging
import pickle
import string

import numpy as np
import pandas as pd

from fuzzywuzzy import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class NameMatcher:
    """Class for matching client data with pep list, sanction list and leaked papers.

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

        self._log.info(f"Using client data path: {client_data_path!r}")
        self._client_data_path = client_data_path

    def match_name(
        self,
        open_source_parsed: pd.DataFrame(),
        type_screening: str,
        train_model: bool = True,
    ) -> pd.DataFrame:
        """Parses all client csv files from the input path into a DataFrame.

        The main purpose here is to concatenate the client files and parse
        the names and date of births. That is normalize(nfkd), transliterate and
        casefold the names of all entities. The results are returned as a
        pandas.DataFrame.

        Parameters
        ----------
        input_path: str
            Path to the csv files, there could be more than one file.

        open_source_parsed: pd.DataFrame
            Parsed open source entities.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing names matching the open source dataset.

        """
        self._log.info(
            f"----------{type_screening.capitalize()} string matching has started----------"
        )

        input_data = pd.DataFrame()

        # LK: Conventie is dat je attributen in de __init__ aanmaakt.
        # LK: Waarom worden deze hier ingesteld?
        self._type_screening = type_screening
        self._train_model = train_model

        try:
            # LK: Checken of alle kolommen in de clienten data zitten?
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
            # LK: Alleen leaked gebruikt dus NN matching methode?
            # LK: Waarom niet overal consistent toepassen?
            # LK: Beide methodes een eigen class; zij zijn behoorlijk anders?
            return self._levenshtein(input_data, open_source_parsed)

    def _levenshtein(
        self,
        df_client: pd.DataFrame(),
        df_open_source: pd.DataFrame(),
        limit: int = 2,
        threshold: int = 75,
    ) -> pd.DataFrame():
        """Match client names with open source lists using levenshtein distance.

            Step 1. Match on date of birth to reduce set.
            Step 2. Match on client lastname and complete name from open source.

        Parameters
        ----------
        df_client: pd.DataFrame()
            Contains information of clients to screen.
        df_open_source: pd.DataFrame()
            Contains PEP or sanction list information.
        limit: int
            Maximum number of matches per client.
        threshold: int
            Lower threshold to count as a match.

        Returns
        -------
        pd.DataFrame()
            DataFrame of matched clients.
        """

        df_client = df_client.assign(year=lambda x: x.client_dob.dt.year)

        # "year" column is filled when "dob" is an int, else empty.
        # LK: Dit lijkt me ik een vrij sketchy manier om jaar te isoleren?
        # LK: Daarbij; zou je dit niet in de prep van die OS lijsten willen doen?
        df_open_source["year"] = pd.to_numeric(df_open_source["dob"], "coerce")

        df_open_source["dob"] = pd.to_datetime(df_open_source["dob"]).where(
            df_open_source["year"].isna(), other=pd.NaT
        )

        # DOB matching: Union- merge rows on year OR datetime
        merged = pd.concat(
            [
                # LK: Hier matchen ook alle volledige DOB (want dit is minder exact)...
                df_client.merge(df_open_source, how="inner", on="year"),

                # LK: Wat voegt deze nog toe? Alle koppels zijn hierboven al gemaakt?
                # LK: Misschien is match op alleen geboortejaar wel prima?
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
            merged
            .query(f"match_percentage >= {threshold}")
            .drop(columns=["year"])
            .groupby(["client_name", "client_dob"])
            .apply(lambda grp: grp.nlargest(limit, "match_percentage"))
            .reset_index(drop=True)
        )

        # LK: Er wordt dus ook nog gematched op adres?
        # LK: Zo ja, dan zou ik dat wel als stap in de docstring zetten...
        aggregated = self._same_address(aggregated, df_client)

        # LK: Je hebt vrij vaak select / rename in je code.
        # LK: Probeer dit te concentreren op 1 plek, bijvoorbeeld bij het inlezen.
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

        # LK: En had die _y suffix niet vermeden kunnen worden?
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

        # LK: Zijn ze geinteresseerd in huisgenoten?
        nb_housemates = aggregated.loc[
            aggregated["client_rol"] == "housemate", :
        ].shape[0]

        self._log.debug(f"Number of potential hits: {aggregated.shape[0]}")
        self._log.debug(f"Number of housemates: {nb_housemates}")

        return aggregated

    # LK: De KNN methode is eigenlijk een heel ding op zich...
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
            self._log.info("Building TFIDF matrix.")

            # Training names are not unique because of multiple addresses.
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

        # LK: De gebruiker weet niet wat de test data is...
        self._log.info("Matching client names using KNN.")
        test_tfidf = vectorizer.transform(test["client_name"])
        distances, indices = nbrs.kneighbors(test_tfidf)

        # LK: Deze code kan wel simpeler naar mijn idee...
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

        # LK: Twee vormen van adres matching?
        matches["address_match(higher is better)"] = matches.apply(
            lambda df: self._token_set_ratio(df["lp_address"], df["client_address"]),
            axis="columns",
        )
        matches = self._same_address(matches, test)
        matches.drop_duplicates(inplace=True)

        self._log.info("Done")

        return matches.sort_values(by="index", ascending=True)

    # LK: Duidelijk maken dat het om adressen gaat in naamgeving?
    @staticmethod
    def _token_set_ratio(match: str, target: str) -> float:
        """Matches two addresses using fuzzy matching.

        Addresses are cleansed by converting to lowercase, removing punctuations.
        The addresses are then tokenized and the tokens are sorted alphabetically.
        Finally the amount of overlap is computed using fuzz.ratio().

        Parameters:
        ----------
        match: str
            Address to match as string.

        target: str
            Address to match against as string.

        Returns:
        -------
        float
            Measure of similarity between 0 and 100.


        """
        return fuzz.token_set_ratio(match, target)

    # LK: Naamgeving; aangeven dat het om achternamen gaat.
    @staticmethod
    def _ratio(client: str, target: str) -> float:
        """Match lastnames against names in open source data.

        Parameters
        ----------
        client: str
            Client name as string.

        target: str
            Name to match against as string.

        Returns
        -------
        float
            Measure of similarity between 0 and 100.
        """

        client = client.split()
        vector = []
        for _, token in enumerate(client):
            match = fuzz.ratio(target, token)
            vector.append(match)

        return max(vector)

    def _same_address(
        self, df_matched: pd.DataFrame, df_client: pd.DataFrame
    ) -> pd.DataFrame:
        """Create a list of clients registered at the same address.

            For clients who have a string similarity score above a certain
            threshold check whether other clients are also registered
            at the same address.

        Parameters
        ----------
        df_matched: pandas.DataFrame
            DataFrame where client information (last name and dob) is already
            matched with open source data.

        df_client: pandas.DataFrame
            DataFrame with client information including address.

        Returns
        -------
        pandas.DataFrame
            Returns clients who are registered at the same address.
        """

        df_same_address = df_matched.merge(
            df_client,
            left_on=["client_address"],
            right_on=["client_address"],
            # LK: Lijkt me dat je met deze validatie weinig opschiet?
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
