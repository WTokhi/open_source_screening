"""  Module for parsing names."""

import os
import glob
import logging
import unidecode

from abc import ABC

import pandas as pd
import numpy as np


class NameMixin(ABC):
    """Mixin for parsing names from PEP and sanction lists or leaked papers."""

    # LK: Persoonlijk zou ik geen logger op een mixin zetten.
    # LK: Laat de classes zelf maar loggers maken met de correcte naam.
    # LK: Deze __init__ kan dan helemaal weg.
    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)
        self._log.info(
            "----------Abstract baseclass `NameMixin` is initialized.----------"
        )

    @staticmethod
    def transliterate(value: object) -> str:
        """Normalize name and apply transliteration and case folding.

        Parameters
        ----------
        value: str
            Name as string value.

        Returns
        -------
        str
            Normalized name.
        """

        try:
            # LK: Waarom extern package en niet gewoon unicodedata?
            return unidecode.unidecode(value).casefold()
        except TypeError:
            # If value is not a string, return it as is.
            return value

    @staticmethod
    def convert_islamic_to_gregorian(dob: str) -> str:
        """Convert Islamic date of birth to Gregorian date.

        Parameters
        ----------
        dob : str
            Date of birth in Islamic (Hijri calendar) year or Gregorian.

        Returns
        -------
        str
            Date of birth in Gregorian year.
        """
        # LK: Datum en jaar lopen beetje door elkaar; wat wil je doen?
        # LK: Op zich niet zoveel moeite om jaar uit een datum te plukken toch?
        try:
            dob = str(dob)

            if len(dob) == 4:
                if int(dob) < 1800:
                    dob = int(dob) + 579
                    return str(dob)
                else:
                    return dob
            else:
                return dob

        except ValueError:
            raise ValueError("The input should be a valid string or integer")

    # TODO: Extend list name.
    # LK: clean_name / prune_name zou ik betere benaming vinden.
    @staticmethod
    def parse_name(value: str) -> str:
        """Parse person name by replacing the given characters with blank.

        Parameters
        ----------
        value: str
            Name as string.

        Returns
        -------
        str
            Cleansed name.
        """
        try:
            # LK: "Dhr.", "Mevr." etc? En misschien nog strip() op het resultaat?
            for clean in ["@", "iii", "jr.", "sr.", "Sir", "Lord"]:
                value = value.replace(clean, "")
            return value
        except TypeError:
            # LK: Misschien beter om deze check / conversie centraal te maken?
            # If value is not a string, return it as is
            return value


class Pep(NameMixin):
    """Subclass for parsing PEP lists.

    Note: Loads PEP lists provided as CSV file(s). If there are
    multiple CSV files, they will be combined into a single data
    set.

    Parameters
    ----------
    input_path: str
        Path to one or more CSV files.
    """

    def __init__(self, input_path: str) -> None:
        self._log = logging.getLogger(__name__)
        self._log.info("----------PEP parser has started----------")

        csv_files = glob.glob(os.path.join(input_path + "/pep", "*.csv"))
        if not csv_files:
            msg = f"No PEP CSV files found in {self._input_path!r}."
            self._log.error(msg)
            raise FileNotFoundError(msg)

        input_data = []
        for csv_path in csv_files:
            try:
                input_data.append(
                    pd.read_csv(csv_path, delimiter=",", encoding="utf-8")
                )
            # LK: Deze fouten kunnen hier eigenlijk niet optreden toch?
            except (KeyError, ValueError, RuntimeError) as error:
                msg = f"Error parsing file {csv_path!r}: {error!r}."
                self._log.error(msg)

        self._input_data = pd.concat(input_data, axis=0)

    # LK: Vat de functie samen op de eerste regel (dus in 1 regel).
    # LK: Als dat lastig blijkt is het vaak een teken dat de functie teveel doet.
    # LK: Het bundelen/afscheiden van alle DOB bewerkingen zou een begin kunnen zijn?
    def pep_parser(self) -> pd.DataFrame():
        """The main purpose here is to normalize(nfkd), transliterate and casefold the
        names of all entities. The results are returned as a pandas.DataFrame().

        Returns
        -------
        pandas.DataFrame
            DataFrame containing normalized names and parsed date of births.
        """
        input_data = self._input_data
        data_frame = input_data[
            ["person", "gender", "start", "end", "DOB", "catalog", "position"]
        ]
        data_frame = data_frame.rename(
            columns={
                "catalog": "country",
                "DOB": "dob",
                "start": "date_start",
                "end": "date_end",
            }
        )

        # Remove entries with missing dob and start date
        data_frame = data_frame.dropna(subset=["dob", "date_start"])

        # Select entries where length dob is greater than 3
        data_frame = data_frame.loc[data_frame["dob"].str.len() > 3, :]

        # Remove leading and trailing white spaces
        data_frame = data_frame.applymap(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        # fix dob for islamic years
        data_frame["dob"] = data_frame["dob"].apply(self.convert_islamic_to_gregorian)

        # TODO: Find a better solution. For the time being mask this entity because dob is not correct.
        data_frame = data_frame[
            ~data_frame["dob"].isin(("1973-09-31", "1951-06-31", "2346"))
        ]

        # Normalize names
        data_frame = data_frame.assign(
            person_normalized=lambda df: df["person"]
            .map(self.transliterate)
            .map(self.parse_name)
        )

        # Drop duplicates
        # LK: Doe je dat hier omdat normalisatie duplicaten oplevert?
        # LK: Op zich is zo vroeg mogelijk droppen efficienter.
        data_frame.drop_duplicates(inplace=True)

        # Insert index
        # LK: Waar is dit goed voor?
        data_frame.index = pd.Index(
            [f"pep_{idx + 1}" for idx in data_frame.index], name="index"
        )

        self._log.debug(f"Number of unique persons : {data_frame['person'].nunique()}")
        self._log.debug(
            f"Number of unique countries : {data_frame['country'].nunique()}"
        )
        self._log.debug(f"Number of rows : {data_frame.shape[0]}")

        return data_frame


class Sanction(NameMixin):
    """Subclass for parsing sanction lists.

    Parameters
    ----------
    input_path: str
        Path to one or more the CSV files with sanctioned names.
    """

    def __init__(self, input_path: str) -> None:
        self._log = logging.getLogger(__name__)
        self._log.info("----------Sanction class is initialized----------")

        # LK: Eigenlijk gebeurt hier hetzelfde als de vorige class...
        # LK: Gedeelde __init__ in een base class een optie?
        input_data = pd.DataFrame()
        csv_files = glob.glob(os.path.join(input_path + "/sanction_lists", "*.csv"))

        if not csv_files:
            msg = f"No csv files found in {input_path!r}."
            self._log.error(msg)
            raise FileNotFoundError(msg)

        for csv_path in csv_files:
            try:
                load_data = pd.read_csv(csv_path, delimiter=",", encoding="utf-8")
                input_data = pd.concat([input_data, load_data], axis=0)
            except (KeyError, ValueError, RuntimeError) as error:
                msg = f"Error parsing file {csv_path!r}: {error!r}."
                self._log.error(msg)

        self._input_data = input_data

    # LK: Vergelijkbaar met vorige class; er gebeurt hier veel...
    def sanction_parser(self) -> pd.DataFrame:
        """The main purpose here is to normalize(nfkd), transliterate and casefold the
        names of all entities. The results are returned as a pandas.DataFrame().

        Returns
        -------
        pandas.DataFrame
            DataFrame containing normalized names and parsed date of births.
        """

        self._log.info("----------Sanction list parser has started----------")

        # LK: LET OP: Dit maakt alleen een referentie aan (want mutable object)!
        # LK: De drop_duplicates hieronder wijzigt dus ook self._input_data.
        sanction_list = self._input_data

        # Drop duplicates because the sanctions list can contain same entities
        sanction_list.drop_duplicates(inplace=True)

        # Select persons
        # LK: Je kunt DataFrame methods ook "chainen" he...
        # LK: Je hebt nu namelijk een hele hoop losse regels
        sanction_list_np = sanction_list.loc[
            sanction_list["schema"] == "Person",
            ["schema", "name", "birth_date", "countries", "sanctions", "dataset"],
        ]
        sanction_list_np["schema"] = sanction_list_np["schema"].replace(
            "Person", "person"
        )

        # Remove entries with missing dob. do discuss with team.
        sanction_list_np.dropna(subset=["birth_date"], inplace=True)

        # Lowercase person
        sanction_list_np["name"] = sanction_list_np["name"].str.lower()

        # Extract dob. Persons with multiple date of birth are regarded as different entity
        sanction_list_np["birth_date_fixed"] = (
            sanction_list_np["birth_date"].str.split(";").explode("birth_date_fixed")
        )

        # Extract sanction start date
        sanction_list_np["date_start"] = (
            sanction_list_np["sanctions"].str.split(" ").str[-1]
        )

        sanction_list_np["date_end"] = np.nan

        # Rename column
        sanction_list_np.rename(
            columns={
                "birth_date_fixed": "dob",
                "countries": "country",
                "name": "person",
            },
            inplace=True,
        )

        # select relevant columns
        columns = ["person", "dob", "date_start", "date_end", "country", "dataset"]

        sanction_list_np = sanction_list_np[columns]

        # normalize names
        sanction_list_np["person_normalized"] = sanction_list_np["person"].map(
            self.transliterate
        )

        # fix dob for islamic years
        sanction_list_np["dob"] = sanction_list_np["dob"].map(
            self.convert_islamic_to_gregorian
        )

        # drop duplicates: since there is an overlap between the lists
        sanction_list_np.drop_duplicates(inplace=True)

        # LK: Kan wel compacter met chaining:
        # sanction_list_np = (
        #     sanction_list_np
        #     .assign(
        #         person_normalized=lambda df: df["person"].map(self.transliterate),
        #         dob=lambda df: df["dob"].map(self.convert_islamic_to_gregorian),
        #     )
        #     .drop_duplicates()
        # )

        # check results
        self._log.debug(
            f"Number of unique persons : {sanction_list_np['person'].nunique()}"
        )
        self._log.debug(
            f"Number of unique countries : {sanction_list_np['country'].nunique()}"
        )
        self._log.debug(f"Number of rows : {sanction_list_np.shape[0]}")

        return sanction_list_np


class LeakedPapers(NameMixin):
    """Subclass for parsing leaked papers."""

    def __init__(self, input_path: str) -> None:
        """Read data: officers, entities, relationships and addresses.

        Parameters
        ----------
        input_path: str
            Path to the csv files, there could be more than one file.

        """
        # Dit kan veel compacter met een dicts
        # self._datasets = {}
        # file_specs = {
        #     "nodes-officers": ["node_id", "name", "sourceID"],
        #     "nodes-addresses": ["node_id", "address", "countries"],
        #     "nodes-entities": [
        #         "node_id",
        #         "name",
        #         "incorporation_date",
        #         "inactivation_date",
        #     ],
        #     "relationshisps": [
        #             "node_id_start",
        #             "node_id_end",
        #             "rel_type",
        #             "link",
        #             "start_date",
        #             "end_date",
        #         ]
        # }
        # for dataset, columns in file_specs.items():
        #     csv_file = os.path.join(input_path + "leaked_papers", f"{dataset}.csv")
        #     self._datasets[dataset] = pd.read_csv(
        #         csv_file,
        #         delimiter=",",
        #         encoding="utf-8",
        #         usecols=columns,
        #         low_memory=False,
        #     )

        csv_files = glob.glob(os.path.join(input_path + "/leaked_papers", "*.csv"))

        if not csv_files:
            msg = f"No csv files found in {input_path!r}."
            self._log.error(msg)
            raise FileNotFoundError(msg)

        for csv_path in csv_files:
            if "nodes-addresses" in csv_path:
                self._address = pd.read_csv(
                    csv_path,
                    delimiter=",",
                    encoding="utf-8",
                    usecols=["node_id", "address", "countries"],
                    low_memory=False,
                )
            elif "nodes-entities" in csv_path:
                self._entities = pd.read_csv(
                    csv_path,
                    delimiter=",",
                    encoding="utf-8",
                    usecols=[
                        "node_id",
                        "name",
                        "incorporation_date",
                        "inactivation_date",
                    ],
                    low_memory=False,
                )
            elif "nodes-officers" in csv_path:
                self._officers = pd.read_csv(
                    csv_path,
                    delimiter=",",
                    encoding="utf-8",
                    usecols=["node_id", "name", "sourceID"],
                    low_memory=False,
                )
            elif "relationships" in csv_path:
                self._relationships = pd.read_csv(
                    csv_path,
                    delimiter=",",
                    encoding="utf-8",
                    usecols=[
                        "node_id_start",
                        "node_id_end",
                        "rel_type",
                        "link",
                        "start_date",
                        "end_date",
                    ],
                    low_memory=False,
                )

    def leaked_papers_parser(self) -> pd.DataFrame:
        """The main purpose here is to normalize(nfkd), transliterate and casefold the
        names of all entities. The results are returned as a pandas.DataFrame().


        Returns
        -------
        pandas.DataFrame
            DataFrame containing normalized names and parsed date of births.

        """
        # LK: Consistentie; logging opzetten in de __init__
        self._log = logging.getLogger(__name__)

        self._log.info("----------Leaked Papers parser has started----------")

        # Get the address of the officers.
        # LK: Of officers.merge(relationships, ...).merge(address, ...)
        officer_address = pd.merge(
            self._officers,
            self._relationships,
            left_on="node_id",
            right_on="node_id_start",
        )
        officer_address = pd.merge(
            officer_address,
            self._address,
            left_on="node_id_end",
            right_on="node_id",
            suffixes=("", "_address"),
            how="inner",
        )

        # LK: Is een drop(columns=...) niet eenvoudiger?
        col_source = list(self._officers.columns)
        col_address = ["address", "countries"]
        officer_address = officer_address.loc[
            officer_address["address"].notna(), col_source + col_address
        ]

        # LK: Method chaining...
        officer_address["name"] = (
            officer_address["name"].apply(str).map(self.transliterate)
        )
        officer_address["address"] = officer_address["address"].map(self.transliterate)

        # Get the entities of the officers.
        officer_address_entity = pd.merge(
            officer_address,
            self._relationships,
            left_on="node_id",
            right_on="node_id_start",
        )
        officer_address_entity = pd.merge(
            officer_address_entity,
            self._entities,
            left_on="node_id_end",
            right_on="node_id",
            suffixes=("", "_entity"),
            how="inner",
        )

        # LK: Waarom niet meteen selectie hier maken?
        cols = [
            "node_id",
            "name",
            "address",
            "countries",
            "link",
            "name_entity",
            "start_date",
            "end_date",
            "incorporation_date",
            "inactivation_date",
            "sourceID",
        ]

        # check results
        self._log.debug(
            f"Number of unique persons : {officer_address_entity['name'].nunique()}"
        )
        self._log.debug(
            f"Number of unique countries : {officer_address_entity['countries'].nunique()}"
        )
        # LK: Kolom selectie veranderd niks aan deze output ;-)
        self._log.debug(f"Number of rows : {officer_address_entity[cols].shape[0]}")

        return officer_address_entity[cols]
