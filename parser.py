"""  Module for parsing names."""
# system packages
import os
import json
import glob
import logging

import datetime as dt

import pandas as pd
import numpy as np

import unicodedata
import unidecode

class NameMixin:
    """ 
    Superclass with functions for parsing pep list, sanction list and leaked papers.
    
    """

    def __init__(self) -> None:

        self._log = logging.getLogger(__name__)

        self._log.info("----------Superclass Parser is initialized----------")


    def transliterate(self, value: object) -> str:
        """ Normalize person name by applying unicode normalizing(nfkd), transliteration and casefold().
        
        Parameters
        ----------
        value: object
            person name 

        Returns
        -------
        object
            value normalized to letters between a and z. 
        """

        try:
            return unidecode.unidecode(value).casefold()
        except TypeError:
            # If value is not a string, return it as is
            return value
    
    def convert_dob(self, dob: str) -> str:
        """  Convert Islamic year of birth to Gregorian year by adding 579.

        Parameters
        ----------
        dob : str
            Date of birth in Islamic (Hijri calendar) year or Gregorian.

        Returns
        -------
        str
            Year of birth in Gregorian year.
        """
        try:
            dob = str(dob)

            if (len(dob)==4):
                if (int(dob) < 1800):
                    dob = int(dob) + 579
                    return str(dob)
                else:
                    return dob
            else:
                return dob
        
        except ValueError:
            raise ValueError("The input should be a valid string or integer")

    def parse_name(self, value: str) -> str:
        """ Parse person name by replacing the given characters with blank.
        
        Parameters
        ----------
        dob: object
            person name 

        Returns
        -------
        object
            parsed string
        """
        try:
            for name in ["@", "iii", "jr.", "sr.", "Sir", "Lord"]:
                value = value.replace(name, "")
            return value
        except TypeError:
            # If value is not a string, return it as is
            return value


class Pep(NameMixin):
    """ Subclass for parsing pep lists."""

    def __init__(self, input_path: str) -> None:
        """ All of the pep files are concatonatd.

        Parameters
        ----------
        input_path: str
            Path to the csv files, there could be more than one file.

        """
        # super().__init__()
        
        self._log = logging.getLogger(__name__)
        self._log.info("----------PEP parser has started----------")
        self._input_path = input_path
        input_data: pd.DataFrame = pd.DataFrame()

        csv_files: list = glob.glob(os.path.join(self._input_path + "/pep", "*.csv"))

        if not csv_files:
            msg = f"No csv files found in {self._input_path!r}."
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

    def pep_parser(self) -> pd.DataFrame():
        """ The main purpose here is to normalize(nfkd), transliterate and casefold the 
        names of all entities. The results are returned as a pandas.DataFrame().

        Returns
        -------
        pandas.DataFrame
            DataFrame containing normalized names and parsed date of births.
        """
        input_data = self._input_data
        data_frame = input_data[["person", "gender", "start", "end", "DOB", "catalog", "position"]]
        data_frame = data_frame.rename(columns={"catalog": "country", "DOB": "dob", "start": "date_start", "end": "date_end"})

        # Remove entries with missing dob and start date
        data_frame = data_frame.dropna(subset=["dob"])
        data_frame = data_frame.dropna(subset=["date_start"])

        # Select entries where length dob is greater than 3
        data_frame = data_frame.loc[data_frame["dob"].str.len()>3,:]
        
        # Remove leading and trailing white spaces
        data_frame= data_frame.applymap(lambda x: x.strip() if isinstance(x, str) else x)


        # fix dob for islamic years
        data_frame["dob"] = data_frame["dob"].apply(self.convert_dob)

        # TODO: Find a better solution. For the time being mask this entity because dob is not correct.
        data_frame = data_frame.loc[data_frame["dob"]!="1973-09-31", :]
        data_frame = data_frame.loc[data_frame["dob"]!="1951-06-31", :]
        data_frame = data_frame.loc[data_frame["dob"]!="2346", :]
        
        # Normalize names
        data_frame["person_normalized"] = data_frame["person"].map(self.transliterate)
        data_frame["person_normalized"] = data_frame["person_normalized"].map(self.parse_name)

        # drop duplicates
        data_frame.drop_duplicates(inplace=True)

        # insert index
        item_list = list(range(1, data_frame.shape[0]+1))
        data_frame.index =   ["pep_" + str(item) for item in item_list]
        data_frame.index.name = "index"
        
        self._log.debug(f"Number of unique persons : {data_frame['person'].nunique()}")
        self._log.debug(f"Number of unique countries : {data_frame['country'].nunique()}")  
        self._log.debug(f"Number of rows : {data_frame.shape[0]}")
        
        return data_frame

class Sanction(NameMixin):
    """ Subclass for parsing sanction lists."""

    def __init__(self, input_path: str) -> None:
        """The main purpose here is to concatenate the sanction files. 

        Parameters
        ----------
        input_path: str
            Path to the csv files, there could be more than one file.

        """
        self._log = logging.getLogger(__name__)
        self._log.info("----------Sanction class is initialized----------")

        input_data: pd.DataFrame = pd.DataFrame()
        csv_files: list = glob.glob(os.path.join(input_path + "/sanction_lists", "*.csv"))

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

    def sanction_parser(self) -> pd.DataFrame:
        """ The main purpose here is to normalize(nfkd), transliterate and casefold the 
        names of all entities. The results are returned as a pandas.DataFrame().

        Returns
        -------
        pandas.DataFrame
            DataFrame containing normalized names and parsed date of births.
        """

        self._log.info("----------Sanction list parser has started----------")

        sanction_list = self._input_data

        # Drop duplicates because the sanctions list can contain same entities
        sanction_list.drop_duplicates(inplace=True)

        # Select persons
        sanction_list_np = sanction_list.loc[sanction_list["schema"]=="Person", ["schema","name", "birth_date", "countries", "sanctions", "dataset"]]
        sanction_list_np["schema"] = sanction_list_np["schema"].replace("Person", "person")

        # Remove entries with missing dob. do discuss with team.
        sanction_list_np.dropna(subset=["birth_date"], inplace=True)

        # Lowercase person
        sanction_list_np["name"] = sanction_list_np["name"].str.lower()

        # Extract dob. Persons with multiple date of birth are regarded as different entity
        sanction_list_np["birth_date_fixed"] = sanction_list_np.loc[:, "birth_date"].str.split(";")
        sanction_list_np = sanction_list_np.explode("birth_date_fixed")

        # Extract sanction start date
        sanction_list_np["date_start"] = sanction_list_np.loc[:, "sanctions"].str.split(" ").str[-1]

        sanction_list_np["date_end"] = np.nan

        # Rename column
        sanction_list_np.rename(columns={"birth_date_fixed": "dob",\
            "countries": "country",\
            "name": "person"}, inplace=True)

        #select relevant columns
        columns = ["person", "dob", "date_start", "date_end", "country", "dataset"]

        sanction_list_np = sanction_list_np[columns]

        # normalize names
        sanction_list_np["person_normalized"] = sanction_list_np["person"].map(self.transliterate)

        # fix dob for islamic years
        sanction_list_np["dob"] = sanction_list_np["dob"].apply(self.convert_dob)

        # drop duplicates: since there is an overlap between the lists
        sanction_list_np.drop_duplicates(inplace=True)

        # check results
        self._log.debug(f"Number of unique persons : {sanction_list_np['person'].nunique()}")
        self._log.debug(f"Number of unique countries : {sanction_list_np['country'].nunique()}") 
        self._log.debug(f"Number of rows : {sanction_list_np.shape[0]}")

        return sanction_list_np

class LeakedPapers(NameMixin):
    """ Subclass for parsing leaked papers."""

    def __init__(self, input_path: str) -> None:
        """ Read data: officers, entities, relationships and addresses.

        Parameters
        ----------
        input_path: str
            Path to the csv files, there could be more than one file.

        """
        csv_files: list = glob.glob(os.path.join(input_path + "/leaked_papers", "*.csv"))

        if not csv_files:
            msg = f"No csv files found in {input_path!r}."
            self._log.error(msg)
            raise FileNotFoundError(msg)

        for csv_path in csv_files:
            if "nodes-addresses" in csv_path:
                self._address = pd.read_csv(csv_path, delimiter=",", encoding="utf-8",\
                        usecols=["node_id", "address", "countries"], low_memory=False)
            elif "nodes-entities" in csv_path:
                self._entities = pd.read_csv(csv_path, delimiter=",", encoding="utf-8", \
                    usecols=['node_id', 'name', 'incorporation_date', 'inactivation_date'], low_memory=False)
            elif "nodes-officers" in csv_path:
                self._officers = pd.read_csv(csv_path, delimiter=",", encoding="utf-8", \
                    usecols=["node_id", "name", "sourceID"], low_memory=False)
            elif "relationships" in csv_path:
                self._relationships = pd.read_csv(csv_path, delimiter=",", encoding="utf-8", \
                    usecols=["node_id_start", "node_id_end", "rel_type", "link", "start_date", "end_date"], low_memory=False)


    def leaked_papers_parser(self) -> pd.DataFrame:
        """ The main purpose here is to normalize(nfkd), transliterate and casefold the 
        names of all entities. The results are returned as a pandas.DataFrame().


        Returns
        -------
        pandas.DataFrame
            DataFrame containing normalized names and parsed date of births.
        
        """
        self._log = logging.getLogger(__name__)

        self._log.info("----------Leaked Papers parser has started----------")


        # Get the address of the officers.
        officer_address = pd.merge(self._officers, self._relationships, left_on='node_id', right_on='node_id_start')
        officer_address = pd.merge(officer_address, self._address, left_on='node_id_end', right_on='node_id', suffixes=('', '_address'), how="inner")
        
        col_source = list(self._officers.columns)
        col_address = ["address", "countries"]
        officer_address = officer_address.loc[officer_address["address"].notna(), col_source + col_address]

        officer_address["name"] = officer_address["name"].apply(str).map(self.transliterate)
        officer_address["address"] = officer_address["address"].map(self.transliterate)
        
        # Get the entities of the officers.
        officer_address_entity = pd.merge(officer_address, self._relationships, left_on='node_id', right_on='node_id_start')
        officer_address_entity = pd.merge(officer_address_entity, self._entities, left_on='node_id_end', right_on='node_id', suffixes=('', '_entity'), how="inner")
        cols = [
                'node_id', 'name', 'address', 'countries', 'link', 'name_entity',
                'start_date', 'end_date',  'incorporation_date','inactivation_date','sourceID'
                ]

        # check results
        self._log.debug(f"Number of unique persons : {officer_address_entity['name'].nunique()}")
        self._log.debug(f"Number of unique countries : {officer_address_entity['countries'].nunique()}")
        self._log.debug(f"Number of rows : {officer_address_entity[cols].shape[0]}")
        
        return officer_address_entity[cols]



    