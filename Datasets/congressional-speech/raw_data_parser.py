
# python system packages
import csv
import itertools
import logging
import os
import ipdb
from sklearn.model_selection import train_test_split

# external packages
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger()

"""
Utility Functions for Parsing and Extracting things from the raw data

# NOTE: Because of the size of the data files they are not easy to check into Github
# It is assumed that there is a directory called 'hein-bound' in the root level of this
# repo and that it is the extracted data from `https://stacks.stanford.edu/file/druid:md374tz9962/hein-bound.zip`

Details about this dataset are found in `https://stacks.stanford.edu/file/druid:md374tz9962/codebook_v2.pdf`

speeches_{congress}.txt
Format: speech_id | speech_text

1110000001|The Representativeselect and their guests will please remain standing and join in the Pledge of Allegiance.
1110000002|As directed by law. the Clerk of the House has prepared the official roll of the Representativeselect. Certificates of election covering 435 seats in the 111th Congress have been received by the Clerk of the House. and the names of those persons whose credentials show that they were regularly elected as Representatives in accordance with the laws of their respective States or of the United States will be called. The Representativeselect will record their presence by electronic device and their names will be recorded in alphabetical order by State. beginning with the State of Alabama. to determine whether a quorum is present. Representativeselect will have a minimum of 15 minutes to record their presence by electronic device. Representativeselect who have not obtained their voting ID cards may do so now in the Speakers lobby.

descr_{congress}.txt
Format: speech_id|chamber|date|number_within_file|speaker|first_name|last_name|state|gender|line_start|line_end|file|char_count|word_count

1110000006|H|20090106|00006|The CLERK|Unknown|Unknown|Unknown|Special|000710|000717|01062009.txt|228|38
1110000007|H|20090106|00007|Mr. LARSON of Connecticut|Unknown|LARSON|Connecticut|M|000718|000745|01062009.txt|953|157

{congress}_SpeakerMap.txt
Format: speakerid|speech_id|lastname|firstname|chamber|state|gender|party|district|nonvoting

111120160|1110000007|LARSON|JOHN|H|CT|M|D|1|voting
111117010|1110000009|PENCE|MIKE|H|IN|M|R|6|voting

"""

def load_speaker_map(congress, data_directory="../hein-bound", use_pandas=False):
    """
    Load in the data from the speakers map file
    Can return either a dataframe representation of this file
    or a dictionary keyed on the speech id {speech_id: {speech_data}}

    congress: a string with the congressional session ex: "111"
    """
    if type(congress) == int:
        if congress < 100:
            congress = "0" + str(congress)
    filepath = os.path.join(data_directory, f"{congress}_SpeakerMap.txt")
    if use_pandas:
        return pd.read_csv(filepath, delimiter="|", encoding="ISO-8859-1")
    with open(filepath, encoding="ISO-8859-1") as f:
        reader = csv.DictReader(f, delimiter="|")
        return {str(row["speech_id"]): row for row in reader}

def load_descriptions(congress, data_directory="../hein-bound", use_pandas=False):
    """
    Load in the data from the speech description file which has metadata about the circumstances around the speech
    Can return either a dataframe representation of this file
    or a dictionary keyed on the speech id {speech_id: {speech_data}}

    congress: a string with the congressional session ex: "111"
    """
    if type(congress) == int:
        if congress < 100:
            congress = "0" + str(congress)
    filepath = os.path.join(data_directory, f"descr_{congress}.txt")
    if use_pandas:
        return pd.read_csv(filepath, delimiter="|", encoding="ISO-8859-1")
    
    with open(filepath, encoding="ISO-8859-1") as f:
        reader = csv.DictReader(f, delimiter="|")
        return {str(row["speech_id"]): row for row in reader}

def load_speeches(congress, data_directory="../hein-bound", use_pandas=False):
    """
    Load in the data from the speeches file which has the actual text os the speeches
    Can return either a dataframe representation of this file
    or a dictionary keyed on the speech id {speech_id: {speech_data}}

    congress: a string with the congressional session ex: "111"
    """
    # handle edge case of congress having a 0 prepended to it
    if type(congress) == int:
        if congress < 100:
            congress = "0" + str(congress)

    filepath = os.path.join(data_directory, f"speeches_{congress}.txt")
    if use_pandas:
        return pd.read_csv(filepath, delimiter="|", encoding="ISO-8859-1", error_bad_lines=False)
    with open(filepath, encoding="ISO-8859-1") as f:
        reader = csv.DictReader(f, delimiter="|")
        return {str(row["speech_id"]): row for row in reader}

def load_merged_session_data(congress, data_directory="../hein-bound", use_pandas=False):
    """
    Return a data structure that has all of the independent data of speeches, speaker map
    and descriptions file merged together
    """
    speeches = load_speeches(congress, data_directory, use_pandas)
    descriptions = load_descriptions(congress, data_directory, use_pandas)
    speaker_map = load_speaker_map(congress, data_directory, use_pandas)

    if use_pandas:
        unified_dataframe = speeches
        unified_dataframe = unified_dataframe.merge(descriptions, on="speech_id", suffixes=("_speech", "_descr"))
        unified_dataframe = unified_dataframe.merge(speaker_map, on="speech_id", suffixes=("_speech", "_speaker"))
        return unified_dataframe

    # with dictionaries
    for speech_id in speeches:
        
        # perform the 'update' command on the speech dict
        if speech_id in descriptions:
            speeches[speech_id].update(descriptions.get(speech_id))
        
        # NOTE: Everywhere the description says there is the 'SPEAKER' or
        # The VICE PRESIDENT or similar officers there is no record in the
        # speaker_map file
        # TODO: Decide how to handle this nicely/gracefully
        if speech_id in speaker_map:
            speeches[speech_id].update(speaker_map.get(speech_id))

    return speeches

def load_session_speeches_by_party(congress, data_directory="../hein-bound", use_pandas=False):
    """
    Convenience to load all the speeches from a single congress and to separate them into
    the list of texts by the Democrats, Republicans, Independents and Unlabeled.
    NOTE: This will only work for some of the more recent congressional sessions as there
    are different parties for some of the earlier congresses.

    4 lists of dictionary objects for: Dem, GOP, Ind, UNK
    OR
    4 Data Frames for: Dem, GOP, Ind, UNK
    Return Format: [{},{}],[{},{}],[{},{}],[{},{}]
    """
    merged_data = load_merged_session_data(congress, data_directory, use_pandas)
    if use_pandas:
        return (merged_data[merged_data["party"] == "D"],
                merged_data[merged_data["party"] == "R"],
                merged_data[merged_data["party"] == "I"],
                merged_data[~merged_data["party"].isin(["I", "R", "D"])])

    # Case where we have a large set of merged speech data
    merged_dicts = {"D": [], "R": [], "I": [], "U": []}
    for speech_id in merged_data:
        # This handles the case where the speaker is not clearly linked to
        # a known entity with a row in the speaker_map
        if "party" in merged_data[speech_id]:
            merged_dicts[merged_data[speech_id]["party"]].append(merged_data[speech_id])
        else:
            merged_dicts["U"].append(merged_data[speech_id])
    
    return merged_dicts["D"], merged_dicts["R"], merged_dicts["I"], merged_dicts["U"]

def load_random_corpus_set(congress, data_directory="../hein-bound"):
    """
    Function to load a set of speeches, shuffle them and then return 2 distinct sets of texts
    The purpose is to be able to compare word embeddings between a random shuffle of the data
    and the polarized embeddings
    NOTE: Pandas only
    Return: random_speeches_df_1, random_speeches_df_2
    """
    merged_data = load_merged_session_data(congress, data_directory, use_pandas=True)
    df_1, df_2 = train_test_split(merged_data, test_size=0.5)
    return df_1, df_2


class SpeechesIterator:
    """
    Adapted from 
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
    #training-your-own-model

    Since we need to loop & re-loop through the speeches, this is a wrapper around
    `load_merged_session_data` to include multiple sessions at once,
    lazily loading session by session
    """
    def __init__(
        self,
        tokenizer,
        congresses,
        parties=["D", "R", "I"],
        data_directory="../hein-bound",
        min_sent_len=4,
        ngram_range=(1, 1),
        use_noun_chunks=False,
        lowercase=True,
        vocab=None,
        workers=8,
        batch_size=5000,
        random_set=False,
    ):
        self.tokenizer = tokenizer
        self.data_directory = data_directory
        self.congress_range = congresses
        self.parties = parties
        self.min_sent_len = min_sent_len
        self.ngram_range = ngram_range
        self.use_noun_chunks = use_noun_chunks
        self.lowercase = lowercase
        self.vocab = vocab
        self.workers = workers
        self.batch_size = batch_size

        # set up the parser
        to_lower = lambda x: x.text
        if lowercase:
            to_lower = lambda x: x.lower_
        
        to_vocab = lambda x: x
        if vocab is not None:
            to_vocab = lambda x: self.vocab[x]
        
        self.token_parser = lambda x: to_vocab(to_lower(x))
    
    def __iter__(self):
        """
        Wrapper around `load_merged_session_data` to include multiple sessions at once,
        for a single party, lazily loading session by session

        TODO: works with pandas only
        """
        for congress in self.congress_range:
            LOGGER.info(f"On congress {congress}")
            speeches = load_merged_session_data(
                congress, self.data_directory, use_pandas=True
            )
            
            speeches = speeches.loc[speeches["party"].isin(self.parties)]
            
            speeches = speeches.sort_values(["date", "speech_id"]) # likely redundant
            speeches["yearmon"] = speeches["date"].astype(str).str.slice(stop=6)
            _iterator = zip(
                speeches.iterrows(),
                self.tokenizer.pipe(
                    speeches["speech"].astype(str),
                    n_threads=self.workers,
                    batch_size=self.batch_size,
                )
            )

            for (_, row), doc in tqdm(_iterator, total=len(speeches)):
                for i, sent in enumerate(doc.sents):
                    if len(sent) >= self.min_sent_len:
                        yield {
                            "id": f"{row.speech_id}_{i}",
                            "speech_id": row.speech_id,
                            "speaker_id": row.speakerid,
                            "source_file": row.file,
                            "text": str(sent),

                            "congress": congress,
                            "date": row.date,
                            "yearmon": row.yearmon,

                            "party": row.party,
                            "chamber": row.chamber_speaker,
                            "first_name": row.firstname,
                            "last_name": row.lastname,
                            "gender": row.gender_speaker,
                            "state": row.state_speaker,
                            "district": row.district,
                            "is_voting": row.nonvoting == "voting",
                        }
            
            LOGGER.info("Completed reading through data")

# The main function contains some basic tests for the functions in this file
# as well as example usage
if __name__ == "__main__":
    
    # set the logging level so that the test messages will appear
    logging.basicConfig(level=logging.DEBUG)

    DATA_DIR = "../hein-bound"
    d, g, i, n = load_session_speeches_by_party("111", data_directory=DATA_DIR, use_pandas=False)
    print(len(d), len(g), len(i), len(n))
    quit()



    dict_speakers = load_speaker_map("111", data_directory=DATA_DIR)
    assert dict_speakers["1110000007"]["lastname"] == "LARSON"
    LOGGER.info("Load Speaker Map Passed")

    pandas_speakers = load_speaker_map("111", use_pandas=True, data_directory=DATA_DIR)
    assert len(pandas_speakers) > 0
    LOGGER.info("Load Speaker Map Passed with Pandas")

    descriptions = load_descriptions("111", data_directory=DATA_DIR)
    assert descriptions["1110000007"]["chamber"] == "H"
    LOGGER.info("Load Descriptions Passed")

    speeches = load_speeches("111", data_directory=DATA_DIR)
    assert speeches["1110000007"]
    assert type(speeches["1110000007"]["speech"]) == str
    LOGGER.info("Load Speeches Passed")

    merged_data = load_merged_session_data("111", data_directory=DATA_DIR)
    assert len(merged_data["1110000007"]["speech"]) > 0
    assert merged_data["1110000007"]["chamber"] == "H"
    assert merged_data["1110000007"]["party"] == "D"
    LOGGER.info("Load Merged Data Structure Passed")

    merged_data_pandas = load_merged_session_data("111", use_pandas=True, data_directory=DATA_DIR)
    assert "party" in merged_data_pandas
    LOGGER.info("Load Merged Data Structure with Pandas Passed")
