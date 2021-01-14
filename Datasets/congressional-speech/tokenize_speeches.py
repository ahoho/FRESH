# standard librarires
import argparse
import logging
import json
from pathlib import Path

import spacy
import spacy.lang
from spacy.lang.en import English

from raw_data_parser import SpeechesIterator

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# we'll try out using pandas for now
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="hein-bound")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument(
        "--congress",
        default="114",
        nargs="+",
        type=int,
        help="Congress(es) to load. For multiple, specify a range with 2 values, `--congress 110 115`"
    )
    parser.add_argument("--spacy_model",
        default="default",
        help="spacy language model to use"
    )

    parser.add_argument("--min_sent_len",
        default=4,
        type=int,
        help="Minimum sentence length to consider"
    )
    parser.add_argument("--min_sents_per_example",
        default=1,
        type=int,
        help="Sentences to concatenate into a single example (if speech spans multiple)"
    )
    parser.add_argument("--not_lazy",
        default=True,
        dest="lazy",
        action="store_false",
        help="Load all data in at once?"
    )
    parser.add_argument("--lowercase",
        default=True,
        action="store_true",
    )
    parser.add_argument("--no_lowercase",
        default=True,
        dest="lowercase",
        action="store_false",
    )
    parser.add_argument("--ngram_range",
        default=[1],
        nargs="+",
        type=int,
        help="How to parse ngrams."
    )
    
    parser.add_argument("--workers",
        default=-1,
        type=int,
        help="Number of parallel workers"
    )
    parser.add_argument("--batch_size",
        default=5000,
        type=int,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    congresses = range(min(args.congress), max(args.congress) + 1)
    # The congresses that are lower than 100 have a 0 prefixed to them, after determining this
    # range of values we can cast the congress values to strings and add the 0s back so that
    # files will be parsed better
    congresses = [f"{cong:0>3}" for cong in congresses]
    ngram_range = (min(args.ngram_range), max(args.ngram_range))

    LOGGER.info("Starting Processing...")
    # TODO: Confirm our tokenization defaults are reasonable -- should we lowercase?data
    # May need to map capitalized congresspeople (NANCY PELOSI) to standard spelling
    pipelines_to_disable = ["tagger", "parser", "ner"]
    
    try:
        nlp = spacy.load(args.spacy_model, disable=pipelines_to_disable)
        nlp.add_pipe(nlp.create_pipe("sentencizer"), first=True)
    except OSError:
        if args.spacy_model != "default":
            LOGGER.warning(
                f"Spacy model {args.spacy_model} not found, using rule-based tokenizer"
            )
        nlp = English(disable=pipelines_to_disable)
        nlp.add_pipe(nlp.create_pipe("sentencizer"), first=True)
        
    # lazily load speeches
    # (where "lazy" means session-by-session, and is still somewhat memory-intensive)
    speeches = SpeechesIterator(
        nlp,
        congresses,
        parties=["D", "R", "I"],
        data_directory=args.data_dir,
        min_sent_len=args.min_sent_len,
        min_sents_per_example=args.min_sents_per_example,
        ngram_range=ngram_range,
        use_noun_chunks=False,
        lowercase=args.lowercase,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    with open(output_dir / "full_speeches.jsonl", "w") as outfile:
        LOGGER.info("Parsing Speeches")
        for line in speeches:
            outfile.write(json.dumps(line) + "\n")
