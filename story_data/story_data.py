import datasets
from typing import List
import pickle
import logging

logger = logging.getLogger()

_DESCRIPTION = 'StoryData'
_CITATION = 'copyright'

class StoryConfig(datasets.BuilderConfig):
    """BuilderConfig for Story."""

    def __init__(self, features, data_url, citation, url, label_classes=("False", "True"), **kwargs):
        """BuilderConfig for Story.

        Args:
        features: *list[string]*, list of the features that will appear in the
            feature dict. Should not include "label".
        data_url: *string*, url to download the zip file from.
        citation: *string*, citation for the data set.
        url: *string*, url for information about the data set.
        label_classes: *list[string]*, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
        **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 1.0.0: new
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url

class Story(datasets.GeneratorBasedBuilder):
    """The Story."""

    BUILDER_CONFIGS = [
        StoryConfig(
            name="original",
            description="Original",
            features=["narrative", "utterance", "response"],
            data_url="https://storyhelper.ewha.ac.kr",
            citation="StoryHelper",
            url="https://github.com/google-research-datasets/boolean-questions",
        ),
        StoryConfig(
            name="storyhelper",
            description="StoryHelper",
            features=["narrative", "utterance", "response"],
            #label_classes=["entailment", "not_entailment"],
            data_url="https://storyhelper.ewha.ac.kr",
            citation="StoryHelper",
            url="https://github.com/rudinger/winogender-schemas",
        ),
    ]
    _URL = "../data/"
    _URLS = {
        "train": _URL + "train.gr.pkl",
        "dev": _URL + "dev.gr.pkl",
    }

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls_to_download = self._URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, 'rb') as f:
            utterance, response, narrative, gt_response, y_true = pickle.load(f)
            #print(utterance.shape)
            #print(response.shape)

            for i in range(len(utterance)):
                id_ = i
                yield id_, {
                    "idx": id_,
                    "utterance": utterance[i],
                    "response": response[i],
                    "narrative": narrative[i],
                    "gt_response": gt_response[i],
                    "label": float(y_true[i]),
                }

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "utterance": datasets.Array2D(shape=(11,50), dtype='int32'),
                    "response": datasets.Sequence(datasets.Value("int32")),
                    "narrative": datasets.Sequence(datasets.Value("int32")),
                    "gt_response": datasets.Sequence(datasets.Value("int32")),
                    "label": datasets.Value("float32"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
        )
