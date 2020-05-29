"""Interface and classes for loading NFL playcalling data from Kaggle"""

import abc
from collections import namedtuple
from enum import Enum
from typing import List

import pandas as pd

PlayOutcome = namedtuple('PlayOutcome', ['type', 'yards', 'prob'])


class PlayType(Enum):
    PASS = 1
    RUN = 2
    QB_SNEAK = 3


class OutcomeType(Enum):
    BALL_MOVED = 1
    INTERCEPTION = 2
    FUMBLE = 3
    TOUCHDOWN = 4


class BaseLoader:
    """Abstract class defining interface for loading

    Attributes:
        filename (str): relative path to file containing data
    """

    def __init__(self, filename: str):
        self.fn = filename
        self.df = pd.read_csv(self.fn)

    @abc.abstractmethod()
    def get_probability(self, down: int, to_go: int, position: int,
                        play: PlayType) -> List[PlayOutcome]:
        raise NotImplementedError()


class DownOnlyLoader(BaseLoader):
    """Data loading class considering purely down and distance info

    Reads from a provided historical playcalling data and generates
    probability distributes for given state-action pairs.

    This particular class only considers the current down and the distance
    needed to make the next first down (e.g. 2nd and 5). Field position is
    NOT considered when making.

    Attributes:
        filename (str): relative path to file containing data
    """

    def __init__(self, filename: str):
        super().__init__(filename)

    def get_probability(self, down: int, to_go: int, play: PlayType,
                        position: int = None) -> List[PlayOutcome]:
        """Get the probability distribute for a given state-action pair

        Args:
            down (int): current down (1st through 4th)
            to_go (int): distance (in yards) to achieve first down
            play (PlayType): fgg

        Returns:
            A list of tuples specifying the possible outcomes and probabilities.
            For example:

            [
                ('BALL_MOVED', 5, 0.50),
                ('BALL_MOVED', 10, 0.25),
                ('INTERCEPTION', -5, 0.2),
                ('TOUCHDOWN', 25, 0.05)
            ]

        Raises:
            ValueError: An error when no data existing for the input combination
                provided
        """
        pass


class FieldPositionLoader(BaseLoader):

    def get_probability(self, down: int, to_go: int, position: int,
                        play: PlayType) -> List[PlayOutcome]:
        pass
