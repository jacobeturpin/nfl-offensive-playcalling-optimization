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

    @abc.abstractmethod
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

    # noinspection PyIncorrectDocstring
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
        if play is PlayType.QB_SNEAK:
            data = self.df[self.df['qb_scramle'] == 1 and
                           self.df['play_type'] != 'no_play']
        elif play is PlayType.RUN:
            data = self.df[self.df['play_type'] == 'run']
        elif play is PlayType.PASS:
            data = self.df[self.df['play_type'] == 'pass']
        else:
            raise ValueError('PlayType not accepted for this class')

        def determine_outcome(row: pd.Series):
            if row['interception']:
                outcome = OutcomeType.INTERCEPTION
            elif row['fumble_lost']:
                outcome = OutcomeType.FUMBLE
            else:
                outcome = OutcomeType.BALL_MOVED
            return outcome.name

        data['outcome'] = data.apply(determine_outcome, axis=1)
        count = len(data)
        agg_cols = ['outcome', 'ydsnet']
        prob_data = data.groupby(agg_cols).size().to_frame('prob') / count
        prob_data.reset_index(inplace=True)
        return [PlayOutcome(*p) for p in prob_data.values]


class FieldPositionLoader(BaseLoader):

    def get_probability(self, down: int, to_go: int, position: int,
                        play: PlayType) -> List[PlayOutcome]:
        pass


if __name__ == '__main__':
    fn = './data/nfl-play-by-play.csv'
    dol = DownOnlyLoader(fn)

    print('Prob')
    outcomes = dol.get_probability(2, 8, PlayType.RUN)

    for o in outcomes:
        print(o)
