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
    FIELD_GOAL = 4
    PUNT = 5


class OutcomeType(Enum):
    BALL_MOVED = 1
    INTERCEPTION = 2
    FUMBLE = 3
    TOUCHDOWN = 4
    FIELD_GOAL_MADE = 5
    FIELD_GOAL_MISSED = 6


class BaseLoader:
    """Abstract class defining interface for loading

    Attributes:
        filename (str): relative path to file containing data
    """

    def __init__(self, filename: str, dtypes_dict: dict = None):
        self.fn = filename
        self.df = pd.read_csv(self.fn, dtype = dtypes_dict)

        self.df_indices = {}
        self.df_indices['fg'] = self.df.loc[self.df['play_type'] == 'field_goal'].index
        self.df_indices['punt'] = self.df.loc[self.df['play_type'] == 'punt'].index
        self.df_indices['qb_sneak'] = self.df.loc[(self.df['qb_scramble'] == 1) & (self.df['play_type'] != 'no_play')].index
        self.df_indices['run'] = self.df[self.df['play_type'] == 'run'].index
        self.df_indices['pass'] = self.df[self.df['play_type'] == 'pass'].index

        def __yardline_correction(row: pd.Series) -> int:
            """Transforms 0-100 scale yardline data to be direction of possession team

            The default loaded data is based on cardinal directionality (e.g. North-South or
            East-West). The purpose of this function is to transform this data such that
            the output represent how far the possession team must travel to the goal line
            (e.g. )

            Example:
                dataframe.apply(yardline_correction, axis=1)

                OWN 25 (yardline_100 = 25) -> yardline_100 = 75

            Args:
                row (pd.Series): single row of play-by-play data
            Returns:
                int: corrected 0-100 scale yardline distance based on distance from score
            """

            if row['posteam'] == row['side_of_field'] and row['yardline_100'] < 50:
                return row['yardline_100'] + 2 * (50 - row['yardline_100'])
            if row['posteam'] != row['side_of_field'] and row['yardline_100'] > 50:
                return row['yardline_100'] - 2 * (row['yardline'] - 50)
            return row['yardline_100']
        self.df.loc[:,'fieldpos'] = self.df.apply(__yardline_correction, axis=1)

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
            data = self.df.loc[(self.df['qb_scramble'] == 1) & (self.df['play_type'] != 'no_play')]
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
        agg_cols = ['outcome', 'yards_gained']
        prob_data = data.groupby(agg_cols).size().to_frame('prob') / count
        prob_data.reset_index(inplace=True)
        return [PlayOutcome(*p) for p in prob_data.values]


class FieldPositionLoader(BaseLoader):

    def __init__(self, filename: str, dtypes_dict: dict = None):
        super().__init__(filename, dtypes_dict)

    # @staticmethod
    # def __yardline_correction(row: pd.Series) -> int:
    #     """Transforms 0-100 scale yardline data to be direction of possession team

    #     The default loaded data is based on cardinal directionality (e.g. North-South or
    #     East-West). The purpose of this function is to transform this data such that
    #     the output represent how far the possession team must travel to the goal line
    #     (e.g. )

    #     Example:
    #         dataframe.apply(yardline_correction, axis=1)

    #         OWN 25 (yardline_100 = 25) -> yardline_100 = 75

    #     Args:
    #         row (pd.Series): single row of play-by-play data
    #     Returns:
    #         int: corrected 0-100 scale yardline distance based on distance from score
    #     """

    #     if row['posteam'] == row['side_of_field'] and row['yardline_100'] < 50:
    #         return row['yardline_100'] + 2 * (50 - row['yardline_100'])
    #     if row['posteam'] != row['side_of_field'] and row['yardline_100'] > 50:
    #         return row['yardline_100'] - 2 * (row['yardline'] - 50)
    #     return row['yardline_100']

    def get_probability(self, down: int, to_go: int, position: int,
                        play: PlayType) -> List[PlayOutcome]:

        if play is PlayType.FIELD_GOAL:
            data = self.df.iloc[self.df_indices['fg'], :]
        elif play is PlayType.PUNT:
            data = self.df.iloc[self.df_indices['punt'], :]
        elif play is PlayType.QB_SNEAK:
            data = self.df.iloc[self.df_indices['qb_sneak'], :]
        elif play is PlayType.RUN:
            data = self.df.iloc[self.df_indices['run'], :]
        elif play is PlayType.PASS:
            data = self.df.iloc[self.df_indices['pass'], :]
        else:
            raise ValueError('PlayType not accepted for this class')

        # self.df.loc[:,'fieldpos'] = self.df.apply(FieldPositionLoader.__yardline_correction, axis=1)
        data = data.loc[data['fieldpos'] == position]

        def determine_outcome(row: pd.Series):

            if play is PlayType.FIELD_GOAL:
                outcome = OutcomeType.FIELD_GOAL_MADE if row['field_goal_result'] == 'made' \
                    else OutcomeType.FIELD_GOAL_MISSED
            elif row['interception']:
                outcome = OutcomeType.INTERCEPTION
            elif row['fumble_lost']:
                outcome = OutcomeType.FUMBLE
            elif position == row['yards_gained']:
                outcome = OutcomeType.TOUCHDOWN
            else:
                outcome = OutcomeType.BALL_MOVED
            return outcome.name

        count = len(data)
        
        # check that there is data for the field position in question
        if count == 0 and play is PlayType.FIELD_GOAL:
          return [PlayOutcome('FIELD_GOAL_MISSED',min(-10, position - 10),1)]
        elif count == 0:
          return [PlayOutcome('BALL_MOVED',0,1)]

        else:
          data.loc[:,'outcome'] = data.apply(determine_outcome, axis=1).copy(deep=True)

          if play is PlayType.FIELD_GOAL:
              data.loc['yards_gained'] = data.apply(
                  lambda r: position if r['outcome'] == 'FIELD_GOAL_MADE' else min(-10, position - 10),
                  axis=1)
          if play is PlayType.PUNT:
              data['yards_gained'] = data.apply(
                  lambda r: r['kick_distance'] - r['punt_in_endzone'] if not r['punt_in_endzone'] else position - 20,
                  axis=1)
          agg_cols = ['outcome', 'yards_gained']
          prob_data = data.groupby(agg_cols).size().to_frame('prob') / count
          prob_data.reset_index(inplace=True)
          return [PlayOutcome(*p) for p in prob_data.values]


if __name__ == '__main__':
    fn = './data/nfl-play-by-play.csv'


    # dol = DownOnlyLoader(fn)
    #
    # print('Prob')
    # outcomes = dol.get_probability(2, 8, PlayType.RUN)
    # for o in outcomes:
    #     print(o)

    fol = FieldPositionLoader(fn)

    #region offensive plays

    print('Prob')
    outcomes = fol.get_probability(2, 8, 80, PlayType.PASS)
    for o in outcomes:
        print(o)
    print('\n\n')

    #endregion

    #region field goal outcomes

    outcomes = fol.get_probability(4, 8, 15, PlayType.FIELD_GOAL)
    for o in outcomes:
        print(o)
    print('\n\n')

    outcomes = fol.get_probability(4, 8, 20, PlayType.FIELD_GOAL)
    for o in outcomes:
        print(o)
    print('\n\n')

    outcomes = fol.get_probability(4, 8, 25, PlayType.FIELD_GOAL)
    for o in outcomes:
        print(o)
    print('\n\n')

    #endregion


    #region punting

    outcomes = fol.get_probability(4, 8, 80, PlayType.PUNT)
    for o in outcomes:
        print(o)
    print('\n\n')

    outcomes = fol.get_probability(4, 8, 40, PlayType.PUNT)
    for o in outcomes:
        print(o)
    print('\n\n')

    #endregion
