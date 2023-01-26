from typing import Union

from pandas._typing import FilePath, WriteBuffer


class SoccerTrackMixin(object):
    def save_dataframe(
        self,
        path_or_buf: Union[FilePath, WriteBuffer[bytes], WriteBuffer[str]],
    ) -> None:
        """Save a dataframe to a file.

        Args:
            df (pd.DataFrame): Dataframe to save.
            path_or_buf (FilePath | WriteBuffer[bytes] | WriteBuffer[str]): Path to save the dataframe.
        """
        assert self.columns.nlevels == 3, "Dataframe must have 3 levels of columns"
        if self.attrs:
            # write dataframe attributes to the csv file
            with open(path_or_buf, "w") as f:
                for k, v in self.attrs.items():
                    f.write(f"#{k}:{v}\n")
            self.to_csv(path_or_buf, mode="a")
        else:
            self.to_csv(path_or_buf, mode="w")

    def iter_frames(self, apply_func=None):
        """Iterate over the frames of the dataframe.

        Args:
            apply_func (function, optional): Function to apply to each group. Defaults to None.
        """
        if apply_func is None:
            apply_func = lambda x: x

        for index, group in self.groupby("frame"):
            yield index, apply_func(group)

    def iter_players(self, apply_func=None, drop=True):
        """Iterate over the players of the dataframe.

        Args:
            apply_func (function, optional): Function to apply to each group. Defaults to None.
            drop (bool, optional): Drop the level of the dataframe. Defaults to True.
        """
        if apply_func is None:
            apply_func = lambda x: x
        for index, group in self.groupby(level=("TeamID", "PlayerID"), axis=1):
            if drop:
                yield index, apply_func(
                    group.droplevel(level=("TeamID", "PlayerID"), axis=1)
                )
            else:
                yield index, apply_func(group)

    def iter_teams(self, apply_func=None, drop=True):
        """Iterate over the teams of the dataframe.

        Args:
            apply_func (function, optional): Function to apply to each group. Defaults to None.
            drop (bool, optional): Drop the level of the dataframe. Defaults to True.
        """
        if apply_func is None:
            apply_func = lambda x: x
        for index, group in self.groupby(level="TeamID", axis=1):
            if drop:
                yield index, apply_func(group.droplevel(level=("TeamID"), axis=1))
            else:
                yield index, apply_func(group)

    def iter_attributes(self, apply_func=None, drop=True):
        """Iterate over the attributes of the dataframe.

        Args:
            apply_func (function, optional): Function to apply to each group. Defaults to None.
            drop (bool, optional): Drop the level of the dataframe. Defaults to True.
        """
        if apply_func is None:
            apply_func = lambda x: x
        for index, group in self.groupby(level="Attributes", axis=1):
            if drop:
                yield index, apply_func(group.droplevel(level=("Attributes"), axis=1))
            else:
                yield index, group

    def to_long_df(self, level="Attributes", dropna=True):
        """Convert a dataframe to a long format.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            level (str, optional): Level to convert to long format. Defaults to 'Attributes'. Options are 'Attributes', 'TeamID', 'PlayerID'.

        Returns:
            pd.DataFrame: Dataframe in long format.
        """
        df = self.copy()

        levels = ["TeamID", "PlayerID", "Attributes"]
        levels.remove(level)

        df = df.stack(level=levels, dropna=dropna)
        return df

    def is_long_format(self):
        """Check if the dataframe is in long format.

        Returns:
            bool: True if the dataframe is in long format, False otherwise.
        """
        return self.index.nlevels == 3

    def get_frame(self, frame):
        """Get a specific frame from the dataframe.

        Args:
            frame (int): Frame to get.

        Returns:
            pd.DataFrame: Dataframe with the frame.
        """
        if self.is_long_format():
            return self.xs(frame, level="frame")
        return self[self.index == frame]
