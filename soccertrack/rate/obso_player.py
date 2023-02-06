import math
import re
from datetime import time

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from soccertrack.rate import Metrica_IO as mio
from soccertrack.rate import Metrica_PitchControl as mpc
from soccertrack.rate import Metrica_Velocities as mvel


def calc_obso(PPCF, Transition, Score, tracking, attack_direction=0):
    # calculate obso in single frame
    # PPCF, Score : 50 * 32
    # Transition : 100 * 64
    Transition = np.array((Transition))
    Score = np.array((Score))
    ball_grid_x = int((tracking["ball_x"] + 52.5) // (105 / 50))
    ball_grid_y = int((tracking["ball_y"] + 34) // (68 / 32))

    # When out of the pitch
    if ball_grid_x < 0:
        ball_grid_x = 0
    elif ball_grid_x > 49:
        ball_grid_x = 49
    if ball_grid_y < 0:
        ball_grid_y = 0
    elif ball_grid_y > 31:
        ball_grid_y = 31

    Transition = Transition[
        31 - ball_grid_y : 63 - ball_grid_y, 49 - ball_grid_x : 99 - ball_grid_x
    ]

    if attack_direction < 0:
        Score = np.fliplr(Score)
    elif attack_direction > 0:
        Score = Score
    else:
        print("input attack direction is 1 or -1")

    obso = PPCF * Transition * Score

    return obso, Transition


def calc_player_evaluate(player_pos, evaluation):
    # player_pos:(x, y) col
    # evaluation : evaluation grid (32 * 50)

    # grid size
    grid_size_x = 105 / 50
    grid_size_y = 68 / 32

    player_grid_x = (player_pos[0] + 52.5) // grid_size_x
    player_grid_y = (player_pos[1] + 34) // grid_size_y

    # When out of the pitch
    if player_grid_x < 0:
        player_grid_x = 0
    elif player_grid_x > 49:
        player_grid_x = 49
    if player_grid_y < 0:
        player_grid_y = 0
    elif player_grid_y > 31:
        player_grid_y = 31

    # data format int in grid number
    player_grid_x = int(player_grid_x)
    player_grid_y = int(player_grid_y)

    # be careful for index number (y cordinate, x cordinate)
    player_ev = evaluation[player_grid_y, player_grid_x]

    return player_ev


def calc_player_evaluate_match(OBSO, tracking_home, tracking_away):
    # calculate player evaluation at event
    # input:obso(grid evaluation), events(event data in Metrica format), tracking home and away (tracking data)
    # return home_obso, away_obso(player evaluation at event)

    # set DataFrame column name
    column_name = ["frame"]
    home_column = tracking_home.columns
    home_player_num = [s[:-2] for s in home_column if re.match("Home_\d*_x", s)]
    home_column_name = column_name + home_player_num
    away_column = tracking_away.columns
    away_player_num = [s[:-2] for s in away_column if re.match("Away_\d*_x", s)]
    away_column_name = column_name + away_player_num
    home_index = list(range(len(tracking_home)))
    away_index = list(range(len(tracking_away)))
    home_obso = pd.DataFrame(columns=home_column_name, index=home_index)
    away_obso = pd.DataFrame(columns=away_column_name, index=away_index)

    # initialize event number in home and away
    home_event_num = 0
    away_event_num = 0
    for frame in tqdm(range(len(tracking_home))):
        home_event_num += 1
        home_obso["frame"].iloc[home_event_num - 1] = frame
        for player in home_player_num:
            home_player_pos = [
                tracking_home[player + "_x"].iloc[frame],
                tracking_home[player + "_y"].iloc[frame],
            ]
            home_obso[player].iloc[home_event_num - 1] = calc_player_evaluate(
                home_player_pos, OBSO[frame]
            )

        away_event_num += 1
        away_obso["frame"].iloc[away_event_num - 1] = frame
        for player in away_player_num:
            away_player_pos = [
                tracking_away[player + "_x"].iloc[frame],
                tracking_away[player + "_y"].iloc[frame],
            ]
            away_obso[player].iloc[away_event_num - 1] = calc_player_evaluate(
                away_player_pos, OBSO[frame]
            )

    return home_obso, away_obso


def calc_onball_obso(events, tracking_home, tracking_away, home_obso, away_obso):
    # calculate on-ball obso because obso is not defined in on-ball
    # input : event data in format Metrica
    # output : home_onball_obso and away_onball_obso in format pandas dataframe

    # set dataframe column name
    home_name = home_obso.columns[2:]
    away_name = away_obso.columns[2:]

    # set output dataframe
    home_onball_obso = pd.DataFrame(
        columns=home_obso.columns, index=list(range(len(home_obso)))
    )
    away_onball_obso = pd.DataFrame(
        columns=away_obso.columns, index=list(range(len(away_obso)))
    )

    # initialize event number in home and away
    home_event_num = 0
    away_event_num = 0

    # search on ball player
    for num, frame in enumerate(tqdm(events["Start Frame"])):
        if events["Team"].iloc[num] == "Home":
            home_event_num += 1
            dis_dict = {}
            home_onball_obso["event_frame"].iloc[home_event_num - 1] = frame
            home_onball_obso["event_number"].iloc[home_event_num - 1] = num
            for name in home_name:
                if np.isnan(tracking_home[name + "_x"].iloc[frame]):
                    continue
                else:
                    # initialize distance in format dictionary
                    player_pos = np.array(
                        [
                            tracking_home[name + "_x"].iloc[frame],
                            tracking_home[name + "_y"].iloc[frame],
                        ]
                    )
                    ball_pos = np.array(
                        [
                            tracking_home["ball_x"].iloc[frame],
                            tracking_home["ball_y"].iloc[frame],
                        ]
                    )
                    ball_dis = np.linalg.norm(player_pos - ball_pos)
                    dis_dict[name] = ball_dis
            # home onball player, that is the nearest player to the ball
            onball_player = min(dis_dict, key=dis_dict.get)
            home_onball_obso[onball_player].iloc[home_event_num - 1] = home_obso[
                onball_player
            ].iloc[home_event_num - 1]
        elif events["Team"].iloc[num] == "Away":
            away_event_num += 1
            dis_dict = {}
            away_onball_obso["event_frame"].iloc[away_event_num - 1] = frame
            away_onball_obso["event_number"].iloc[away_event_num - 1] = num
            for name in away_name:
                if np.isnan(tracking_away[name + "_x"].iloc[frame]):
                    continue
                else:
                    # initialize distance in format dictionary
                    player_pos = np.array(
                        [
                            tracking_away[name + "_x"].iloc[frame],
                            tracking_away[name + "_y"].iloc[frame],
                        ]
                    )
                    ball_pos = np.array(
                        [
                            tracking_away["ball_x"].iloc[frame],
                            tracking_away["ball_y"].iloc[frame],
                        ]
                    )
                    ball_dis = np.linalg.norm(player_pos - ball_pos)
                    dis_dict[name] = ball_dis
            # away onball player, that is the nearest player to the ball
            onball_player = min(dis_dict, key=dis_dict.get)
            away_onball_obso[onball_player].iloc[away_event_num - 1] = away_obso[
                onball_player
            ].iloc[away_event_num - 1]
        else:
            continue

    return home_onball_obso, away_onball_obso


def convert_Metrica_for_event(event_df):
    # convert eventdata (from spadl to Metrica)
    # event_df : event data in spadl format

    # set column name
    column_name = [
        "Team",
        "Type",
        "Subtype",
        "Period",
        "Start Frame",
        "Start Time [s]",
        "End Frame",
        "End Time [s]",
        "From",
        "To",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
    ]
    Metrica_df = pd.DataFrame(columns=column_name)
    Metrica_df["Period"] = event_df["period_id"]
    Metrica_df["Start X"] = event_df["start_x"] - 52.5
    Metrica_df["Start Y"] = event_df["start_y"] - 34
    Metrica_df["End X"] = event_df["end_x"] - 52.5
    Metrica_df["End Y"] = event_df["end_y"] - 34
    Metrica_df["From"] = event_df["player_name"]
    Metrica_df["Type"] = event_df["type_name"]
    Metrica_df["Subtype"] = event_df["result_name"]

    first_period_time = event_df["time_seconds"][event_df["period_id"] == 1]
    first_endtime = max(first_period_time)
    second_period_time = (
        event_df["time_seconds"][event_df["period_id"] == 2] + first_endtime
    )
    start_time = pd.concat([first_period_time, second_period_time])
    end_time = start_time.shift(-1)
    start_frame = event_df["start_frame"]
    end_frame = start_frame.shift(-1)

    Metrica_df["Start Time [s]"] = start_time
    Metrica_df["Start Frame"] = start_frame
    Metrica_df["End Time [s]"] = end_time
    Metrica_df["End Frame"] = end_frame

    Team_list = event_df["team_id"]
    team_id_uni = sorted(event_df["team_id"].unique())

    for i in range(len(Team_list)):
        if Team_list[i] == team_id_uni[1]:
            Team_list[i] = "Home"
        elif Team_list[i] == team_id_uni[2]:
            Team_list[i] = "Away"

    Metrica_df["Team"] = Team_list

    return Metrica_df


def check_home_away_event(events, tracking_home, tracking_away):
    # check wether corresponded event data and tracking data defined as 'Home' or 'Away'
    # input : events in format Metrica, tracking data in format Metrica

    # search nearest player home and away
    # set player name (ex. Home_1, ...)
    home_column = tracking_home.columns
    away_column = tracking_away.columns
    home_name = [s[:-2] for s in home_column if re.match("Home_\d*_x", s)]
    away_name = [s[:-2] for s in away_column if re.match("Away_\d*_x", s)]

    # calculate distace player to ball
    # set home distance
    home_dis = []
    for player in home_name:
        # Exception handling for no entry player
        if np.isnan(tracking_home[player + "_x"].iloc[0]):
            continue
        else:
            ball_pos = np.array(
                [tracking_home["ball_x"].iloc[0], tracking_home["ball_y"].iloc[0]]
            )
            player_pos = np.array(
                [
                    tracking_home[player + "_x"].iloc[0],
                    tracking_home[player + "_y"].iloc[0],
                ]
            )
            home_dis.append(np.linalg.norm(player_pos - ball_pos))

    # set away distance
    away_dis = []
    for player in away_name:
        # Exception handling for no entry player
        if np.isnan(tracking_away[player + "_x"].iloc[0]):
            continue
        else:
            ball_pos = np.array(
                [tracking_away["ball_x"].iloc[0], tracking_away["ball_y"].iloc[0]]
            )
            player_pos = np.array(
                [
                    tracking_away[player + "_x"].iloc[0],
                    tracking_away[player + "_y"].iloc[0],
                ]
            )
            away_dis.append(np.linalg.norm(player_pos - ball_pos))

    # judge kick-off team
    if min(home_dis) < min(away_dis):
        kickoff_team = "Home"
    else:
        kickoff_team = "Away"
    # print('kickoff:{}'.format(kickoff_team))
    # check team in events
    for i in range(len(events[events["Start Frame"] == 0])):
        if events.loc[i]["Team"] != "Home" and events.loc[i]["Team"] != "Away":
            continue
        elif kickoff_team != events.loc[i]["Team"]:
            # replace 'Home' to 'Away' and 'Away' to 'Home'
            events = events.replace({"Team": {"Home": "Away", "Away": "Home"}})
            # print('change team name')
            break
    return events


def set_trackingdata(tracking_home, tracking_away):
    # data preprocessing tracking data
    # input : tarcking data (x, y) position data
    tracking_home = tracking_home.drop(columns="Unnamed: 0")
    tracking_away = tracking_away.drop(columns="Unnamed: 0")

    # preprocessing player position
    entry_home_df = tracking_home.iloc[0].isnull()
    entry_away_df = tracking_away.iloc[0].isnull()
    home_column = tracking_home.columns
    away_column = tracking_away.columns
    home_player_num = [s[:-2] for s in home_column if re.match("Home_\d*_x", s)]
    away_player_num = [s[:-2] for s in away_column if re.match("Away_\d*_x", s)]

    # replace nan
    for player in home_player_num:
        if entry_home_df[player + "_x"]:
            tracking_home[player + "_x"] = tracking_home[player + "_x"].fillna(
                method="ffill"
            )
            tracking_home[player + "_y"] = tracking_home[player + "_y"].fillna(
                method="ffill"
            )
        else:
            tracking_home[player + "_x"] = tracking_home[player + "_x"].fillna(
                method="bfill"
            )
            tracking_home[player + "_y"] = tracking_home[player + "_y"].fillna(
                method="bfill"
            )

    for player in away_player_num:
        if entry_away_df[player + "_x"]:
            tracking_away[player + "_x"] = tracking_away[player + "_x"].fillna(
                method="ffill"
            )
            tracking_away[player + "_y"] = tracking_away[player + "_y"].fillna(
                method="ffill"
            )
        else:
            tracking_away[player + "_x"] = tracking_away[player + "_x"].fillna(
                method="bfill"
            )
            tracking_away[player + "_y"] = tracking_away[player + "_y"].fillna(
                method="bfill"
            )

    # data interpolation in ball position in tracking data
    tracking_home["ball_x"] = tracking_home["ball_x"].interpolate()
    tracking_home["ball_y"] = tracking_home["ball_y"].interpolate()
    tracking_away["ball_x"] = tracking_away["ball_x"].interpolate()
    tracking_away["ball_y"] = tracking_away["ball_y"].interpolate()

    # check nan ball position x and y in tracking data
    tracking_home["ball_x"] = tracking_home["ball_x"].fillna(method="bfill")
    tracking_home["ball_y"] = tracking_home["ball_y"].fillna(method="bfill")
    tracking_away["ball_x"] = tracking_away["ball_x"].fillna(method="bfill")
    tracking_away["ball_y"] = tracking_away["ball_y"].fillna(method="bfill")

    # filter:Savitzky-Golay
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    return tracking_home, tracking_away


def remove_offside_obso(events, tracking_home, tracking_away, home_obso, away_obso):
    # remove obso value(to 0) for offise player
    # events:event data (Metrica format), tracking home and away:tracking data (Metrica foramat)
    # obso (home and away): obso value in each event

    # set parameters for calculating PPCF
    params = mpc.default_model_params()
    GK_numbers = [
        mio.find_goalkeeper(tracking_home),
        mio.find_goalkeeper(tracking_away),
    ]
    # set player name
    home_name = home_obso.columns[2:]
    away_name = away_obso.columns[2:]
    # search offside player
    for event_id in tqdm(range(len(events))):
        # check event team home or away
        if events["Team"].iloc[event_id] == "Home":
            _, _, _, attacking_players = mpc.generate_pitch_control_for_event(
                event_id, events, tracking_home, tracking_away, params, GK_numbers
            )
            attacking_players_name = [p.playername[:-1] for p in attacking_players]
            off_players = home_name ^ attacking_players_name
            for name in off_players:
                home_obso[name][home_obso["event_number"] == event_id] = 0

        elif events["Team"].iloc[event_id] == "Away":
            _, _, _, attacking_players = mpc.generate_pitch_control_for_event(
                event_id, events, tracking_home, tracking_away, params, GK_numbers
            )
            attacking_players_name = [p.playername[:-1] for p in attacking_players]
            off_players = away_name ^ attacking_players_name
            for name in off_players:
                away_obso[name][away_obso["event_number"] == event_id] = 0
        else:
            continue

    return home_obso, away_obso


def check_event_zone(events, tracking_home, tracking_away):
    # check event zone
    # input:event data format Metrica
    # output:evevnt at attackind third, middle zone, defensive third
    # zone is based on -52.5~-17.5, -17.5~+17.5, 17.5~52.5

    # set zone series format pandas Series
    zone_se = pd.DataFrame(columns=["zone"], index=events.index)
    # check attack direction
    for event_num in range(len(events)):
        if events.iloc[event_num]["Period"] == 1:
            if events.iloc[event_num]["Team"] == "Home":
                direction = mio.find_playing_direction(
                    tracking_home[tracking_home["Period"] == 1], "Home"
                )
            elif events.iloc[event_num]["Team"] == "Away":
                direction = mio.find_playing_direction(
                    tracking_away[tracking_away["Period"] == 1], "Away"
                )
            else:
                direction = 0
        elif events.iloc[event_num]["Period"] == 2:
            if events.iloc[event_num]["Team"] == "Home":
                direction = mio.find_playing_direction(
                    tracking_home[tracking_home["Period"] == 2], "Home"
                )
            elif events.iloc[event_num]["Team"] == "Away":
                direction = mio.find_playing_direction(
                    tracking_away[tracking_away["Period"] == 2], "Away"
                )
            else:
                direction = 0
        # add zone defense or middle or attack
        if direction > 0:
            if events.iloc[event_num]["Start X"] < -17.5:
                zone_se.iloc[event_num]["zone"] = "defense"
            elif events.iloc[event_num]["Start X"] > 17.5:
                zone_se.iloc[event_num]["zone"] = "attack"
            else:
                zone_se.iloc[event_num]["zone"] = "middle"
        elif direction < 0:
            if events.iloc[event_num]["Start X"] < -17.5:
                zone_se.iloc[event_num]["zone"] = "attack"
            elif events.iloc[event_num]["Start X"] > 17.5:
                zone_se.iloc[event_num]["zone"] = "defense"
            else:
                zone_se.iloc[event_num]["zone"] = "middle"
        else:
            zone_se.iloc[event_num]["zone"] = 0

    return zone_se


def mark_check(
    tracking_home, tracking_away, tracking_frame, attacking_team, player_num=10
):
    # define mark player in defense team
    mark_df = pd.DataFrame(columns=["Attack", "Defense"])
    # calculate distance ball to player in attack team
    if attacking_team == "Home":
        # calculate distance ball to player in attack team
        home_dis_df = pd.DataFrame(columns=["number", "distance", "x_col", "y_col"])
        ball_pos = np.array(
            [
                tracking_home.iloc[tracking_frame]["ball_x"],
                tracking_home.iloc[tracking_frame]["ball_y"],
            ]
        )
        for num in range(1, 15):
            # skip non-participating player
            if (
                np.isnan(tracking_home.iloc[tracking_frame]["Home_{}_x".format(num)])
                == True
            ):
                continue
            # set position of participating player
            player_pos = np.array(
                [
                    tracking_home.iloc[tracking_frame]["Home_{}_x".format(num)],
                    tracking_home.iloc[tracking_frame]["Home_{}_y".format(num)],
                ]
            )
            # calculate distance attack player to ball
            dis = np.linalg.norm((player_pos - ball_pos))
            home_dis_df = home_dis_df.append(
                {
                    "number": "Home_{}".format(num),
                    "distance": dis,
                    "x_col": player_pos[0],
                    "y_col": player_pos[1],
                },
                ignore_index=True,
            )
        # sort by closest to ball
        home_dis_df = home_dis_df.sort_values("distance").reset_index()
        home_dis_df = home_dis_df.iloc[:player_num]
        # define mark player in defense team
        mark_df["Attack"] = home_dis_df["number"]
        defense_pos = pd.DataFrame(columns=["number", "x_col", "y_col"])
        # set position of defense player
        for num in range(1, 15):
            if (
                np.isnan(tracking_away.iloc[tracking_frame]["Away_{}_x".format(num)])
                == True
            ):
                continue
            defense_pos = defense_pos.append(
                {
                    "number": "Away_{}".format(num),
                    "x_col": tracking_away.iloc[tracking_frame][
                        "Away_{}_x".format(num)
                    ],
                    "y_col": tracking_away.iloc[tracking_frame][
                        "Away_{}_y".format(num)
                    ],
                },
                ignore_index=True,
            )
        # calculate distance defense player to attack player
        for att in range(player_num):
            att_pos = np.array(
                [home_dis_df.iloc[att]["x_col"], home_dis_df.iloc[att]["y_col"]]
            )
            att_dis = []
            for df in range(len(defense_pos)):
                df_pos = np.array(
                    [defense_pos.iloc[df]["x_col"], defense_pos.iloc[df]["y_col"]]
                )
                dis = np.linalg.norm((att_pos - df_pos))
                att_dis.append(dis)
            defense_pos["{}".format(home_dis_df.iloc[att]["number"])] = att_dis
        # check defense player who is closet to attack player
        for num in range(player_num):
            min_index = defense_pos[mark_df.iloc[num]["Attack"]].idxmin()
            mark_df.iloc[num]["Defense"] = defense_pos.loc[min_index]["number"]
            defense_pos = defense_pos.drop(min_index)
    # calculate distance ball to player in attack team
    elif attacking_team == "Away":
        # calculate distance ball to player in attack team
        away_dis_df = pd.DataFrame(columns=["number", "distance", "x_col", "y_col"])
        ball_pos = np.array(
            [
                tracking_away.iloc[tracking_frame]["ball_x"],
                tracking_away.iloc[tracking_frame]["ball_y"],
            ]
        )
        for num in range(1, 15):
            # skip non-participating player
            if (
                np.isnan(tracking_away.iloc[tracking_frame]["Away_{}_x".format(num)])
                == True
            ):
                continue
            # set position of participating player
            player_pos = np.array(
                [
                    tracking_away.iloc[tracking_frame]["Away_{}_x".format(num)],
                    tracking_away.iloc[tracking_frame]["Away_{}_y".format(num)],
                ]
            )
            # calculate distance attack player to ball
            dis = np.linalg.norm((player_pos - ball_pos))
            away_dis_df = away_dis_df.append(
                {
                    "number": "Away_{}".format(num),
                    "distance": dis,
                    "x_col": player_pos[0],
                    "y_col": player_pos[1],
                },
                ignore_index=True,
            )
        # sort by closet to ball
        away_dis_df = away_dis_df.sort_values("distance").reset_index()
        away_dis_df = away_dis_df.iloc[:player_num]
        # define mark player in defense team
        mark_df["Attack"] = away_dis_df["number"]
        defense_pos = pd.DataFrame(columns=["number", "x_col", "y_col"])
        # set position of defense player
        for num in range(1, 15):
            if (
                np.isnan(tracking_home.iloc[tracking_frame]["Home_{}_x".format(num)])
                == True
            ):
                continue
            defense_pos = defense_pos.append(
                {
                    "number": "Home_{}".format(num),
                    "x_col": tracking_home.iloc[tracking_frame][
                        "Home_{}_x".format(num)
                    ],
                    "y_col": tracking_home.iloc[tracking_frame][
                        "Home_{}_y".format(num)
                    ],
                },
                ignore_index=True,
            )
        # calculate distance defense player to attack player
        for att in range(player_num):
            att_pos = np.array(
                [away_dis_df.iloc[att]["x_col"], away_dis_df.iloc[att]["y_col"]]
            )
            att_dis = []
            for df in range(len(defense_pos)):
                df_pos = np.array(
                    [defense_pos.iloc[df]["x_col"], defense_pos.iloc[df]["y_col"]]
                )
                dis = np.linalg.norm((att_pos - df_pos))
                att_dis.append(dis)
            defense_pos["{}".format(away_dis_df.iloc[att]["number"])] = att_dis
        # check defense player who is closet to attack player
        for num in range(player_num):
            min_index = defense_pos[mark_df.iloc[num]["Attack"]].idxmin()
            mark_df.iloc[num]["Defense"] = defense_pos.loc[min_index]["number"]
            defense_pos = defense_pos.drop(min_index)

    return mark_df


def extract_shotseq(event_data):
    # this function is extract shot sequence
    # input : event data
    # output : shot dataframe(Team, shot event number and frame, start event number and frame)
    # get shot event
    shot_event = event_data[event_data["Type"] == "shot"]
    shot_event_num = list(shot_event.index)
    # set shot dataframe
    shot_df = pd.DataFrame(
        columns=[
            "Team",
            "shot_event",
            "start_event",
            "start_frame",
            "end_frame",
            "frame_length",
            "time_length[s]",
            "result",
        ]
    )
    # get start event
    start_event_num = []
    for num in shot_event_num:
        shot_Team = event_data.loc[num]["Team"]
        pre_Team = event_data.loc[num]["Team"]
        tmp = num
        while shot_Team == pre_Team:
            tmp = tmp - 1
            pre_Team = event_data.loc[tmp]["Team"]
        start_event_num.append(tmp + 1)

    # set shot_df
    shot_df["Team"] = shot_event["Team"]
    shot_df["result"] = shot_event["Subtype"]
    shot_df["shot_event"] = shot_event_num
    shot_df["start_event"] = start_event_num
    shot_df = shot_df.reset_index(drop=True)
    # search start frame
    for i in range(len(shot_event_num)):
        shot_df["start_frame"].loc[i] = event_data["Start Frame"].loc[
            start_event_num[i]
        ]
        shot_df["end_frame"].loc[i] = event_data["Start Frame"].loc[shot_event_num[i]]
        shot_df["frame_length"].loc[i] = (
            shot_df["end_frame"].loc[i] - shot_df["start_frame"].loc[i]
        )
        shot_df["time_length[s]"].loc[i] = shot_df["frame_length"].loc[i] / 25

    return shot_df


def calc_shot_obso(
    shot_df,
    event_data,
    tracking_home,
    tracking_away,
    jursey_data,
    player_data,
    Trans,
    EPV,
):
    # this function is calcurating shot obso, add shot_obso to shot_df
    # set parameter
    # set OBSO list
    OBSO_list = []
    params = mpc.default_model_params()
    GK_numbers = [
        mio.find_goalkeeper(tracking_home),
        mio.find_goalkeeper(tracking_away),
    ]
    # add columns (shot_obso)
    shot_df["shot_obso"] = 0
    shot_df["shot_player"] = "Nan"
    # calculate obso in shot sequences
    for idx in range(len(shot_df)):
        ev_frame = shot_df.loc[idx]["end_frame"] - 1
        attacking_team = event_data.loc[shot_df.loc[idx]["shot_event"]]["Team"]
        # check GK_numbers
        if np.isnan(tracking_home.loc[ev_frame]["Home_" + GK_numbers[0] + "_x"]):
            GK_numbers[0] = "12"
        if np.isnan(tracking_away.loc[ev_frame]["Away_" + GK_numbers[1] + "_x"]):
            GK_numbers[1] = "12"
        PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
            tracking_home, tracking_away, ev_frame, attacking_team, params, GK_numbers
        )
        # check attacking direction
        if attacking_team == "Home":
            if event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(
                    tracking_home[tracking_home["Period"] == 1], "Home"
                )
            elif event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direction = mio.find_playing_direction(
                    tracking_home[tracking_home["Period"] == 2], "Home"
                )
        elif attacking_team == "Away":
            if event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(
                    tracking_away[tracking_away["Period"] == 1], "Away"
                )
            elif event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direciton = mio.find_playing_direction(
                    tracking_away[tracking_away["Period"] == 2], "Away"
                )
        OBSO, _ = calc_obso(
            PPCF, Trans, EPV, tracking_home.loc[ev_frame], attack_direction=direction
        )
        OBSO_list.append(OBSO)
        # search shot player
        if attacking_team == "Home":
            shot_player_name = event_data.loc[shot_df.loc[idx]["shot_event"]]["From"]
            jursey_num = int(player_data[player_data["選手名"] == shot_player_name]["背番号"])
            track_num = jursey_data[jursey_data["Home"] == jursey_num].iloc[0, 0]
            shot_player = "Home_" + str(track_num)
            shot_player_x = tracking_home.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_home.loc[ev_frame][shot_player + "_y"]
            shot_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        elif attacking_team == "Away":
            shot_player_name = event_data.loc[shot_df.loc[idx]["shot_event"]]["From"]
            jursey_num = int(player_data[player_data["選手名"] == shot_player_name]["背番号"])
            track_num = jursey_data[jursey_data["Away"] == jursey_num].iloc[0, 0]
            shot_player = "Away_" + str(track_num)
            shot_player_x = tracking_away.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_away.loc[ev_frame][shot_player + "_y"]
            shot_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        # insert shot obso
        shot_df["shot_player"].loc[idx] = shot_player
        shot_df["shot_obso"].loc[idx] = shot_obso

    return shot_df, OBSO_list


def generate_ghost_trajectory(tracking_home, tracking_away, shot):
    # generate ghost trajectory
    # input: tracking data (home and away), shot:shot sequence format pandas series
    # output: tracking_home_ghost, tracking_away_ghost
    # set start and end frame
    start_frame = shot["start_frame"]
    end_frame = shot["end_frame"]
    # max time length = 10 sec
    if end_frame - start_frame > 250:
        start_frame = end_frame - 250
    # define mark player
    mark_df = mark_check(
        tracking_home,
        tracking_away,
        shot["end_frame"],
        attacking_team=shot["shot_player"][:4],
    )
    # extract preeict player
    a1 = shot["shot_player"]
    d1 = mark_df[mark_df["Attack"] == a1].iloc[0]["Defense"]
    for i in range(len(mark_df)):
        if not mark_df.loc[i]["Attack"] == a1:
            a2 = mark_df.loc[i]["Attack"]
            d2 = mark_df.loc[i]["Defense"]
            break
    # generate ghost player
    tracking_home_ghost = tracking_home
    tracking_away_ghost = tracking_away
    a2_x_ghost = []
    a2_y_ghost = []
    d1_x_ghost = []
    d1_y_ghost = []
    d2_x_ghost = []
    d2_y_ghost = []
    # check start velocity
    if a1[:-2] == "Home":
        a2_vx = tracking_home.loc[start_frame][a2 + "_vx"]
        a2_vy = tracking_home.loc[start_frame][a2 + "_vy"]
        d1_vx = tracking_away.loc[start_frame][d1 + "_vx"]
        d1_vy = tracking_away.loc[start_frame][d1 + "_vy"]
        d2_vx = tracking_away.loc[start_frame][d2 + "_vx"]
        d2_vy = tracking_away.loc[start_frame][d2 + "_vy"]
        # predict liner tracjectory
        for i in range(end_frame - start_frame + 1):
            a2_x_ghost.append(
                tracking_home.loc[start_frame][a2 + "_x"] + (a2_vx / 25 * i)
            )
            a2_y_ghost.append(
                tracking_home.loc[start_frame][a2 + "_y"] + (a2_vy / 25 * i)
            )
            d1_x_ghost.append(
                tracking_away.loc[start_frame][d1 + "_x"] + (d1_vx / 25 * i)
            )
            d1_y_ghost.append(
                tracking_away.loc[start_frame][d1 + "_y"] + (d1_vy / 25 * i)
            )
            d2_x_ghost.append(
                tracking_away.loc[start_frame][d2 + "_x"] + (d2_vx / 25 * i)
            )
            d2_y_ghost.append(
                tracking_away.loc[start_frame][d2 + "_y"] + (d2_vy / 25 * i)
            )
        # insert ghost player
        tracking_home_ghost.loc[start_frame:end_frame][a2 + "_x"] = a2_x_ghost
        tracking_home_ghost.loc[start_frame:end_frame][a2 + "_y"] = a2_y_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d1 + "_x"] = d1_x_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d1 + "_y"] = d1_y_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d2 + "_x"] = d2_x_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d2 + "_y"] = d2_y_ghost
    elif a1[:-2] == "Away":
        a2_vx = tracking_away.loc[start_frame][a2 + "_vx"]
        a2_vy = tracking_away.loc[start_frame][a2 + "_vy"]
        d1_vx = tracking_home.loc[start_frame][d1 + "_vx"]
        d1_vy = tracking_home.loc[start_frame][d1 + "_vy"]
        d2_vx = tracking_home.loc[start_frame][d2 + "_vx"]
        d2_vy = tracking_home.loc[start_frame][d2 + "_vy"]
        # predict liner tracjectory
        for i in range(end_frame - start_frame + 1):
            a2_x_ghost.append(
                tracking_away.loc[start_frame][a2 + "_x"] + (a2_vx / 25 * i)
            )
            a2_y_ghost.append(
                tracking_away.loc[start_frame][a2 + "_y"] + (a2_vy / 25 * i)
            )
            d1_x_ghost.append(
                tracking_home.loc[start_frame][d1 + "_x"] + (d1_vx / 25 * i)
            )
            d1_y_ghost.append(
                tracking_home.loc[start_frame][d1 + "_y"] + (d1_vy / 25 * i)
            )
            d2_x_ghost.append(
                tracking_home.loc[start_frame][d2 + "_x"] + (d2_vx / 25 * i)
            )
            d2_y_ghost.append(
                tracking_home.loc[start_frame][d2 + "_y"] + (d2_vy / 25 * i)
            )
        # insert ghost player
        tracking_away_ghost.loc[start_frame:end_frame][a2 + "_x"] = a2_x_ghost
        tracking_away_ghost.loc[start_frame:end_frame][a2 + "_y"] = a2_y_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d1 + "_x"] = d1_x_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d1 + "_y"] = d1_y_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d2 + "_x"] = d2_x_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d2 + "_y"] = d2_y_ghost

    return tracking_home_ghost, tracking_away_ghost


def calc_virtual_obso(tracking_home, tracking_away, event_data, shot_df, Trans, EPV):
    # this function calcurate obso in virtual state
    # set pameters
    params = mpc.default_model_params()
    GK_numbers = [
        mio.find_goalkeeper(tracking_home),
        mio.find_goalkeeper(tracking_away),
    ]
    # calcurate virtual obso
    ghost_obso_list = []
    OBSO_list = []
    for idx in range(len(shot_df)):
        if shot_df.loc[idx]["frame_length"] == 0:
            ghost_obso_list.append("nan")
            continue
        ev_frame = shot_df.loc[idx]["end_frame"] - 1
        attacking_team = shot_df.loc[idx]["shot_player"][:4]
        tracking_home_ghost, tracking_away_ghost = generate_ghost_trajectory(
            tracking_home, tracking_away, shot_df.loc[idx]
        )
        # check GK_numbers
        if np.isnan(tracking_home.loc[ev_frame]["Home_" + GK_numbers[0] + "_x"]):
            GK_numbers[0] = "12"
        if np.isnan(tracking_away.loc[ev_frame]["Away_" + GK_numbers[1] + "_x"]):
            GK_numbers[1] = "12"
        PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
            tracking_home_ghost,
            tracking_away_ghost,
            ev_frame,
            attacking_team,
            params,
            GK_numbers,
        )
        # checking attacking direction
        if attacking_team == "Home":
            if event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(
                    tracking_home_ghost[tracking_home_ghost["Period"] == 1], "Home"
                )
            elif event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direction = mio.find_playing_direction(
                    tracking_home_ghost[tracking_home_ghost["Period"] == 2], "Home"
                )
        elif attacking_team == "Away":
            if event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(
                    tracking_away_ghost[tracking_away_ghost["Period"] == 1], "Away"
                )
            elif event_data.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direction = mio.find_playing_direction(
                    tracking_away_ghost[tracking_away_ghost["Period"] == 2], "Away"
                )
        OBSO, _ = calc_obso(
            PPCF,
            Trans,
            EPV,
            tracking_home_ghost.loc[ev_frame],
            attack_direction=direction,
        )
        OBSO_list.append(OBSO)
        # assign evaluate
        if attacking_team == "Home":
            shot_player = shot_df.loc[idx]["shot_player"]
            shot_player_x = tracking_home_ghost.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_home_ghost.loc[ev_frame][shot_player + "_y"]
            ghost_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        elif attacking_team == "Away":
            shot_player = shot_df.loc[idx]["shot_player"]
            shot_player_x = tracking_away_ghost.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_away_ghost.loc[ev_frame][shot_player + "_y"]
            ghost_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        ghost_obso_list.append(ghost_obso)

    return ghost_obso_list, OBSO_list


def integrate_shotseq_tracking(
    tracking_home, tracking_away, event_data, player_data, jursey_data, Trans, EPV
):
    # this function is integrate shot sequence on tracking
    # input :tracking data(home, away), event data, player_data
    # output :FM_seq_tracking, opponent_seq_tracking
    # check FM event and tracking
    # team id of FM is 124 in player data
    shot_df = extract_shotseq(event_data)
    shot_df, _ = calc_shot_obso(
        shot_df,
        event_data,
        tracking_home,
        tracking_away,
        jursey_data,
        player_data,
        Trans,
        EPV,
    )
    FM_team = player_data[player_data["チームID"] == 124].iloc[0][
        "ホームアウェイF"
    ]  # 1:Home, 2:Away
    if FM_team == 1:
        FM_shot = shot_df[shot_df["Team"] == "Home"].reset_index(drop=True)
        opponent_shot = shot_df[shot_df["Team"] == "Away"].reset_index(drop=True)
    elif FM_team == 2:
        FM_shot = shot_df[shot_df["Team"] == "Away"].reset_index(drop=True)
        opponent_shot = shot_df[shot_df["Team"] == "Home"].reset_index(drop=True)
    # eliminate 0 sec sequence
    FM_shot = FM_shot[FM_shot["frame_length"] != 0].reset_index(drop=True)
    opponent_shot = opponent_shot[opponent_shot["frame_length"] != 0].reset_index(
        drop=True
    )
    # FM extract tracking shot sequence
    FM_seq_tracking = []
    opponent_seq_tracking = []
    for seq_num in range(len(FM_shot)):
        start_frame = FM_shot.loc[seq_num]["start_frame"]
        end_frame = FM_shot.loc[seq_num]["end_frame"] - 1
        # max frame length is 250 (10sec)
        if end_frame - start_frame >= 250:
            start_frame = end_frame - 250
        # check entry player
        if FM_team == 1:
            FM_start = tracking_home.loc[start_frame].dropna().index
            opponent_start = tracking_away.loc[start_frame].dropna().index
            FM_player_num = [s[:-2] for s in FM_start if re.match("Home_\d*_x", s)]
            opponent_player_num = [
                s[:-2] for s in opponent_start if re.match("Away_\d*_x", s)
            ]
            # define shot player as a2
            a2 = FM_shot.loc[seq_num]["shot_player"]
            # set other players
            other_FM_player = list(set(FM_player_num) - set([a2]))
            FM_players = [s + "_x" for s in other_FM_player] + [
                s + "_y" for s in other_FM_player
            ]
            opponent_players = [s + "_x" for s in opponent_player_num] + [
                s + "_y" for s in opponent_player_num
            ]
            other_players = sorted(FM_players) + sorted(opponent_players)
            a2_pos = [a2 + "_x", a2 + "_y"]
            entry_pos = a2_pos + other_players + ["ball_x", "ball_y"]
            # set velocity columns
            FM_players_vel = [s + "_vx" for s in other_FM_player] + [
                s + "_vy" for s in other_FM_player
            ]
            opponent_players_vel = [s + "_vx" for s in opponent_player_num] + [
                s + "_vy" for s in opponent_player_num
            ]
            other_players_vel = sorted(FM_players_vel) + sorted(opponent_players_vel)
            a2_vel = [a2 + "_vx", a2 + "_vy"]
            entry_vel = a2_vel + other_players_vel + ["ball_vx", "ball_vy"]
            entry_players = entry_pos + entry_vel

        elif FM_team == 2:
            FM_start = tracking_away.loc[start_frame].dropna().index
            opponent_start = tracking_home.loc[start_frame].dropna().index
            FM_player_num = [s[:-2] for s in FM_start if re.match("Away_\d*_x", s)]
            opponent_player_num = [
                s[:-2] for s in opponent_start if re.match("Home_\d*_x", s)
            ]
            # define shot player as a2
            a2 = FM_shot.loc[seq_num]["shot_player"]
            # set other players
            other_FM_player = list(set(FM_player_num) - set([a2]))
            FM_players = [s + "_x" for s in other_FM_player] + [
                s + "_y" for s in other_FM_player
            ]
            opponent_players = [s + "_x" for s in opponent_player_num] + [
                s + "_y" for s in opponent_player_num
            ]
            other_players = sorted(FM_players) + sorted(opponent_players)
            a2_pos = [a2 + "_x", a2 + "_y"]
            entry_pos = a2_pos + other_players + ["ball_x", "ball_y"]
            # set velocity columns
            FM_players_vel = [s + "_vx" for s in other_FM_player] + [
                s + "_vy" for s in other_FM_player
            ]
            opponent_players_vel = [s + "_vx" for s in opponent_player_num] + [
                s + "_vy" for s in opponent_player_num
            ]
            other_players_vel = sorted(FM_players_vel) + sorted(opponent_players_vel)
            a2_vel = [a2 + "_vx", a2 + "_vy"]
            entry_vel = a2_vel + other_players_vel + ["ball_vx", "ball_vy"]
            entry_players = entry_pos + entry_vel
        # set tracking data
        tracking_df = pd.DataFrame(columns=entry_players)
        total_tracking = pd.merge(tracking_home, tracking_away)
        for player in entry_pos:
            tracking_df[player] = total_tracking.loc[start_frame - 50 : end_frame][
                player
            ]  # -50 is use of the burn-in
        # calc velocity
        for i, player in enumerate(entry_vel):
            for j, frame in enumerate(range(start_frame - 50, end_frame + 1)):
                tracking_df[player].iloc[j] = (
                    total_tracking[entry_pos[i]].loc[frame + 1]
                    - total_tracking[entry_pos[i]].loc[frame]
                ) * 25
        # append list of match sequence
        FM_seq_tracking.append(tracking_df)

    for seq_num in range(len(opponent_shot)):
        start_frame = opponent_shot.loc[seq_num]["start_frame"]
        end_frame = opponent_shot.loc[seq_num]["end_frame"] - 1
        # max frame length is 250 (10sec)
        if end_frame - start_frame >= 250:
            start_frame = end_frame - 250
        # check entry player
        if FM_team == 1:
            FM_start = tracking_home.loc[start_frame].dropna().index
            opponent_start = tracking_away.loc[start_frame].dropna().index
            FM_player_num = [s[:-2] for s in FM_start if re.match("Home_\d*_x", s)]
            opponent_player_num = [
                s[:-2] for s in opponent_start if re.match("Away_\d*_x", s)
            ]
            # define shot player as a2
            a2 = opponent_shot.loc[seq_num]["shot_player"]
            # set other players
            other_opponent_player = list(set(opponent_player_num) - set([a2]))
            FM_players = [s + "_x" for s in FM_player_num] + [
                s + "_y" for s in FM_player_num
            ]
            opponent_players = [s + "_x" for s in other_opponent_player] + [
                s + "_y" for s in other_opponent_player
            ]
            other_players = sorted(opponent_players) + sorted(FM_players)
            a2_pos = [a2 + "_x", a2 + "_y"]
            entry_pos = a2_pos + other_players + ["ball_x", "ball_y"]
            # set velocity columns
            FM_players_vel = [s + "_vx" for s in FM_player_num] + [
                s + "_vy" for s in FM_player_num
            ]
            opponent_players_vel = [s + "_vx" for s in other_opponent_player] + [
                s + "_vy" for s in other_opponent_player
            ]
            other_players_vel = sorted(opponent_players_vel) + sorted(FM_players_vel)
            a2_vel = [a2 + "_vx", a2 + "_vy"]
            entry_vel = a2_vel + other_players_vel + ["ball_vx", "ball_vy"]
            entry_players = entry_pos + entry_vel

        elif FM_team == 2:
            FM_start = tracking_away.loc[start_frame].dropna().index
            opponent_start = tracking_home.loc[start_frame].dropna().index
            FM_player_num = [s[:-2] for s in FM_start if re.match("Away_\d*_x", s)]
            opponent_player_num = [
                s[:-2] for s in opponent_start if re.match("Home_\d*_x", s)
            ]
            # define shot player as a2
            a2 = opponent_shot.loc[seq_num]["shot_player"]
            other_opponent_player = list(set(opponent_player_num) - set([a2]))
            FM_players = [s + "_x" for s in FM_player_num] + [
                s + "_y" for s in FM_player_num
            ]
            opponent_players = [s + "_x" for s in other_opponent_player] + [
                s + "_y" for s in other_opponent_player
            ]
            other_players = sorted(opponent_players) + sorted(FM_players)
            a2_pos = [a2 + "_x", a2 + "_y"]
            entry_pos = a2_pos + other_players + ["ball_x", "ball_y"]
            # set velocity columns
            FM_players_vel = [s + "_vx" for s in FM_player_num] + [
                s + "_vy" for s in FM_player_num
            ]
            opponent_players_vel = [s + "_vx" for s in other_opponent_player] + [
                s + "_vy" for s in other_opponent_player
            ]
            other_players_vel = sorted(opponent_players_vel) + sorted(FM_players_vel)
            a2_vel = [a2 + "_vx", a2 + "_vy"]
            entry_vel = a2_vel + other_players_vel + ["ball_vx", "ball_vy"]
            entry_players = entry_pos + entry_vel

        # set tracking data
        tracking_df = pd.DataFrame(columns=entry_players)
        total_tracking = pd.merge(tracking_home, tracking_away)
        for player in entry_pos:
            tracking_df[player] = total_tracking.loc[start_frame - 50 : end_frame][
                player
            ]  # -50 is use of burn-in
        # calc velocity
        for i, player in enumerate(entry_vel):
            for j, frame in enumerate(range(start_frame - 50, end_frame + 1)):
                tracking_df[player].iloc[j] = (
                    total_tracking[entry_pos[i]].loc[frame + 1]
                    - total_tracking[entry_pos[i]].loc[frame]
                ) * 25
        # append list of match sequence
        opponent_seq_tracking.append(tracking_df)

    return FM_seq_tracking, opponent_seq_tracking


def calc_press_value(at_pos, df_pos, df_goal_pos):
    # calcurate pressure value in toda's research
    """
    # Args
    at_pos:attacking position like array
    df_pos:defense position like array
    df_goal_pos:goal positon in defense team like array

    # Returns
    press_value(float):value of pressure
    """
    # set ndarray
    at_pos = np.array(at_pos)
    df_pos = np.array(df_pos)
    df_goal_pos = np.array(df_goal_pos)
    # calcurate angle defense and goal
    dis_at_df = np.linalg.norm(df_pos - at_pos)
    goal_vec = df_goal_pos - at_pos
    df_vec = df_pos - at_pos
    cos = np.dot(goal_vec, df_vec) / (np.linalg.norm(goal_vec) * np.linalg.norm(df_vec))
    if cos >= 1 / math.sqrt(2):
        press_value = 1 - dis_at_df / 4
    elif cos <= -1 / math.sqrt(2):
        press_value = 1 - dis_at_df / 2
    else:
        press_value = 1 - dis_at_df / 3
    # not define press value in so far defense
    if press_value < 0:
        press_value = 0

    return press_value


def get_attack_sequence(event_data, player_data):
    """
    # Args
    event_data: event data format Metrica
    player_data: involve team data

    # Returns
    attack_df: data of attack sequence
    """
    # define attack sequence
    attack_df = pd.DataFrame(
        columns=[
            "Team",
            "start_event",
            "start_frame",
            "end_event",
            "end_frame",
            "frame_length",
            "time_length[s]",
            "end_event_type",
        ]
    )
    FM_team = player_data[player_data["チームID"] == 124].iloc[0][
        "ホームアウェイF"
    ]  # 1:Home, 2:Away
    if FM_team == 1:  # opponent is 2(Away)
        attack_event = event_data[event_data["Team"] == "Away"]
    elif FM_team == 2:  # opponent is 1(Home)
        attack_event = event_data[event_data["Team"] == "Home"]
    attack_index = attack_event.index
    # extract consecutive number
    seq_list = []
    index_list = []
    for i in range(len(attack_index)):
        index_list.append(attack_index[i])
        if i == len(attack_index) - 1:
            break
        elif attack_index[i] + 1 == attack_index[i + 1]:
            continue
        else:
            seq_list.append(index_list)
            index_list = []
    # assign dataframe
    Team_list = [attack_event["Team"].iloc[0]] * len(seq_list)
    attack_df["Team"] = Team_list
    for i in range(len(attack_df)):
        attack_df["start_event"].loc[i] = seq_list[i][0]
        attack_df["end_event"].loc[i] = seq_list[i][-1]
        attack_df["start_frame"].loc[i] = attack_event.loc[seq_list[i][0]][
            "Start Frame"
        ]
        attack_df["end_frame"].loc[i] = attack_event.loc[seq_list[i][-1]]["End Frame"]
        attack_df["frame_length"].loc[i] = (
            attack_df["end_frame"].loc[i] - attack_df["start_frame"].loc[i]
        )
        attack_df["time_length[s]"].loc[i] = attack_df["frame_length"].loc[i] / 25
        attack_df["end_event_type"].loc[i] = attack_event.loc[seq_list[i][-1]]["Type"]

    return attack_df


def attack_sequence2tracking(tracking_home, tracking_away, attack_df):
    """
    # Args
    trakcing_home: tracking data of home team
    tracking_away: tracking data of away team
    attack_df: dataframe of attack sequence

    # Returns
    seq_attack_tracking: tracking data for attack sequence in match
    """
    # set dataframe into list
    seq_attack_tracking = []
    # set team name 'Home' or 'Away'
    if attack_df["Team"].loc[0] == "Home":
        opponent_team = "Home"
        opponent_tracking = tracking_home
        FM_team = "Away"
        FM_tracking = tracking_away
    elif attack_df["Team"].loc[0] == "Away":
        opponent_team = "Away"
        opponent_tracking = tracking_away
        FM_team = "Home"
        FM_tracking = tracking_home
    # check attack direction
    first_direction = mio.find_playing_direction(
        opponent_tracking[opponent_tracking["Period"] == 1], opponent_team
    )
    second_direction = mio.find_playing_direction(
        opponent_tracking[opponent_tracking["Period"] == 2], opponent_team
    )
    first_tracking = pd.merge(
        opponent_tracking[opponent_tracking["Period"] == 1],
        FM_tracking[FM_tracking["Period"] == 1],
    )
    second_tracking = pd.merge(
        opponent_tracking[opponent_tracking["Period"] == 2],
        FM_tracking[FM_tracking["Period"] == 2],
    )

    if first_direction == -1:
        first_tracking = first_tracking.iloc[:, 2:] * (-1)
    else:
        first_tracking = first_tracking.iloc[:, 2:]
    if second_direction == -1:
        second_tracking = second_tracking.iloc[:, 2:] * (-1)
    else:
        second_tracking = second_tracking.iloc[:, 2:]
    total_tracking = pd.concat([first_tracking, second_tracking], ignore_index=True)
    for seq_num in range(len(attack_df)):
        start_frame = attack_df.loc[seq_num]["start_frame"]
        end_frame = int(attack_df.loc[seq_num]["end_frame"])
        # check entry players
        opponent_start = opponent_tracking.loc[start_frame].dropna().index
        FM_start = FM_tracking.loc[start_frame].dropna().index
        opponent_player_num = [
            s[:-2] for s in opponent_start if re.match(opponent_team + "_\d*_x", s)
        ]
        FM_player_num = [s[:-2] for s in FM_start if re.match(FM_team + "_\d*_x", s)]
        # set position columns
        opponent_players_pos = [s + "_x" for s in opponent_player_num] + [
            s + "_y" for s in opponent_player_num
        ]
        FM_players_pos = [s + "_x" for s in FM_player_num] + [
            s + "_y" for s in FM_player_num
        ]
        entry_pos = (
            sorted(FM_players_pos) + sorted(opponent_players_pos) + ["ball_x", "ball_y"]
        )
        # set velocity columns
        opponent_players_vel = [s + "_vx" for s in opponent_player_num] + [
            s + "_vy" for s in opponent_player_num
        ]
        FM_players_vel = [s + "_vx" for s in FM_player_num] + [
            s + "_vy" for s in FM_player_num
        ]
        entry_vel = (
            sorted(opponent_players_vel)
            + sorted(FM_players_vel)
            + ["ball_vx", "ball_vy"]
        )
        entry_players = entry_pos + entry_vel
        # set tracking dataframe
        tracking_df = pd.DataFrame(columns=entry_players)
        for player in entry_pos:
            tracking_df[player] = total_tracking.loc[start_frame:end_frame][player]
        # calc velocity
        for i, player in enumerate(entry_vel):
            for j, frame in enumerate(range(start_frame, end_frame + 1)):
                tracking_df[player].iloc[j] = (
                    total_tracking[entry_pos[i]].loc[frame + 1]
                    - total_tracking[entry_pos[i]].loc[frame]
                ) * 25
        # append match sequence into list
        seq_attack_tracking.append(tracking_df)

    return seq_attack_tracking


def create_tracking_df(predict, seq_num=0, player_num=0):
    """
    # Args
    predict: predict tracking shape=(frame_length=121, player=3, seqs_len=1405, feature_len=92)
    seq_num: sequence number
    player_num: predict player number 0-2

    # Returns
    attack_tracking: attacking players position (as Home)
    defense_tracking: defensing players position (as Away)
    """
    # up sampling 10Hz -> 25Hz
    predict = signal.resample_poly(predict, 5, 2, axis=0, padtype="line")
    # set column name
    home_columns = [
        "Home_1_x",
        "Home_1_y",
        "Home_2_x",
        "Home_2_y",
        "Home_3_x",
        "Home_3_y",
        "Home_4_x",
        "Home_4_y",
        "Home_5_x",
        "Home_5_y",
        "Home_6_x",
        "Home_6_y",
        "Home_7_x",
        "Home_7_y",
        "Home_8_x",
        "Home_8_y",
        "Home_9_x",
        "Home_9_y",
        "Home_10_x",
        "Home_10_y",
        "Home_11_x",
        "Home_11_y",
        "ball_x",
        "ball_y",
    ]
    away_columns = [
        "Away_1_x",
        "Away_1_y",
        "Away_2_x",
        "Away_2_y",
        "Away_3_x",
        "Away_3_y",
        "Away_4_x",
        "Away_4_y",
        "Away_5_x",
        "Away_5_y",
        "Away_6_x",
        "Away_6_y",
        "Away_7_x",
        "Away_7_y",
        "Away_8_x",
        "Away_8_y",
        "Away_9_x",
        "Away_9_y",
        "Away_10_x",
        "Away_10_y",
        "Away_11_x",
        "Away_11_y",
        "ball_x",
        "ball_y",
    ]
    attack_tracking = pd.DataFrame(
        columns=home_columns, index=[list(range(len(predict)))]
    )
    defense_tracking = pd.DataFrame(
        columns=away_columns, index=[list(range(len(predict)))]
    )
    times = list(range(len(predict)))
    # set tracking
    for i in range(len(predict)):
        for j in range(3, 12):
            attack_tracking["Home_" + str(j) + "_x"].loc[i] = predict[i][player_num][
                seq_num
            ][4 * (j + 1)]
            attack_tracking["Home_" + str(j) + "_y"].loc[i] = predict[i][player_num][
                seq_num
            ][4 * (j + 1) + 1]
            defense_tracking["Away_" + str(j) + "_x"].loc[i] = predict[i][player_num][
                seq_num
            ][4 * (j + 10)]
            defense_tracking["Away_" + str(j) + "_y"].loc[i] = predict[i][player_num][
                seq_num
            ][4 * (j + 10) + 1]
        attack_tracking["Home_1_x"].loc[i] = predict[i][player_num][seq_num][0]
        attack_tracking["Home_1_y"].loc[i] = predict[i][player_num][seq_num][1]
        attack_tracking["Home_2_x"].loc[i] = predict[i][player_num][seq_num][12]
        attack_tracking["Home_2_y"].loc[i] = predict[i][player_num][seq_num][13]
        defense_tracking["Away_1_x"].loc[i] = predict[i][player_num][seq_num][4]
        defense_tracking["Away_1_y"].loc[i] = predict[i][player_num][seq_num][5]
        defense_tracking["Away_2_x"].loc[i] = predict[i][player_num][seq_num][8]
        defense_tracking["Away_3_x"].loc[i] = predict[i][player_num][seq_num][9]
        attack_tracking["ball_x"].loc[i] = predict[i][player_num][seq_num][88]
        attack_tracking["ball_y"].loc[i] = predict[i][player_num][seq_num][88]
        defense_tracking["ball_x"].loc[i] = predict[i][player_num][seq_num][89]
        defense_tracking["ball_y"].loc[i] = predict[i][player_num][seq_num][89]
    attack_tracking["Time [s]"] = times
    defense_tracking["Time [s]"] = times

    return attack_tracking, defense_tracking
