import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import random

# ==========================================
# Mappings
# ==========================================


MODE_MAPPING = {
    # Car
    "Mode::Car": "car",

    # Walk
    "Mode::Walk": "walk",

    # Bike
    "Mode::Bicycle": "bike",
    "Mode::Ebicycle": "bike",
    "Mode::MotorbikeScooter": "bike",

    # Bus
    "Mode::Bus": "bus",

    # Train (rail-based)
    "Mode::Train": "train",
    "Mode::RegionalTrain": "train",
    "Mode::LightRail": "train",
    "Mode::Tram": "train",
}

PURPOSE_MAPPING = {
    "home": "home",
    "work": "work",
    "eat": "eat",

    # leisure group
    "leisure": "leisure",
    "sport": "leisure",
    "family_friends": "leisure",
    "shopping": "leisure",

    # errand group
    "errand": "errand",
    "assistance": "errand",
    "wait": "errand",
}

def load_and_process_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset, apply mappings, and sort.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed dataframe with mapped 'mode' and 'purpose' columns,
                      sorted by user_id, event_date, and seq_idx.
    """
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Apply MODE_MAPPING to 'trip_mode' column to create 'mode' column
    df['mode'] = df['trip_mode'].map(MODE_MAPPING)

    # Apply PURPOSE_MAPPING to 'end_purpose' column to create 'purpose' column
    df['purpose'] = df['end_purpose'].map(PURPOSE_MAPPING)

    
    # convert event date and event time to datetime
    
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['event_time'] = pd.to_datetime(df['event_time'])
    # Sort by ['user_id', 'event_date', 'seq_idx']
    df = df.sort_values(by=['user_id','event_date','seq_idx'])
    # # Filter out any rows where mode or purpose is NaN after mapping (if any)
    
    df = df.dropna()
    
    return df

def create_user_sequences(df: pd.DataFrame) -> Dict[str, List[List[Tuple[str, str]]]]:
    """
    Group data into sequences of (Mode, Purpose) tuples per user.
    
    Args:
        df (pd.DataFrame): Sorted processed dataframe.

    Returns:
        Dict[str, List[List[Tuple[str, str]]]]: A dictionary where keys are user_ids
        and values are lists of sequences.
        Each sequence is a list of (mode, purpose) tuples corresponding to a day (or continuous block).
        
        Example structure:
        {
            'user_1': [
                [('car', 'work'), ('walk', 'eat'), ('car', 'home')],  # Day 1
                [('bus', 'leisure'), ('bus', 'home')]                # Day 2
            ],
            ...
        }
    """
    user_dict = {}
    # Group by user_id
    user_trips = df.groupby('user_id')
    # NOTE: user_trips is a groupby object, not a df!
    for user_id , user_df in user_trips:
        user_dict[user_id] = []

        daily_trips = user_df.groupby('event_date')
        for _,daily_user_trip in daily_trips:
            seq = list(zip(
            daily_user_trip['mode'],
            daily_user_trip['purpose'],
            daily_user_trip['event_time']
            ))

            user_dict[user_id].append(seq)

    return user_dict



def train_test_split_by_user(sequences: Dict[str, Any], test_size: float = 0.2, random_seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Split data ensuring all sequences from a specific user go into the same split.
    
    Args:
        sequences (Dict): Dictionary of user sequences from create_user_sequences.
        test_size (float): Proportion of users to include in the test split.
        random_seed (int): Seed for reproducibility.

    Returns:
        Tuple[List, List]: (train_sequences, test_sequences)
        Each element is a flattened list of sequences (lists of tuples) suitable for HMM training/testing.
        We flatten the user structure because the HMM doesn't care about user ID, just the observed sequences.
    """
    # all unique user_ids
    user_ids = list(sequences.keys())
    #  shuffle user_ids
    random.seed(random_seed)
    random.shuffle(user_ids)
    # Split user_ids into train_users and test_users based on test_size
    user_count = len(user_ids)
    test_count = int(user_count * test_size)
    test_user = user_ids[:test_count]
    train_user = user_ids[test_count:]
    # Agg all sequences from train_users into a single list (train_sequences)
    train_sequences = []
    for u in train_user:
        for seq in sequences[u]:
            train_sequences.append(seq)
    #Agg all sequences from test_users into a single list (test_sequences)
    test_sequences = []
    for u in test_user:
        for seq in sequences[u]:
            test_sequences.append(seq)
    
    return (train_sequences,test_sequences)


# TEST CASES

if __name__ == "__main__":
    # test

    filepath = "../data/mode_purpose_hmm.csv"

    df = load_and_process_data(filepath)

    user_dict = create_user_sequences(df=df)

    for user,trips in user_dict.items():
        print("------ user -------")
        print(user)
        print('------ trips of the user -----')
        print(trips)
        break

    tp = train_test_split_by_user(sequences=user_dict,test_size=0.2,random_seed=42)
    import pdb; pdb.set_trace()
    train, test = tp[0], tp[1]

    print("----- train set entry -----")
    print(train[0])
 
    # summary
    print("----- data summary -----")
    print("Train len:", len(train))
    print("Test len:", len(test))
    print("Sample train seq length:", len(train[2]))
    print("First few obs in first train seq:", train[2][:3])

    # stats
    max_len_train=max_len_test=0
    min_len_train=min_len_test=1000

    for seq_train,seq_test in zip(train,test):
        if len(seq_train)>max_len_train:
            max_len_train = len(seq_train)
        elif len(seq_train)<min_len_train:
            min_len_train = len(seq_train)

        if len(seq_test)>max_len_test:
            max_len_test = len(seq_test)
        elif len(seq_test)<min_len_test:
            min_len_test = len(seq_test)
    # train stats
    print("----- training set stats -----")
    print(f"maxixum length of daily trips in training: {max_len_train}")
    print(f"min length of daily trips in training: {min_len_train} ")
    print('----- testing set stats -----')
    print(f"maxixum length of daily trips in testing: {max_len_test}")
    print(f"min length of daily trips in testing: {min_len_test} ")
    




