import pandas as pd
import numpy as np
import random
from pickle import dump
import os

# Set a random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def run_game(
    subject_id,
    main_folder,
    output_dir,
    RESECTION,
    NODES,
    rounds=1000,
    max_sigma=4,
    verbose=True
    ):

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(main_folder, f"cvs_pairs.csv"))
    connectivity_measures = ['PAC'] #list(df['CM'].unique())
    nodes = NODES[subject_id]
    resection = RESECTION[subject_id]
    group_size = min(int(len(nodes) * 0.1), len(resection))

    if verbose:
        print(f"\n▶ Running Epigame for subject {subject_id}")
        print(f"  Nodes: {len(nodes)} | Node group size: {group_size}")
        print(f"  Connectivity measures: {len(connectivity_measures)}")
        print(f"  Rounds per game: {rounds}")

    class Player:
        def __init__(self, AI, deck, name, n_in_hand=5):
            """
            Initializes a Player object.
            Parameters:
            - AI: The artificial intelligence strategy used by the player.
            - deck: The deck of cards available for the player.
            - name: The name of the player.
            - n_in_hand: The number of cards initially dealt to the player's hand.
            """
            self.deck = deck
            self.n_in_hand = n_in_hand
            self.logic = AI
            self.cards = []  # Player's hand
            self.score = []  # Player's score (not used)
            self.name = name

        def shuffle(hand): hand.cards = sorted(hand.deck)

        def check(scores): return scores.real
        
        def play(god, *players):
            """
            Determines the player's move in a game.
            Parameters:
            - god: The player making the move.
            - players: Other players in the game.
            Returns:
            - int: The chosen card to play.
            """
            best, other = max(god.cards), []
            for player in players:
                other += player.cards
            if best >= max(other): return best
            else: return min(god.cards)

        def card(draw):
            """Draws a card based on the player's AI logic."""
            return draw.logic(draw)

    def rn_choice(logic):
        return logic.cards.pop(random.randint(0, len(logic.cards) - 1))

    def play(*game):
        # Initialize each player's real and best score attributes to 0
        for player in game:
            player.real, player.best = 0, 0
            player.shuffle()
        # Define a function to find the winners among the players
        def resolve(hand, by=[], best=-1):
            for player, card in hand:
                # Check if the card is equal to the best card, add player to 'by' list
                if best == card:
                    by.append(player)
                # If the card is greater than the best, update 'best' and set 'by' to [player]
                if best < card:
                    best = card
                    by = [player]
            return by
        winners = resolve([(player, player.card()) for player in game])
        # Increment the 'real' attribute for each winner
        for player in winners:
            player.real += 1
        # Update each player's score by appending the result of the 'check' method
        for player in game:
            player.score.append(player.check())

    def summary_stat(cvs_list):
        return (max(cvs_list)*min(cvs_list)) / (np.mean(cvs_list))

    def match_resection(group_scores, resection):
        """
        Matches resection items to group_scores.
        Automatically detects if resection contains electrodes (e.g., 'A1')
        or channels (e.g., 'A1-A2') and returns a match dictionary.
        """
        # Check if all resection entries are in channel format (contain '-')
        is_channel = all('-' in r for r in resection)

        if is_channel:
            # Resection contains full channel names like 'A1-A2'
            resection_match = {
                player: [ch for ch in player if ch in resection]
                for player in group_scores}
        else:
            # Resection contains individual electrodes like 'A1'
            def contacts_in_resection(group):
                matches = []
                for pair in group:
                    contacts = pair.split('-')
                    matches.extend([c for c in contacts if c in resection])
                return matches
            resection_match = {
                player: contacts_in_resection(player)
                for player in group_scores
            }
        return resection_match

    def check_winners(sorted_players, sigmas, resection, group_size):
        """
        Check winners' performance based on cross-validation scores and resection overlap.
        Parameters:
        - sorted_players (dict): Dictionary containing sorted players' cross-validation scores.
        - sigmas (float): Number of standard deviations for defining the threshold.
        - nodes (list): List of node labels.
        - resection (list): List of nodes in the resection area.
        - group_size (float): Player or random group node number.
        Returns:
        - Tuple: (number of winners above threshold, winner-loser ratio, median resection overlap, mean resection overlap).
        """
        # Calculate group scores and resection matches
        group_scores = {player:sum(sorted_players[player]) for player in sorted_players}
        resection_match = match_resection(group_scores, resection)

        mean = np.average(list(group_scores.values()))
        std = np.std(list(group_scores.values()))

        # Define threshold based on mean and standard deviations
        thresh = mean + (std * sigmas)
        winners_above_thresh = [players for players in group_scores if group_scores[players] >= thresh]
        N_winners_above_thresh = len(winners_above_thresh)

        if N_winners_above_thresh > 0:
            # Calculate resection overlap for winners and losers
            resection_overlap_winners = [len(resection_match[players]) / group_size for players in winners_above_thresh]
            resection_overlap_losers = [len(resection_match[players]) / group_size for players in resection_match if
                                        players not in winners_above_thresh]
            # Calculate the winner-loser ratio
            winner_loser_ratio = np.mean(resection_overlap_winners) / np.mean(resection_overlap_losers) if np.mean(resection_overlap_losers) > 0 else None
            return N_winners_above_thresh, winner_loser_ratio
        else:
            return N_winners_above_thresh, 0
        
    result_all_cm = {}
    df_sub = df[df.Subject == subject_id]

    seen = set()
    players = []
    while len(players) < len(nodes) * 5:
        candidate = tuple(sorted(np.random.choice(nodes, size=group_size, replace=False)))
        if candidate not in seen:
            seen.add(candidate)
            players.append(candidate)
    players = sorted(players)

    if verbose: print(f"  Generated {len(players)} player groups")

    for i,measure in enumerate(connectivity_measures):

        if verbose: print(f"\n  [{i+1}/{len(connectivity_measures)}] Connectivity measure: {measure}")

        df_sub_cm = df_sub[df_sub.CM == measure]
        hands = []  # initialize hands, a list of tuples (player, deck)
        for group_labels in players:
            if group_labels not in [hand[0] for hand in hands]:
                split_labels = df_sub_cm['Labels'].str.split('<->', expand=True)
                relevant_rows = df_sub_cm[((split_labels[0].isin(group_labels)) & split_labels[1].isin(group_labels))]
                group_deck = list(relevant_rows['CVS'].apply(lambda x: summary_stat(list(map(float, x.strip('[]').split())))))
                hands.append((group_labels, group_deck))
        n_cards = len(hands[0][1])
        game_score = {tuple(p):[] for p in [hand[0] for hand in hands]}
        for turn in range(rounds):

            if verbose and turn % max(1, rounds // 10) == 0: print(f"Round {turn}/{rounds}")

            for r in range(n_cards):
                game = [Player(AI=rn_choice, deck=hand[1], name=tuple(hand[0]), n_in_hand=n_cards) for hand in hands]
                play(*game)
                scores = sorted([(player.name, player.score) for player in game], key=lambda x:x[1], reverse=True)
                top_score, fall = scores[0][1], 0
                for name, score in scores:
                    if score==top_score: 
                        game_score[name].append(1)
                        fall+=1
                    elif score!=top_score: 
                        break
                for name,score in scores[fall:]: 
                    game_score[name].append(0)
        sorted_players = {k:v for k, v in sorted(game_score.items(), key=lambda item: sum(item[1]), reverse=True)}
        for sigma in range(max_sigma, 0, -1):
            n_winners, ratio = check_winners(sorted_players, sigma, resection, group_size)
            result_all_cm[(measure, sigma)] = {
                "subject": subject_id,
                "measure": measure,
                "N_winners": n_winners,
                "overlap_ratio": ratio,
                "group_size": group_size,
                "game_scores": game_score
            }
            if verbose:
                print(
                    f"    σ={sigma}: "
                    f"N_winners={n_winners}, "
                    f"overlap_ratio={ratio:.3f}" if ratio is not None else
                    f"    σ={sigma}: N_winners={n_winners}, overlap_ratio=None"
                )
    out_path = os.path.join(output_dir, f"scores_sub{subject_id}.p")
    dump(result_all_cm, open(out_path, "wb"))
    if verbose:
        print(f"\n✔ Finished subject {subject_id}")
        print(f"✔ Results saved to: {out_path}")
