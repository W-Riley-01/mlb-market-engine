from resolver import MatchupResolver

# With arsenal
r_new = MatchupResolver()
card_new = r_new.generate_probabilities(batter_id=X, pitcher_id=Y, as_of_date='2025-09-15')

# Without (reverts to old binary FB/OS blend via overall_rates fallback)
r_old = MatchupResolver(load_arsenal_profile=False)
card_old = r_old.generate_probabilities(batter_id=X, pitcher_id=Y, as_of_date='2025-09-15')

print("New:", card_new)
print("Old:", card_old)
print("Delta:", {k: card_new[k] - card_old[k] for k in card_new})