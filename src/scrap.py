from pybaseball import batting_stats_bref, batting_stats

for yr in [2019, 2021, 2023, 2024]:
    print("=== YEAR", yr, "===")
    try:
        df_bref = batting_stats_bref(yr)
        print("bref shape:", df_bref.shape)
    except Exception as e:
        print("bref FAILED:", e)

    try:
        df_fg = batting_stats(yr, yr)
        print("fg shape:", df_fg.shape)
    except Exception as e:
        print("fg FAILED:", e)
