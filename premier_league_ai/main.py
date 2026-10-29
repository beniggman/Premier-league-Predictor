
import pandas as pd
from pathlib import Path

from support import calculate_points, last_five
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# LOAD THE DATASET

data_path = Path(__file__).parent / "23.csv"

matches = pd.read_csv(data_path)

required_columns = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST"
]

matches = matches[required_columns].copy()

matches["Date"] = pd.to_datetime(
    matches["Date"],
    dayfirst=True,
    errors="coerce"
)

matches = matches.dropna(
    subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
)

matches = matches.sort_values("Date").reset_index(drop=True)

print("Dataset loaded:", len(matches), "matches")


# CALCULATE PRE-MATCH TEAM FORM


team_results = {}
match_forms = []

for match_date, daily_matches in matches.groupby("Date", sort=True):

    for _, match in daily_matches.iterrows():

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        if home_team not in team_results:
            team_results[home_team] = []

        if away_team not in team_results:
            team_results[away_team] = []

        home_form = last_five(team_results[home_team])
        away_form = last_five(team_results[away_team])

        match_forms.append({
            "Date": match_date,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "HomeForm": home_form,
            "AwayForm": away_form,
            "Result": match["FTR"]
        })

    for _, match in daily_matches.iterrows():

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]
        result = match["FTR"]

        if result == "H":
            team_results[home_team].append("W")
            team_results[away_team].append("L")

        elif result == "A":
            team_results[home_team].append("L")
            team_results[away_team].append("W")

        elif result == "D":
            team_results[home_team].append("D")
            team_results[away_team].append("D")


form_data = pd.DataFrame(match_forms)

print("\nFirst 10 matches:")
print(form_data.head(10).to_string(index=False))

print("\nTotal matches:", len(form_data))


# PREPARE DATA FOR MACHINE LEARNING


features = ["HomeForm", "AwayForm"]

X = form_data[features]
y = form_data["Result"]


# SPLIT CHRONOLOGICALLY


split_index = int(len(form_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining matches:", len(X_train))
print("Testing matches:", len(X_test))


# TRAIN THE MODEL

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nModel training complete!")


# EVALUATE THE MODEL


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["H", "D", "A"],
        target_names=["Home win", "Draw", "Away win"],
        zero_division=0
    )
)

print("\nConfusion matrix (actual rows, predicted columns):")
print(
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=["H", "D", "A"]),
        index=["Actual H", "Actual D", "Actual A"],
        columns=["Predicted H", "Predicted D", "Predicted A"]
    )
)


# PREDICT A MATCH BETWEEN TWO TEAMS


def predict_match(home_team, away_team):

    if home_team == away_team:
        print("Choose two different teams.")
        return

    if home_team not in team_results:
        print(f"Unknown home team: {home_team}")
        return

    if away_team not in team_results:
        print(f"Unknown away team: {away_team}")
        return

    home_form = last_five(team_results[home_team])
    away_form = last_five(team_results[away_team])

    new_match = pd.DataFrame(
        [[home_form, away_form]],
        columns=features
    )

    prediction = model.predict(new_match)[0]
    probabilities = model.predict_proba(new_match)[0]

    result_names = {
        "H": "Home win",
        "D": "Draw",
        "A": "Away win"
    }

    print(f"\n{home_team} vs {away_team}")
    print(f"Home team recent-form points: {home_form}")
    print(f"Away team recent-form points: {away_form}")
    print("Predicted outcome:", result_names[prediction])

    print("\nModel probability estimates:")
    for outcome, probability in zip(model.classes_, probabilities):
        print(f"{result_names[outcome]}: {probability:.1%}")


# TRY A PREDICTION


predict_match("Arsenal", "Man City")
