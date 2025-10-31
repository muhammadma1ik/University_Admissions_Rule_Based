# Loads data -> split into training and final test -> learn rule numbers (5-fold)
# -> freeze rules -> one-time final test -> save metrics, confusion matrices, rules, and probe cases.

import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

from src.score_points import ScorePointsModel
from src.all_gates import AllGatesModel

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----- Load -----
DATA_PATH = os.path.join("data", "ucla_admissions.csv")
data = pd.read_csv(DATA_PATH)

expected = {"admit", "gre", "gpa", "rank"}
if not expected.issubset(set(data.columns)):
    raise ValueError(f"CSV must contain columns: {expected}")

X = data[["gre", "gpa", "rank"]].copy()
y = data["admit"].copy()

# ----- Split (keep class ratio similar) -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training size:", len(X_train))
print("Final test size:", len(X_test))
print("\nTraining label counts:\n", y_train.value_counts())
print("\nFinal test label counts:\n", y_test.value_counts())

# ----- Menus of values to try -----
gpa_menu = [3.2, 3.4, 3.6]
gre_menu = [300, 315, 325]
rank_menu = [1, 2, 3]
total_needed_menu = [1, 2, 3]  # only for points model

# ----- Train both models (selection done only on training pile) -----
points_model = ScorePointsModel(gpa_menu, gre_menu, rank_menu, total_needed_menu, random_state=42).fit(X_train, y_train)
gates_model  = AllGatesModel(gpa_menu, gre_menu, rank_menu, random_state=42).fit(X_train, y_train)

with open(os.path.join(RESULTS_DIR, "chosen_parameters_and_learning_scores.json"), "w") as f:
    json.dump({
        "points_model": {"best_params": points_model.best_params, "learning_scores_average": points_model.cv_scores_},
        "all_gates_model": {"best_params":  gates_model.best_params,  "learning_scores_average": gates_model.cv_scores_}
    }, f, indent=2)

print("\nChosen params (from training only):")
print("Points:", points_model.best_params)
print("All-gates:", gates_model.best_params)

# ----- Final one-time test -----
def final_test(model):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    res = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_for_admit": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_for_admit": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_for_admit": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "cm": cm.tolist()
    }
    return res, cm

points_res, points_cm = final_test(points_model)
gates_res,  gates_cm  = final_test(gates_model)

def dump_results(name, res, cm):
    print(f"\n=== Final test: {name} ===")
    for k in ["accuracy", "precision_for_admit", "recall_for_admit", "f1_for_admit", "balanced_accuracy"]:
        print(f"{k}: {res[k]:.4f}")
    print("Confusion matrix (rows true: [reject, admit], cols predicted: [reject, admit]):\n", cm)

    pd.DataFrame(cm, index=["true_reject", "true_admit"], columns=["pred_reject", "pred_admit"]) \
        .to_csv(os.path.join(RESULTS_DIR, f"confusion_matrix__{name.replace(' ', '_')}.csv"))
    pd.DataFrame([res]).drop(columns=["cm"]).to_csv(os.path.join(RESULTS_DIR, f"metrics__{name.replace(' ', '_')}.csv"), index=False)

dump_results("Points model", points_res, points_cm)
dump_results("All-gates model", gates_res, gates_cm)

# ----- Save final rules in plain text -----
with open(os.path.join(RESULTS_DIR, "final_rules__Points_model.txt"), "w") as f:
    bp = points_model.best_params
    f.write(
        "Points model rules:\n"
        f"- If GPA >= {bp['gpa_cut']} add 1 point\n"
        f"- If GRE >= {bp['gre_cut']} add 1 point\n"
        f"- If rank <= {bp['rank_max']} add 1 point\n"
        f"- Admit if total points >= {bp['total_needed']}\n"
    )

with open(os.path.join(RESULTS_DIR, "final_rules__All-gates_model.txt"), "w") as f:
    bp = gates_model.best_params
    f.write(
        "All-gates model rule:\n"
        f"- Admit only if GPA >= {bp['gpa_cut']} AND GRE >= {bp['gre_cut']} AND rank <= {bp['rank_max']}\n"
        "- Otherwise reject\n"
    )

# ----- Tiny probe cases for discussion -----
probe = pd.DataFrame([
    {"gre": 330, "gpa": 3.1, "rank": 2},  # very high GRE, low GPA
    {"gre": 305, "gpa": 3.8, "rank": 3},  # high GPA, modest GRE
    {"gre": 315, "gpa": 3.5, "rank": 1},  # strong + good rank
    {"gre": 315, "gpa": 3.5, "rank": 4},  # same but worst rank
])
with open(os.path.join(RESULTS_DIR, "probe_cases_points.json"), "w") as f:
    json.dump(points_model.decision_path(probe), f, indent=2)
with open(os.path.join(RESULTS_DIR, "probe_cases_all_gates.json"), "w") as f:
    json.dump(gates_model.decision_path(probe), f, indent=2)

print("\nSaved everything under ./results. Done.")
