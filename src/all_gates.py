import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

class AllGatesModel:
    def __init__(self, gpa_menu, gre_menu, rank_menu, random_state=42):
        self.gpa_menu = gpa_menu
        self.gre_menu = gre_menu
        self.rank_menu = rank_menu
        self.random_state = random_state
        self.best_params = None   # dict with keys: gpa_cut, gre_cut, rank_max
        self.cv_scores_ = None

    def _predict_with_params(self, X, p):
        gpa_ok  = (X["gpa"].values >= p["gpa_cut"])
        gre_ok  = (X["gre"].values >= p["gre_cut"])
        rank_ok = (X["rank"].values <= p["rank_max"])
        return (gpa_ok & gre_ok & rank_ok).astype(int)

    def _scores(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_for_admit": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall_for_admit": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "f1_for_admit": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        best_f1 = -1.0
        best_params = None
        best_avg = None

        for gpa_cut in self.gpa_menu:
            for rank_max in self.rank_menu:
                for gre_cut in self.gre_menu:
                    fold_scores = []
                    for tr_idx, val_idx in skf.split(X, y):
                        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                        p = {"gpa_cut": gpa_cut, "gre_cut": gre_cut, "rank_max": rank_max}
                        y_hat = self._predict_with_params(X_val, p)
                        fold_scores.append(self._scores(y_val, y_hat))
                    avg = {k: float(np.mean([fs[k] for fs in fold_scores])) for k in fold_scores[0].keys()}

                    if avg["f1_for_admit"] > best_f1:
                        best_f1 = avg["f1_for_admit"]
                        best_params = {"gpa_cut": gpa_cut, "gre_cut": gre_cut, "rank_max": rank_max}
                        best_avg = avg

        self.best_params = best_params
        self.cv_scores_ = best_avg
        return self

    def predict(self, X):
        if self.best_params is None:
            raise RuntimeError("Call fit(...) first.")
        return self._predict_with_params(X, self.best_params)

    def decision_path(self, X):
        if self.best_params is None:
            raise RuntimeError("Call fit(...) first.")
        p = self.best_params
        gpa_ok  = (X["gpa"].values >= p["gpa_cut"])
        gre_ok  = (X["gre"].values >= p["gre_cut"])
        rank_ok = (X["rank"].values <= p["rank_max"])
        admit = (gpa_ok & gre_ok & rank_ok).astype(int)

        rows = []
        for i in range(len(X)):
            rows.append({
                "gpa": float(X.iloc[i]["gpa"]),
                "gre": float(X.iloc[i]["gre"]),
                "rank": int(X.iloc[i]["rank"]),
                "gpa_passed": bool(gpa_ok[i]),
                "gre_passed": bool(gre_ok[i]),
                "rank_passed": bool(rank_ok[i]),
                "admit": int(admit[i])
            })
        return rows
