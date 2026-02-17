import csv
import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

"""
idea here is to load participants from csv, then provide iterator on demand for data
there are millions of keystrokes so it's dumb to load them all into memory

for now i want to take advantage of the keyrecs structure as well.

MAYBE? will use fixed to get a "benchmark" of each user to normalize by. fixed
is many reps in of retyping the same random string of text, however it's quite
short so might have some noise. ignoring it rn tho
"""


class KeyRecsData:
    """
    SPECIFICALLY LOADS KEYRECS DATASET AND DOES EMBEDDING STUFF
    """

    def __init__(
        self,
        demographics: str | pathlib.Path,
        fixed_data: str | pathlib.Path,
        free_data: str | pathlib.Path,
    ):
        """
        potential alt iterator approach is to shard into individuals files
        if file is too big/not grouped, but text is grouped so for simplicity
        i will be using byte offsets / ranges
        """

        # for debug rn, might do differently later
        self.data_info = {
            "src": {
                "demographics": demographics,
                "fixed_data": fixed_data,
                "free_data": free_data,
            }
        }

        self.participants = {}
        self.country_to_layout = {}
        self.free_embeddings = {}

    def _normalize(
        self,
        X: np.ndarray,
        *,
        tempo_col: int = 3,
        eps: float = 1e-6,
        center: bool = True,
        clip: Optional[float] = 6.0,
    ):
        X = X.astype(np.float64, copy=False)
        t = np.median(X[:, tempo_col])
        denom = max(abs(t), eps)
        Xn = X / denom

        if center:
            Xn = Xn - np.median(Xn, axis=0)

        if clip is not None:
            Xn = np.clip(Xn, -clip, clip)

        return Xn

    def _country_to_layout(self):
        pass

    def iter_free(self, par):
        """
        take par (participant) and ret (seshid,fingerprint)

        FREE HEADERS

        participant,session,
        key1,key2,DU.key1.key1,DD.key1.key2,DU.key1.key2,UD.key1.key2,UU.key1.key2
        """

        # yield (participant, session, embedding) foreach session
        span = self.participants[par].get("free_span")
        if not span:
            return
        start, end = span

        free_pth = self.data_info["src"]["free_data"]

        with open(free_pth, "r", newline="", encoding="utf-8") as f:
            header = f.readline().strip()
            cols = [c.strip() for c in header.split(",") if c.strip()]

            f.seek(start)

            curr = None
            buf = []

            while f.tell() < end:
                line = f.readline()
                if not line:
                    break

                row = next(csv.reader([line]))
                if row and row[-1] == "":
                    row = row[:-1]

                if len(row) != len(cols):
                    continue

                d = dict(zip(cols, row))

                session = int(d["session"].strip())

                # dont change to if curr bc curr int
                if curr is None:
                    curr = session
                elif session != curr:
                    X = np.asarray(buf, dtype=np.float64)

                    if len(X) > 0:
                        embedding = self.fingerprint_sesh(X)
                        yield (par, curr, len(buf), embedding)

                    buf = []
                    curr = session

                vals = [
                    d["DU.key1.key1"],
                    d["DD.key1.key2"],
                    d["DU.key1.key2"],
                    d["UD.key1.key2"],
                    d["UU.key1.key2"],
                ]

                if any(v == "" for v in vals):
                    continue

                buf.append([float(fv) for fv in vals])

            if buf:
                X = np.asarray(buf, dtype=np.float64)
                embedding = self.fingerprint_sesh(X)
                yield (par, curr, len(buf), embedding)

    def fingerprint_sesh(self, x):
        """
        do normalization and ret embedding
        """
        Xn = self._normalize(x)
        q_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        # need to do per column
        qs = np.quantile(Xn, q_levels, axis=0)

        embedding = qs.T.reshape(-1)

        overlap_ud = float(np.mean(Xn[:, 3] < 0))
        overlap_dd = float(np.mean(Xn[:, 1] < 0))

        return np.concatenate([embedding, [overlap_ud, overlap_dd]])

    @classmethod
    def from_files(
        cls,
        demographics: str | pathlib.Path,
        fixed_data: str | pathlib.Path,
        free_data: str | pathlib.Path,
    ):
        key_recs_data = KeyRecsData(demographics, fixed_data, free_data)

        participants: dict

        with open(demographics, newline="", encoding="utf-8") as f:
            participants = {row["participant"]: row for row in csv.DictReader(f)}

        # go through free & assign ranges for each participant (offset by id len when looping)
        curr_par: str | None = None
        start = None

        # eventually should prolly look into one pass normalization however it seems kind of messy
        # so instead im just making the byte ranges THEN iterating. on my larger dataset will prolly
        # have to cook up some dumb stuff tho
        with open(free_data, "rb") as fr:
            # SKIP HEADER
            fr.readline()

            while True:
                pos = fr.tell()
                line = fr.readline()
                if not line:
                    break

                parti = line.split(b",", 1)[0].decode("utf-8").strip()

                if parti != curr_par:
                    if curr_par:
                        participants[curr_par]["free_span"] = (start, pos)
                    curr_par = parti
                    start = pos

            if curr_par:
                participants[curr_par]["free_span"] = (start, fr.tell())

        key_recs_data.participants = participants

        return key_recs_data


def keyrec_dataframe(
    demographics: str | pathlib.Path,
    fixed_data: str | pathlib.Path,
    free_data: str | pathlib.Path,
    min_r: int,
):

    k = KeyRecsData.from_files(demographics, fixed_data, free_data)

    for p in k.participants.keys():
        for par, session, r, emb in k.iter_free(p):
            if r < min_r:
                continue
            k.free_embeddings.setdefault(par, {})[session] = {"r": r, "emb": emb}

    rows = []
    for par, sessions in k.free_embeddings.items():
        demo = k.participants[par]
        for session, obj in sessions.items():
            emb = obj["emb"]
            r = obj["r"]

            row = {
                "participant": par,
                "session": int(session),
                "r": int(r),
                **{f"e{i}": float(emb[i]) for i in range(len(emb))},
                "gender": demo.get("gender"),
                "age": demo.get("age"),
                "handedness": demo.get("handedness"),
                "nationality": demo.get("nationality"),
            }
            rows.append(row)

    df_sessions = pd.DataFrame(rows)
    df_sessions["handedness"] = df_sessions["handedness"].str.strip().str.lower()
    df_sessions["gender"] = df_sessions["gender"].str.strip().str.lower()
    df_sessions["age"] = df_sessions["age"].astype(int)
    df_sessions["age_bin"] = pd.cut(
        df_sessions["age"],
        # todo: add buckets for younger age groups once dataset is more diverse
        bins=[0, 18, 25, 35, 50, 100],
        labels=["0-18", "19-25", "25-35", "35-50", "50+"],
    )

    df_sessions.to_parquet("out/keyrecs_sessions.parquet")

    emb_cols = [c for c in df_sessions.columns if c.startswith("e")]
    df_participants = df_sessions.groupby("participant", as_index=False).agg(
        n_sessions=("session", "nunique"),
        total_r=("r", "sum"),
        **{c: (c, "mean") for c in emb_cols},
        gender=("gender", "first"),
        handedness=("handedness", "first"),
        age=("age", "first"),
        age_bin=("age_bin", "first"),
        nationality=("nationality", "first"),
    )

    df_participants.to_parquet("out/keyrecs_participants.parquet")

    print("Saved to out/keyrecs_sessions.parquet and out/keyrecs_participants.parquet")

    # i should unironically make output paths an option but im lazy
    return "out/keyrecs_sessions.parquet", "out/keyrecs_participants.parquet"


if __name__ == "__main__":
    DEMOGRAPHICS_PATH = "data/demographics.csv"
    FIXED_PATH = "data/fixed-test.csv"
    FREE_PATH = "data/free-text.csv"

    MIN_R = 50

    kr_sessions, kr_parti = keyrec_dataframe(
        DEMOGRAPHICS_PATH, FIXED_PATH, FREE_PATH, MIN_R
    )
