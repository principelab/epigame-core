import os
import pandas as pd
from epigame.utils import REc

def aggregate_cv_scores(result_dir, subject_ids, output_csv, freq_bands=[None,(1,4),(4,8),(8,13),(13,30),(30,70),(70,150)]):
    Subject, Pair, Labels, CM, CVS = [], [], [], [], []

    for bands in freq_bands:
        connectivity_measures = ["PAC"] if bands is None else ["SCR", "SCI", "PLV", "PLI", "CC"]

        for measure in connectivity_measures:
            for subject_id in subject_ids:
                ext = f"-{bands[0]}-{bands[1]}" if bands else ""
                filename = os.path.join(result_dir, f"{subject_id}-{measure}{ext}.res")

                if not os.path.exists(filename):
                    print(f"Missing: {filename}")
                    continue

                res = REc.load(filename).data.pairs

                for pair_tuple in res:
                    Subject.append(subject_id)
                    Pair.append(pair_tuple[0])
                    Labels.append(pair_tuple[1])
                    CM.append(measure + ext)
                    CVS.append(pair_tuple[2])

    df = pd.DataFrame({"Subject": Subject, "Pair": Pair, "Labels": Labels, "CM": CM, "CVS": CVS})
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
