import pickle
import time
from collections import defaultdict

import requests

import serverfiles


# https://www.ebi.ac.uk/biostudies/help#
URL_ARRAYEXPRESS = "https://www.ebi.ac.uk/biostudies/api/v1/search?facet.collection=arrayexpress&pageSize=1"

URL_GEO = "https://download.biolab.si/datasets/geo/"

# https://github.com/ncbi/DbGaP-FHIR-API-Docs/blob/production/quickstart.md
URL_DBGAP = "https://dbgap-api.ncbi.nlm.nih.gov/fhir/x1"


def arrayexpress() -> None:
    STEP_SIZE = 1

    sample_distribution = dict()
    
    # get total number of datasets
    res = requests.get(URL_ARRAYEXPRESS)
    N_datasets = res.json()["totalHits"]
    print(f"All datasets: {N_datasets}")

    # for each dataset size
    remaining = N_datasets
    for i in range(1, 200 + 1, STEP_SIZE):
        # they have no rate limit, so this is ok i guess?
        res = requests.get(f"{URL_ARRAYEXPRESS}&sample_count=%5B{i}%20TO%20{i+STEP_SIZE-1}%5D")
        sample_distribution[i] = res.json()["totalHits"]
        # print(f"{i} samples: {sample_distribution[i]}")

        remaining -= sample_distribution[i]

    sample_distribution[201] = remaining
    print("Remaining datasets:", remaining)

    with open("data/samples_arrayexpress.pickle", "wb") as f:
        pickle.dump(sample_distribution, f)


def geo() -> None:
    server_files = serverfiles.ServerFiles(server=URL_GEO)
    data_info = server_files.allinfo()

    sample_distribution = defaultdict(int)
    for data in data_info.values():
        sample_distribution[int(data["sample_count"])] += 1

    with open("data/samples_geo.pickle", "wb") as f:
        pickle.dump(sample_distribution, f)


def dbgap() -> None:
    sample_distribution = defaultdict(int)

    next_url = f"{URL_DBGAP}/ResearchStudy"
    page = 0
    while True:
        if next_url is None:
            print("Cannot find next url: ", data["link"])
            break

        time.sleep(0.2)
        res = requests.get(next_url)
        if res.status_code != 200:
            print(res.status_code, res.json())
            break
        data = res.json()
        next_url = next((l["url"] for l in data["link"] if l["relation"] == "next"), None)

        # iterate over studies, find sample count
        page += 1
        print("Page", page)
        try:
            for study in data["entry"]:
                study_extensions = study["resource"]["extension"]
                for se in study_extensions:
                    if se["url"].endswith("ResearchStudy-Content"):
                        for see in se["extension"]:
                            if see["url"].endswith("ResearchStudy-Content-NumSamples"):
                                sample_distribution[see["valueCount"]["value"]] += 1
                                break
                        break
        except:
            break

        print(sum(k*v for k, v in sample_distribution.items()))

    with open("data/samples_dbgap.pickle", "wb") as f:
        pickle.dump(sample_distribution, f)


if __name__ == "__main__":
    arrayexpress()
    geo()
    dbgap()
    ...
