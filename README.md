# how to run

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_md

python3 pseudonymize.py

# Workflow

1. spacy entity recognition
2. regular expression matching
3. generalized versions of identifying data


# Example

    "We are considering Elena Rodriguez for the Senior DevOps role. She currently makes $165,000 at Netflix and lives in Los Gatos."

1.  Post-Tokenization (Pass 1 & 2):

    "We are considering `Person A` for the Senior DevOps role. She currently makes `Money A` at `Organization A` and lives in `Location A.`"

2.  Generalization Map (Pass 3 Final):

    "We are considering `Person A` for the Senior DevOps role. She currently makes `$100K-$200K` at `Organization A` and lives in `California`."

