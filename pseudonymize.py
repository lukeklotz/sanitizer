
'''
This program attempts to replace sensitive information with pseudonyms.
It takes advantage of the spaCy library, which uses a local LLM to process text.
spaCy is able to "label" and identify named entities in text by calling "label_".
We take advantage of this with a key value pair dictionary called LABEL_TO_CATEGORY which
contains the label name identified by spaCy as well as a value associated
with the named entity. This value is used to generate a pseudonym for the 
anonymized text.
'''


import sys
import string
import spacy
import argparse

LABEL_TO_CATEGORY = {
    "PERSON": "Person",
    "LOC": "Location",
    #"GPE": "Location",
    #"FAC": "Location",
    "ORG": "Organization",
    "MONEY": "Money",
    "DATE": "Date",
    #"TIME": "Time",
    #"CARDINAL": "Number",
    #"ORDINAL": "Ordinal",
}


# generator for generic label replacements
def label_generator():
    letters = string.ascii_uppercase
    n = 0
    while True:
        s, q = "", n
        while True:
            s = letters[q % 26] + s
            q = q // 26 - 1
            if q < 0:
                break
        yield s
        n += 1


class Pseudonymizer:
    # load english tokensizer
    # this class uses spacy to identify named entities
    def __init__(self, model="en_core_web_md"):
        self.spacey_model = spacy.load(model)

    def sanitize(self, text):
        doc = self.spacey_model(text)

        # if there are no named entities, return the original text
        if len(doc.ents) == 0:
            return text
        
        '''
        uncomment here to see named entities
        this is useful for adding to LABEL_TO_CATEGORY
        for ent in doc.ents:
            print(f"ent: {ent.text} type: {ent.label_}")

        '''

        gens = {}

        # create a label generator for each category
        for cat in set(LABEL_TO_CATEGORY.values()):
            gens[cat] = label_generator()
        mapping = {}
        spans = [] 

        #process named entities processed by spacy and
        #create a mapping from names to pseudonuyms and
        #a list of text spans to replace
        for entry in doc.ents:
            category = LABEL_TO_CATEGORY.get(entry.label_)
            if category is None:
                continue
            key = entry.text
            if key not in mapping:
                mapping[key] = f"{category} {next(gens[category])}"
            spans.append((entry.start_char, entry.end_char, mapping[key]))

        # Splice replacements in from right to left so offsets stay valid.
        sanitized = text
        for start, end, pseudonym in sorted(spans, reverse=True):
            sanitized = sanitized[:start] + pseudonym + sanitized[end:]

        return sanitized, mapping

    def similarity_score_text(self, text1, text2):
        doc1 = self.spacey_model(text1)
        doc2 = self.spacey_model(text2)
        return doc1.similarity(doc2)

SAMPLE_INPUTS = [
    "I was seen at St. Jude's yesterday for my Stage 2 Hypertension. "
    "Can you explain if the Lisinopril dosage my doctor, Dr. Aris, "
    "prescribed is standard?",

    "Draft a memo regarding the acquisition of CloudStream Inc. for $450M. "
    "The lead negotiator is Sarah Jenkins from our Austin office, and we "
    "need to finalize this by Friday, the 14th.",

    "We are considering Elena Rodriguez for the Senior DevOps role. "
    "She currently makes $165,000 at Netflix and lives in Los Gatos. "
    "Is this competitive?",

    "I can't log into the production server at 192.168.1.45. "
    "My username is admin_jake, and I keep getting a 'timeout' error.",

    "Please summarize the IEP for student ID 883920. Tommy Miller is "
    "struggling with reading comprehension in Ms. Gable's 4th-grade class.",

    "Generate a listing description for 45 Oakhaven Lane. The owner wants "
    "to highlight the new roof and the proximity to Google's headquarters.",

    "Remind me to pick up my daughter from daycare at 4:30 PM. "
    "Then, book a table for two.",

    "I have $50,000 in my savings account, and I'm worried about the tax "
    "implications for my 2025 filing after moving states.",
]

def print_result(label, before, after, similarity):
    print(f"\n ------------- {label} ---------------")
    print(f" ------ BEFORE: {before}")
    print(f" ------ AFTER : {after}")
    print(f" similarity score: {similarity}")


def main():
    parser = argparse.ArgumentParser(description="Pseudonymize sensitive text.")
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Custom text to sanitize. If omitted, runs against all sample inputs."
    )
    args = parser.parse_args()

    p = Pseudonymizer()

    if args.input:
        sanitized, mapping = p.sanitize(args.input)
        similarity = p.similarity_score_text(args.input, sanitized)
        print_result("Custom Input", args.input, sanitized, similarity)
    else:
        for i, text in enumerate(SAMPLE_INPUTS, 1):
            sanitized, mapping = p.sanitize(text)
            similarity = p.similarity_score_text(text, sanitized)
            print_result(f"Input {i}", text, sanitized, similarity)

if __name__ == "__main__":
    main()
