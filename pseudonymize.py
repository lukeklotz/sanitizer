import string
import spacy
import argparse
import re
from geopy.geocoders import Nominatim


LABEL_TO_CATEGORY = {
    "PERSON": "Person",
    "LOC":    "Location",
    "GPE":    "Location",
    "ORG":    "Organization",
    "MONEY":  "Money",
    "DATE":   "Date",
}

GENERALIZE_CATEGORIES = {"Money", "Date", "Location", "Time", "ID"}

REGEX = [
    (r'\$\d+(?:\.\d+)?[KkMmBb]?\b',              "Money"),
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  "IP Address"),
    (r'([Uu]sername is )(\S+)',                    "Username"),
    (r'\b\d{6,}\b',                                "ID"),
]

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


def generalize_money(text):
    clean = re.sub(r'[$,]', '', text)
    clean = re.sub(r'[Mm]', '000000', clean)
    clean = re.sub(r'[Kk]', '000', clean)

    try:
        amount = float(clean)
        magnitude = 10 ** (len(str(int(amount))) - 1)
        lower = (amount // magnitude) * magnitude
        upper = lower + magnitude

        def fmt(n):
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.0f}B"
            elif n >= 1_000_000:
                return f"{n / 1_000_000:.0f}M"
            elif n >= 1_000:
                return f"{n / 1_000:.0f}K"
            else:
                return f"{n:.0f}"

        return f"{fmt(lower)}-{fmt(upper)}"

    except ValueError:
        return "undisclosed amount"


def generalize_date(text):
    relative_past = {"yesterday", "today", "recently", "last week"}
    if any(word in text.lower() for word in relative_past):
        return "recently"

    relative_future = {"next week", "tomorrow", "in a few days"}
    if any(word in text.lower() for word in relative_future):
        return "soon"

    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        decade = (int(year_match.group()) // 10) * 10
        return f"{decade}s"

    if re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                 text.lower()):
        return "near date"

    return "unspecified time"


def generalize_location(text):
    try:
        geolocator = Nominatim(user_agent="pseudonymizer")
        loc = geolocator.geocode(text, addressdetails=True)
        if loc:
            address = loc.raw.get("address", {})
            return (
                address.get("state") or
                address.get("country") or
                "undisclosed region"
            )
    except Exception:
        pass
    return "undisclosed region"


def generalize_id(text):
    return f"{len(text)}-digit ID"


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
    def __init__(self, model="en_core_web_md"):
        self.spacy_model = spacy.load(model)

    def spacy_sanitize(self, text, mapping, gens):
        doc = self.spacy_model(text)
        spans = []

        for entry in doc.ents:
            category = LABEL_TO_CATEGORY.get(entry.label_)
            if category is None:
                continue
            key = entry.text
            if key not in mapping:
                mapping[key] = f"{category} {next(gens[category])}"
            spans.append((entry.start_char, entry.end_char, mapping[key]))

        sanitized = text
        for start, end, pseudonym in sorted(spans, reverse=True):
            sanitized = sanitized[:start] + pseudonym + sanitized[end:]

        return sanitized

    def regex_sanitize(self, text, mapping, gens):
        sanitized = text
        for pattern, category in REGEX:
            for match in reversed(list(re.finditer(pattern, sanitized))):
                if match.lastindex and match.lastindex >= 2:
                    key = match.group(2)
                    start, end = match.start(2), match.end(2)
                else:
                    key = match.group()
                    start, end = match.start(), match.end()

                if key not in mapping:
                    if category not in gens:
                        gens[category] = label_generator()
                    mapping[key] = f"{category} {next(gens[category])}"

                sanitized = sanitized[:start] + mapping[key] + sanitized[end:]

        return sanitized

    def generalize(self, sanitized, mapping):
        GENERALIZERS = {
            "Money":    generalize_money,
            "Date":     generalize_date,
            "Location": generalize_location,
            "ID": generalize_id
        }

        for original, pseudonym in mapping.items():
            category = pseudonym.split()[0]
            if category in GENERALIZERS:
                sanitized = sanitized.replace(pseudonym, GENERALIZERS[category](original))

        return sanitized

    def sanitize(self, text):
        gens = {cat: label_generator() for cat in set(LABEL_TO_CATEGORY.values())}
        mapping = {}

        sanitized = self.spacy_sanitize(text, mapping, gens)
        sanitized = self.regex_sanitize(sanitized, mapping, gens)
        sanitized = self.generalize(sanitized, mapping)

        return sanitized, mapping

    def similarity_score_text(self, text1, text2):
        doc1 = self.spacy_model(text1)
        doc2 = self.spacy_model(text2)
        return doc1.similarity(doc2)


def print_result(label, before, after, similarity):
    print(f"\n ------------- {label} ---------------")
    print(f" ------ BEFORE: {before}")
    print(f" ------ AFTER : {after}")
    print(f" similarity score: {similarity}")


def main():
    parser = argparse.ArgumentParser(description="Pseudonymize sensitive text.")
    parser.add_argument("-i", "--input", type=str)
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