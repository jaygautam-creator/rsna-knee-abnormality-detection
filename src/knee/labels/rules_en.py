"""v0 English rule-based report -> 12 binary labels.

Purpose is measurement, not production: it tells us the ceiling of report mining
by scoring against the 58 expert-labelled studies. Each condition is a list of
(pattern, polarity) rules applied to sentence-split text with negation scoping.
"""

import re

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Words that negate a finding when they appear before it in the same clause.
NEG = re.compile(
    r"\b(no|not|non|without|absent|negative for|free of|unremarkable|intact|normal|"
    r"preserved|maintained|continuous|homogeneous|neither|nor)\b",
    re.I,
)
# Words that re-affirm despite a nearby negation ("no acute fracture, but ACL tear")
BUT = re.compile(r"\b(but|however|although|though|except|whereas)\b", re.I)

# Terms indicating an actual abnormality, per condition.
POS = {
    "ACL": r"(acl|anterior cruciate)",
    "MCL": r"(mcl|medial collateral)",
    "Medial Meniscus": r"medial meniscus|medial meniscal",
    "Lateral Meniscus": r"lateral meniscus|lateral meniscal",
    "Medial OA": r"medial (femorotibial|tibiofemoral|compartment)",
    "Lateral OA": r"lateral (femorotibial|tibiofemoral|compartment)",
    "PF OA": r"(patellofemoral|patellar|trochlea|trochlear|retropatellar)",
    "Effusion": r"(effusion|joint fluid|suprapatellar (bursa|recess) (distension|distention|fluid))",
    "Synovitis": r"(synovitis|synovial (thickening|proliferation|hypertrophy)|pannus)",
    "Baker's": r"(baker'?s? cyst|popliteal cyst|gastrocnemio-?semimembranosus)",
    "Contusion": r"(contusion|bone bruise|bone marrow (o?edema|contusion)|marrow o?edema)",
    "Fracture": r"(fracture|fractured|osteochondral (defect|lesion|fragment)|avulsion)",
}

# The abnormality qualifier that must co-occur with the anatomy term.
QUAL = {
    "ACL": r"(tear|torn|rupture|ruptur|disrupt|discontinu|sprain|partial|complete|"
           r"strain|injur|attenuat|laxity|absent fibers|increased signal|high signal)",
    "MCL": r"(tear|torn|rupture|ruptur|disrupt|sprain|strain|injur|thicken|"
           r"o?edema|increased signal|high signal|grade)",
    "Medial Meniscus": r"(tear|torn|rupture|degenerat|maceration|macerated|"
           r"displaced|bucket|flap|radial|horizontal|oblique|complex|extrusion|"
           r"truncat|blunted|absent|post-?operative|resect)",
    "Lateral Meniscus": r"(tear|torn|rupture|degenerat|maceration|macerated|"
           r"displaced|bucket|flap|radial|horizontal|oblique|complex|extrusion|"
           r"truncat|blunted|absent|post-?operative|resect|discoid)",
    "Medial OA": r"(osteoarthr|arthros|degenerat|chondral|cartilage (loss|thinning|defect)|"
           r"osteophyt|joint space narrowing|subchondral (cyst|sclerosis))",
    "Lateral OA": r"(osteoarthr|arthros|degenerat|chondral|cartilage (loss|thinning|defect)|"
           r"osteophyt|joint space narrowing|subchondral (cyst|sclerosis))",
    "PF OA": r"(osteoarthr|arthros|chondromalacia|chondral (defect|loss|thinning|ulcer)|"
           r"cartilage (loss|thinning|defect|ulcer)|osteophyt|degenerat)",
    "Effusion": None,       # the term itself is the finding
    "Synovitis": None,
    "Baker's": None,
    "Contusion": None,
    "Fracture": None,
}

# Findings explicitly called trivial/physiologic are treated as absent.
TRIVIAL = re.compile(r"\b(trace|physiologic|physiological|minimal|tiny|negligible)\b", re.I)

_SENT = re.compile(r"(?<=[.;:\n])\s+|\n+")


def _sentences(text: str):
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _negated(sent: str, span: tuple) -> bool:
    """True if a negation cue governs the match at `span`."""
    start = span[0]
    left = sent[:start]
    # limit scope to the current clause
    clause = re.split(r",|\band\b", left)[-1] if len(left) > 60 else left
    if BUT.search(left):
        clause = BUT.split(left)[-1]
    if NEG.search(clause):
        return True
    # trailing negation: "ACL ... is intact / not torn"
    right = sent[span[1]: span[1] + 80]
    if re.match(r"[^.]{0,40}\b(is |are |appears? )?(intact|normal|preserved|"
                r"continuous|unremarkable|not torn|without tear)\b", right, re.I):
        return True
    return False


def label_report(text: str) -> dict:
    out = {c: 0 for c in LABELS}
    if not isinstance(text, str):
        return out
    for sent in _sentences(text):
        for cond in LABELS:
            if out[cond]:
                continue
            anat = re.compile(POS[cond], re.I)
            qual = QUAL[cond]
            for m in anat.finditer(sent):
                if _negated(sent, m.span()):
                    continue
                if qual is None:
                    if TRIVIAL.search(sent[max(0, m.start() - 40): m.start()]):
                        continue
                    out[cond] = 1
                    break
                # anatomy + abnormality qualifier must co-occur in the sentence
                if re.search(qual, sent, re.I):
                    out[cond] = 1
                    break
    return out
