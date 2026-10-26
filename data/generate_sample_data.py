"""Step 1 - create a SMALL SYNTHETIC flotation dataset so the whole pipeline can run out of the box.

!!  This is NOT the paper's data.  The paper uses (a) real lead-zinc plant inspection logs,
!!  (b) flotation PDFs (standards, books, papers, manuals) and (c) 100 expert-annotated sentences.
!!  Replace the files in data/raw/{pdf,images,csv} and data/training/flotation_train.jsonl with real ones
!!  (see README "Datasets you must upload yourself").

Creates
  data/raw/csv/inspection_records.csv        structured data  D_stru  (paper Eq. 1 / Fig. 3)
  data/raw/images/*.png                      froth / flowsheet images
  data/raw/pdf/*.pdf                         unstructured text + figure + caption  D_unstru (Eq. 2)
  data/training/flotation_train.jsonl        100 annotated sentences  (system / user / assistant, Eq. 14 format)
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 7

SYSTEM = ("You are a Froth Flotation Domain Knowledge Engineer. Extract (subject, relation, object) triples "
          "with ontology labels from flotation process statements and answer in JSON.")

PARAM = ["pulp concentration", "pH value", "aeration rate", "froth depth", "impeller speed", "pulp level",
         "flotation time", "grinding fineness", "reagent dosage", "feed flow rate"]
PRODUCT = ["concentrate grade", "recovery rate", "tailing grade", "froth stability", "production efficiency",
           "foam color", "foam size"]
EQUIP = ["flotation cell", "agitator", "froth scraper", "air blower", "slurry pump", "level sensor"]
REAGENT = ["xanthate", "copper sulfate", "lime", "pine oil", "MIBC", "sodium silicate"]
STAGE = ["fast roughing", "roughing", "scavenging", "cleaning"]
ORE = ["lead-zinc ore", "galena", "sphalerite", "pyrite"]


def T(s, st, r, o, ot):
    return dict(subject=s, subject_type=st, relation=r, object=o, object_type=ot)


def make_sentences(n=100):
    rnd = random.Random(SEED)
    P, PR, EQ, R, ST, ORE_ = (lambda l: rnd.choice(l)), (lambda l: rnd.choice(l)), None, None, None, None

    def pick(l, exclude=()):
        return rnd.choice([x for x in l if x not in exclude])

    makers = []

    def inc():
        p, pr = pick(PARAM), pick(PRODUCT)
        return f"Increasing the {p} increases the {pr}.", [T(p, "Parameter", "increases", pr, "Product")]

    def dec():
        p, pr = pick(PARAM), pick(PRODUCT)
        return f"A higher {p} decreases the {pr}.", [T(p, "Parameter", "decreases", pr, "Product")]

    def keep():
        eq, p = pick(EQUIP), pick(PARAM)
        return f"The {eq} keeps the {p} stable.", [T(eq, "Equipment", "keeps", p, "Parameter")]

    def uses():
        st, r = pick(STAGE), pick(REAGENT)
        return f"The {st} stage uses {r}.", [T(st, "ProcessStage", "uses", r, "Reagent")]

    def prod():
        st, pr = pick(STAGE), pick(["concentrate grade", "recovery rate", "tailing grade"])
        return f"The {st} stage produces a stable {pr}.", [T(st, "ProcessStage", "produces", pr, "Product")]

    def foll():
        i = rnd.randrange(len(STAGE) - 1)
        return f"The {STAGE[i + 1]} stage follows the {STAGE[i]} stage.", \
            [T(STAGE[i + 1], "ProcessStage", "follows", STAGE[i], "ProcessStage")]

    def caus():
        r, pr = pick(REAGENT), pick(PRODUCT)
        return f"Excess {r} causes poor {pr}.", [T(r, "Reagent", "causes", pr, "Product")]

    def aff():
        p, pr = pick(PARAM), pick(PRODUCT)
        return f"The {p} affects the {pr}.", [T(p, "Parameter", "affects", pr, "Product")]

    def prop():
        o, p = pick(ORE), pick(["particle size", "grinding fineness"])
        return f"The {o} has a wide {p}.", [T(o, "RawOre", "has_property", p, "Parameter")]

    makers = [inc, dec, keep, uses, prod, foll, caus, aff, prop]
    out, seen = [], set()
    while len(out) < n:
        k = rnd.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]       # avg ~1.7 triples / sentence
        parts, trips = [], []
        for _ in range(k):
            s, t = rnd.choice(makers)()
            parts.append(s.rstrip("."))
            trips += t
        sent = (parts[0] + "".join(f", and {p[0].lower() + p[1:]}" for p in parts[1:])) + "."
        if sent in seen:
            continue
        seen.add(sent)
        out.append((sent, trips))
    return out


def write_training():
    d = ROOT / "data" / "training"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "flotation_train.jsonl", "w", encoding="utf-8") as f:
        for i, (sent, trips) in enumerate(make_sentences(100)):
            f.write(json.dumps(dict(id=i, system=SYSTEM, user=sent,
                                    assistant=json.dumps({"triples": trips}, ensure_ascii=False)),
                               ensure_ascii=False) + "\n")


def write_csv():
    rnd = random.Random(SEED)
    (RAW / "csv").mkdir(parents=True, exist_ok=True)
    rows = [("Time", "Process", "Equipment", "Operating status", "Problem description"),
            ("2025/2/3 8:00", "Roughing", "F-01", "Normal operation", "No abnormalities"),
            ("2025/2/3 8:10", "Cleaning", "F-02", "Normal operation", "Liquid level exceeds set value by 0.05m"),
            ("2025/2/3 8:20", "Scavenging", "S-01", "Motor noise is relatively loud", "Check motor bearings"),
            ("2025/2/3 8:30", "Tailing", "P-01", "Insufficient flow rate", "Pipeline may have slight blockage")]
    procs = {"Fast roughing": "FR", "Roughing": "F", "Scavenging": "S", "Cleaning": "C"}
    status = ["Normal operation", "Normal operation", "Normal operation", "Froth layer too thin",
              "Insufficient flow rate", "Motor noise is relatively loud", "Aeration rate low"]
    problem = ["No abnormalities", "No abnormalities", "Check motor bearings", "Increase frother dosage slightly",
               "Pipeline may have slight blockage", "Liquid level exceeds set value by 0.05m", "Adjust air blower valve"]
    for i in range(26):
        p = rnd.choice(list(procs))
        rows.append((f"2025/2/3 {8 + (i + 4) * 10 // 60}:{(i + 4) * 10 % 60:02d}", p, f"{procs[p]}-0{rnd.randint(1, 4)}",
                     rnd.choice(status), rnd.choice(problem)))
    with open(RAW / "csv" / "inspection_records.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def write_images():
    from PIL import Image, ImageDraw
    (RAW / "images").mkdir(parents=True, exist_ok=True)
    rnd = random.Random(SEED)

    def froth(name, base, n, rmax, tint):
        im = Image.new("RGB", (480, 320), base)
        dr = ImageDraw.Draw(im)
        for _ in range(n):
            x, y, r = rnd.randint(0, 480), rnd.randint(0, 320), rnd.randint(rmax // 3, rmax)
            c = tuple(min(255, max(0, v + rnd.randint(-20, 20))) for v in tint)
            dr.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(235, 235, 235), width=2)
        im.save(RAW / "images" / name)

    froth("froth_normal.png", (110, 110, 115), 220, 22, (150, 150, 158))     # lead-gray, small uniform bubbles
    froth("froth_overflow.png", (60, 55, 50), 70, 45, (95, 85, 75))           # dark, large bubbles
    im = Image.new("RGB", (640, 220), (250, 250, 250))
    dr = ImageDraw.Draw(im)
    names = ["Raw ore", "Fast roughing", "Roughing", "Scavenging", "Cleaning", "Concentrate"]
    for i, nm in enumerate(names):
        x = 15 + i * 105
        dr.rectangle([x, 80, x + 90, 130], outline=(40, 40, 40), width=2)
        dr.text((x + 8, 98), nm, fill=(0, 0, 0))
        if i < len(names) - 1:
            dr.line([x + 90, 105, x + 105, 105], fill=(0, 0, 0), width=2)
    dr.text((250, 160), "Tailing", fill=(0, 0, 0))
    im.save(RAW / "images" / "flowsheet.png")


DOC1 = [
    "Froth flotation separates hydrophobic minerals from hydrophilic gangue by attaching mineral particles to air bubbles.",
    "In a lead-zinc plant the raw ore is ground and mixed with water in the mixing bowl to form a slurry.",
    "Xanthate is the most widely used collector for sulfide minerals such as galena and sphalerite.",
    "Copper sulfate activates sphalerite so that it can be recovered after the lead concentrate has been removed.",
    "Lime is added to raise the pH value and to depress pyrite during the lead flotation circuit.",
    "Pine oil and MIBC are frothers that reduce bubble size and improve froth stability.",
    "The fast roughing stage recovers coarse liberated particles early and protects them from over-grinding.",
    "The roughing stage floats most of the valuable mineral at a pulp concentration of 30 to 35 percent solids.",
    "The scavenging stage treats the roughing tailing to recover remaining valuable mineral and raise the recovery rate.",
    "The cleaning stage upgrades the rougher concentrate by removing entrained gangue and improves concentrate grade.",
    "A pH value of 11.5 keeps pyrite depressed while galena floats with xanthate.",
    "An aeration rate of 0.8 cubic metres per minute per square metre gives a stable froth layer.",
]
CAP1 = [
    "Fig. 1. Froth image under normal operating conditions: bubbles are small and uniform and the foam color is lead gray.",
    "Fig. 2. Froth overflow condition: the foam size is large and the foam color is dark, indicating an over-flotation state.",
    "Fig. 3. Flowsheet of the lead-zinc flotation plant from raw ore through fast roughing, roughing, scavenging and cleaning to concentrate.",
]
DOC2 = [
    "Froth depth is controlled by the pulp level, and a deeper froth layer increases concentrate grade but decreases recovery rate.",
    "When the impeller speed rises, the agitator disperses air more finely and the foam size becomes smaller.",
    "A low aeration rate produces a thin froth layer and reduces production efficiency.",
    "The level sensor measures the pulp level and the slurry pump adjusts the tailing discharge to keep the level stable.",
    "If the flotation time is kept constant, a higher pulp concentration reduces the residence time of particles and the recovery rate falls.",
    "The foam color changes from dark to lead gray when the collector dosage is reduced and the froth becomes cleaner.",
    "Excess pine oil causes a viscous froth that lowers concentrate grade because gangue is entrained.",
    "Operators inspect each cell every ten minutes and record the operating status, equipment number and any abnormality.",
    "Motor noise at the scavenging cell often indicates bearing wear and should be checked at the next inspection.",
    "A blocked tailing pipeline reduces flow rate and may cause the level in the flotation cell to exceed the set value.",
]


def write_pdfs():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
    (RAW / "pdf").mkdir(parents=True, exist_ok=True)
    st = getSampleStyleSheet()

    def build(name, title, paras, figs):
        flow = [Paragraph(title, st["Title"]), Spacer(1, 8)]
        half = len(paras) // 2
        for p in paras[:half]:
            flow += [Paragraph(p, st["BodyText"]), Spacer(1, 4)]
        for img, cap in figs:
            flow += [Spacer(1, 6), Image(str(RAW / "images" / img), width=300, height=200 if "flow" not in img else 103),
                     Paragraph(cap, st["Italic"]), Spacer(1, 6)]
        for p in paras[half:]:
            flow += [Paragraph(p, st["BodyText"]), Spacer(1, 4)]
        SimpleDocTemplate(str(RAW / "pdf" / name), pagesize=A4).build(flow)

    build("flotation_handbook_excerpt.pdf", "Froth Flotation of Lead-Zinc Ore (sample excerpt)", DOC1,
          [("froth_normal.png", CAP1[0]), ("flowsheet.png", CAP1[2])])
    build("flotation_operations_notes.pdf", "Operating Notes for Flotation Cells (sample excerpt)", DOC2,
          [("froth_overflow.png", CAP1[1])])


def main():
    write_images()
    write_csv()
    write_pdfs()
    write_training()
    print("sample dataset written to data/raw/{csv,images,pdf} and data/training/flotation_train.jsonl")


if __name__ == "__main__":
    main()
