"""Step 1 - generate a SYNTHETIC but complete flotation dataset covering every data type listed for the paper.

!!  Everything here is invented text/data written for software testing.  It is NOT the paper's data and NOT real plant
!!  data.  Replace it with real files (see RUN.md, "Data" section) when you have them.

Data type (paper Sec. II)               Generated here
--------------------------------------  ---------------------------------------------------------------------------
Structured data  (D_stru, Eq. 1)         data/raw/csv/inspection_records.csv   safety-inspection records (Fig. 3 format)
                                         data/raw/csv/production_log.csv       hourly production / operating log
Unstructured text                        data/raw/pdf/  six PDFs: academic paper, industry standard, equipment manual,
                                         accident report, production regulations, technical report
Images and figures                       figures embedded in the PDFs (flowsheet, charts, equipment schematics, froth)
                                         + loose images data/raw/images/*.png with same-name .txt captions
Tables and formulas                      numbered tables and numbered equations inside every PDF (and the HTML bulletin)
Semi-structured data                     sectioned PDFs + data/raw/html/maintenance_bulletin.html (headings, table,
                                         figure + figcaption, formula)
Annotated sentences                      data/training/flotation_train.jsonl (100 template sentences, Eq. 14 format)
"""
import csv
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
SEED = 7
NOTE = "SYNTHETIC SAMPLE DOCUMENT generated for software testing. It is not a real publication or plant record."


# ======================================================================================= images
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _box(ax, x, y, w, h, text, fc="#eaf1fb"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#223", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#223", lw=1.2))


def img_flowsheet(path, detailed=False):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6); ax.axis("off")
    names = ["Raw ore", "Grinding", "Mixing bowl", "Fast roughing", "Roughing", "Scavenging"]
    for i, n in enumerate(names):
        _box(ax, 0.3 + i * 2.6, 4.0, 2.0, 1.2, n)
        if i:
            _arrow(ax, 0.3 + i * 2.6 - 0.6, 4.6, 0.3 + i * 2.6, 4.6)
    _box(ax, 5.5, 1.5, 2.4, 1.2, "Cleaning", "#fdeee0")
    _box(ax, 9.3, 1.5, 2.4, 1.2, "Concentrate", "#e3f4e6")
    _box(ax, 13.0, 1.5, 2.4, 1.2, "Tailing", "#f3e3e3")
    _arrow(ax, 8.4, 4.0, 7.0, 2.7); _arrow(ax, 7.9, 2.1, 9.3, 2.1); _arrow(ax, 14.2, 4.0, 14.2, 2.7)
    if detailed:
        ax.text(8.0, 0.6, "Cleaning I - II - III;  Roughing I - II;  Scavenging I - II - III", ha="center", fontsize=8, style="italic")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def img_cell(path):
    from matplotlib.patches import Circle, Rectangle
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
    ax.add_patch(Rectangle((2, 1), 6, 5, fc="#cfe3f5", ec="#223", lw=2))                 # pulp tank
    ax.add_patch(Rectangle((2, 6), 6, 1.2, fc="#f2f2f2", ec="#223", lw=1))                # froth layer
    rnd = random.Random(3)
    for _ in range(26):
        ax.add_patch(Circle((rnd.uniform(2.3, 7.7), rnd.uniform(6.2, 7.0)), rnd.uniform(.12, .3), fc="white", ec="#667"))
    ax.plot([5, 5], [2.2, 8.4], color="#223", lw=3)                                       # shaft
    ax.add_patch(Rectangle((3.9, 1.9), 2.2, .5, fc="#aab", ec="#223"))                    # impeller
    ax.add_patch(Rectangle((4.3, 8.4), 1.4, .55, fc="#ffd9a8", ec="#223"))                # motor
    ax.add_patch(Rectangle((7.6, 6.9), 1.6, .5, fc="#e8e8e8", ec="#223"))                 # launder
    for txt, xy, xyt in [("Motor", (5, 8.9), (6.6, 8.6)), ("Impeller", (5, 2.1), (8.4, 1.6)),
                         ("Froth layer", (3.0, 6.6), (0.2, 7.8)), ("Pulp", (3.0, 4), (0.4, 4)),
                         ("Overflow launder", (8.4, 7.3), (7.2, 8.0)), ("Level sensor", (2.4, 5.4), (0.1, 5.7))]:
        ax.annotate(txt, xy=xy, xytext=xyt, fontsize=8, arrowprops=dict(arrowstyle="-", color="#556"))
    _arrow(ax, 0.2, 2.3, 2.0, 2.3); ax.text(0.2, 2.6, "Feed inlet", fontsize=8)
    _arrow(ax, 8.0, 1.5, 9.8, 1.5); ax.text(8.1, 0.8, "Tailing discharge", fontsize=8)
    _arrow(ax, 5.0, 0.2, 5.0, 1.9); ax.text(5.15, 0.5, "Air inlet", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def img_feeder(path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
    for x, t, c in [(0.3, "Xanthate tank", "#fdeee0"), (4.2, "Metering pump", "#eaf1fb"), (8.0, "Flow meter", "#eaf1fb"),
                    (11.4, "Flotation cell", "#e3f4e6")]:
        _box(ax, x, 1.8, 2.4, 1.4, t, c)
    for a, b in [(2.7, 4.2), (6.6, 8.0), (10.4, 11.4)]:
        _arrow(ax, a, 2.5, b, 2.5)
    ax.text(7, 4.2, "Reagent dosing line", ha="center", style="italic")
    ax.text(5.4, 1.1, "controller", ha="center", fontsize=8); _arrow(ax, 5.4, 1.3, 5.4, 1.8)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def img_chart_ph(path):
    plt = _mpl()
    ph = [9.5, 10.5, 11.0, 11.5, 12.0]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(ph, [71.2, 80.5, 86.0, 91.3, 84.1], "o-", label="Pb recovery rate (%)")
    ax.plot(ph, [12.4, 9.6, 7.8, 5.9, 5.1], "s--", label="Zn recovery rate (%)")
    ax.set_xlabel("pH value"); ax.set_ylabel("Recovery rate (%)"); ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def img_chart_kinetics(path):
    import numpy as np
    plt = _mpl()
    t = np.array([0.5, 1, 2, 3, 4, 6, 8]); r = 92 * (1 - np.exp(-0.55 * t))
    rnd = random.Random(1)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(t, [v + rnd.uniform(-1.5, 1.5) for v in r], "o", label="measured")
    tt = np.linspace(0, 8, 80); ax.plot(tt, 92 * (1 - np.exp(-0.55 * tt)), "-", label="first-order fit")
    ax.set_xlabel("Flotation time (min)"); ax.set_ylabel("Recovery rate (%)"); ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def img_froth(path, base, n, rmax, tint, seed):
    from PIL import Image, ImageDraw
    rnd = random.Random(seed)
    im = Image.new("RGB", (480, 320), base)
    dr = ImageDraw.Draw(im)
    for _ in range(n):
        x, y, r = rnd.randint(0, 480), rnd.randint(0, 320), rnd.randint(max(rmax // 3, 3), rmax)
        o = rnd.randint(-20, 20)                                                  # same offset on all channels -> stays gray
        c = tuple(min(255, max(0, v + o)) for v in tint)
        dr.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(235, 235, 235), width=2)
        dr.ellipse([x - r // 3, y - r // 2, x - r // 6, y - r // 3], fill=(245, 245, 245))      # highlight
    im.save(path)


# loose images: <name>: (drawer, caption sidecar text)
LOOSE = {
    "froth_normal": (lambda p: img_froth(p, (110, 110, 115), 220, 22, (150, 150, 158), 1),
                     "Fig. A1. Froth image under normal operating conditions: bubbles are small and uniform and the foam color is lead gray."),
    "froth_overflow": (lambda p: img_froth(p, (60, 55, 50), 70, 45, (95, 85, 75), 2),
                       "Fig. A2. Froth overflow condition: the foam size is large and the foam color is dark, indicating an over-flotation state."),
    "froth_thin_layer": (lambda p: img_froth(p, (150, 150, 150), 40, 16, (185, 185, 190), 3),
                         "Fig. A3. Thin froth layer caused by a low aeration rate: the froth depth is small and the bubbles are sparse."),
    "equipment_flotation_cell": (img_cell,
                                 "Fig. A4. Equipment image of a mechanical flotation cell showing the motor, impeller, froth layer, level sensor and tailing discharge."),
    "process_flowsheet": (lambda p: img_flowsheet(p, detailed=False),
                          "Fig. A5. Flowsheet of the lead-zinc flotation plant from raw ore through grinding, fast roughing, roughing, scavenging and cleaning to concentrate and tailing."),
}


def write_loose_images():
    d = RAW / "images"
    if d.exists():
        for f in d.glob("*"):
            f.unlink()
    d.mkdir(parents=True, exist_ok=True)
    for name, (draw, cap) in LOOSE.items():
        draw(str(d / f"{name}.png"))
        (d / f"{name}.txt").write_text(cap, encoding="utf-8")


# ========================================================================================== CSV
PROCS = {"Fast roughing": ["FR-01", "FR-02"], "Roughing": ["F-01", "F-02", "F-03", "F-04"],
         "Scavenging": ["S-01", "S-02", "S-03"], "Cleaning": ["C-01", "C-02", "C-03"], "Tailing": ["P-01", "P-02"]}
STATUS = [("Normal operation", "No abnormalities", 11),
          ("Normal operation", "Liquid level exceeds set value by 0.05m", 1),
          ("Motor noise is relatively loud", "Check motor bearings", 2),
          ("Insufficient flow rate", "Pipeline may have slight blockage", 2),
          ("Froth layer too thin", "Increase frother dosage slightly", 2),
          ("Aeration rate low", "Adjust air blower valve", 2),
          ("Foam color is dark", "Reduce collector dosage", 1),
          ("Pulp level is high", "Reduce slurry pump speed", 1),
          ("Impeller vibration detected", "Inspect impeller and shaft coupling", 1)]


def write_csv():
    rnd = random.Random(SEED)
    d = RAW / "csv"
    d.mkdir(parents=True, exist_ok=True)
    pool = [(s, p) for s, p, w in STATUS for _ in range(w)]
    rows = [("Time", "Process", "Equipment", "Operating status", "Problem description")]
    minute = 8 * 60
    for i in range(150):                                              # safety inspection records, Fig. 3 format
        proc = list(PROCS)[i % len(PROCS)]
        st, pr = rnd.choice(pool)
        day, mm = 3 + (minute // (24 * 60)), minute % (24 * 60)
        rows.append((f"2025/2/{day} {mm // 60}:{mm % 60:02d}", proc, rnd.choice(PROCS[proc]), st, pr))
        minute += 10
    with open(d / "inspection_records.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    logs = [("Time", "Process", "pH value", "Aeration rate (m3/min)", "Pulp concentration (%)",
             "Reagent dosage (g/t)", "Concentrate grade (%)", "Recovery rate (%)")]
    for h in range(72):                                               # production log (operation record)
        ph = round(rnd.uniform(10.4, 12.0), 1)
        air = round(rnd.uniform(0.6, 1.0), 2)
        grade = round(52 + 12 * (1 - abs(ph - 11.5) / 1.5) - 6 * (air - .8) + rnd.uniform(-1, 1), 1)
        rec = round(80 + 10 * (1 - abs(ph - 11.5) / 1.5) + 8 * (air - .8) + rnd.uniform(-1.5, 1.5), 1)
        logs.append((f"2025/2/{3 + h // 24} {h % 24}:00", rnd.choice(list(PROCS)[:4]), ph, air,
                     round(rnd.uniform(28, 36), 1), rnd.randint(30, 60), grade, rec))
    with open(d / "production_log.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(logs)


# ========================================================================================== PDFs
def _styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    st = getSampleStyleSheet()
    st.add(ParagraphStyle("Eq", parent=st["BodyText"], alignment=1, fontName="Helvetica-Oblique", spaceBefore=4, spaceAfter=6))
    st.add(ParagraphStyle("Cap", parent=st["BodyText"], fontName="Helvetica-Oblique", fontSize=9, spaceAfter=8))
    st.add(ParagraphStyle("Note", parent=st["BodyText"], fontSize=7, textColor="#777777"))
    st["BodyText"].fontSize = 10
    return st


def _vector_chart():
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    d = Drawing(330, 190)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 40, 30, 270, 130
    lp.data = [[(0.6, 54.0), (0.7, 56.5), (0.8, 58.0), (0.9, 56.2), (1.0, 52.9)],
               [(0.6, 80.0), (0.7, 84.5), (0.8, 89.0), (0.9, 91.0), (1.0, 92.0)]]
    lp.lines[0].strokeWidth = 1.6
    lp.lines[1].strokeWidth = 1.6
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax = 0.6, 1.0
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = 50, 95
    d.add(lp)
    d.add(String(120, 8, "Aeration rate (m3/min)", fontSize=8))
    d.add(String(48, 172, "Grade (%) and recovery rate (%)", fontSize=8))
    return d


def build_pdf(path, title, blocks, tmp):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from PIL import Image as PILImage
    st = _styles()
    flow = [Paragraph(title, st["Title"]), Paragraph(NOTE, st["Note"]), Spacer(1, 8)]
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            flow += [Spacer(1, 6), Paragraph(b[1], st["Heading1"])]
        elif kind == "h2":
            flow.append(Paragraph(b[1], st["Heading2"]))
        elif kind == "p":
            flow += [Paragraph(b[1], st["BodyText"]), Spacer(1, 3)]
        elif kind == "eq":
            flow.append(Paragraph(b[1], st["Eq"]))
        elif kind == "table":
            _, cap, rows = b
            t = Table(rows, hAlign="LEFT")
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.7, colors.black), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
            flow += [Paragraph(cap, st["Cap"]), t, Spacer(1, 10)]
        elif kind == "fig":
            _, drawer, cap, width = b
            p = tmp / f"fig_{abs(hash((str(path), cap))) % 10**8}.png"
            drawer(str(p))
            w, h = PILImage.open(p).size
            flow += [Spacer(1, 4), Image(str(p), width=width, height=width * h / w), Paragraph(cap, st["Cap"])]
        elif kind == "vfig":
            _, cap = b
            flow += [Spacer(1, 4), _vector_chart(), Paragraph(cap, st["Cap"])]
    SimpleDocTemplate(str(path), pagesize=A4, title=title, author="synthetic").build(flow)


def documents(tmp):
    D = {}
    D["academic_paper_flotation_kinetics.pdf"] = ("Effect of pH Value and Aeration Rate on Lead-Zinc Flotation Kinetics", [
        ("h1", "Abstract"),
        ("p", "This paper investigates how the pH value, the aeration rate and the collector dosage affect the flotation kinetics of galena and sphalerite in a lead-zinc ore. Batch tests show that a pH value of 11.5 keeps pyrite depressed and increases concentrate grade. A higher aeration rate increases the froth depth and the recovery rate but decreases concentrate grade when gangue is entrained."),
        ("h1", "1 Introduction"),
        ("p", "Froth flotation is the main process for separating galena and sphalerite from lead-zinc ore. Xanthate acts as the collector and MIBC acts as the frother in the rougher circuit. Copper sulfate activates sphalerite before the zinc circuit, and lime raises the pH value and depresses pyrite."),
        ("p", "Industrial plants use fast roughing, roughing, scavenging and cleaning stages to raise the recovery rate and the concentrate grade. The scavenging stage treats the roughing tailing, and the cleaning stage upgrades the rougher concentrate."),
        ("h1", "2 Materials and methods"),
        ("p", "Batch flotation tests used a laboratory flotation cell with an impeller speed of 1200 rpm. The pulp concentration was kept at 30 percent solids and the flotation time was kept constant at 8 minutes. The recovery rate was calculated with Equation (1), and the kinetic behaviour was described by the first-order model in Equation (2)."),
        ("eq", "Recovery = (C × c) / (F × f) × 100%     (1)"),
        ("eq", "R(t) = Rmax × (1 - exp(-k × t))     (2)"),
        ("p", "Here C is the concentrate mass, c is the concentrate grade, F is the feed mass, f is the feed grade, k is the rate constant and t is the flotation time."),
        ("h1", "3 Results and discussion"),
        ("p", "Table 1 lists the recovery rate and the concentrate grade at five pH values. The lead recovery rate increases when the pH value rises from 9.5 to 11.5 and decreases at a pH value of 12.0."),
        ("table", "Table 1. Lead recovery rate, lead grade and zinc recovery rate at different pH values.",
         [["pH value", "Pb recovery (%)", "Pb grade (%)", "Zn recovery (%)"], ["9.5", "71.2", "48.1", "12.4"],
          ["10.5", "80.5", "52.3", "9.6"], ["11.0", "86.0", "55.7", "7.8"], ["11.5", "91.3", "58.4", "5.9"],
          ["12.0", "84.1", "56.0", "5.1"]]),
        ("fig", img_chart_ph, "Fig. 1. Lead and zinc recovery rate versus pH value in batch flotation tests.", 280),
        ("p", "A higher aeration rate increases the froth depth, and a deeper froth layer increases concentrate grade but decreases the recovery rate of fine particles. Excess collector dosage reduces the selectivity because xanthate also adsorbs on pyrite."),
        ("fig", img_chart_kinetics, "Fig. 2. Recovery rate versus flotation time with the fitted first-order curve.", 280),
        ("p", "The fitted rate constant k is 0.55 per minute and the maximum recovery rate Rmax is 92 percent. A higher impeller speed increases the rate constant but also increases the entrainment of gangue."),
        ("h1", "4 Conclusions"),
        ("p", "A pH value of 11.5 gives the best balance between concentrate grade and recovery rate. The aeration rate and the collector dosage must be controlled together to keep the froth stable."),
    ])
    D["industry_standard_flotation_operation.pdf"] = ("Technical Specification for Operating Lead-Zinc Flotation Plants", [
        ("h1", "1 Scope"),
        ("p", "This specification defines operating requirements for the fast roughing, roughing, scavenging and cleaning stages of a lead-zinc flotation plant. It applies to the control of pH value, aeration rate, pulp concentration and reagent dosage."),
        ("h1", "2 Terms and definitions"),
        ("p", "Recovery rate is the percentage of valuable metal in the feed that reports to the concentrate. Concentrate grade is the content of valuable metal in the concentrate. Froth depth is the height of the froth layer above the pulp level."),
        ("h1", "3 Technical requirements"),
        ("p", "The pH value in the roughing stage shall be kept between 10.5 and 11.5 with lime. The pulp concentration shall be kept between 30 and 35 percent solids. The aeration rate shall be adjusted so that the froth layer remains stable."),
        ("table", "Table 1. Operating limits for each flotation stage.",
         [["Stage", "pH value", "Pulp concentration (%)", "Aeration rate (m3/min)", "Xanthate dosage (g/t)"],
          ["Fast roughing", "10.5-11.0", "32-35", "0.8-1.0", "40-60"], ["Roughing", "10.5-11.5", "30-35", "0.7-0.9", "30-50"],
          ["Scavenging", "10.5-11.0", "28-32", "0.6-0.8", "20-30"], ["Cleaning", "11.0-12.0", "20-25", "0.5-0.7", "10-20"]]),
        ("p", "The recovery rate of each stage shall be calculated with Equation (1) and reported every shift."),
        ("eq", "R = (C × c) / (F × f) × 100%     (1)"),
        ("h1", "4 Inspection and monitoring"),
        ("p", "Operators shall inspect each flotation cell every ten minutes and record the operating status, the equipment number and any abnormality. A level sensor shall measure the pulp level in every cell. The slurry pump shall adjust the tailing discharge to keep the pulp level stable."),
        ("p", "If the foam color becomes dark or the foam size becomes large, the operator shall reduce the collector dosage and check the aeration rate."),
        ("table", "Table 2. Inspection frequency.",
         [["Item", "Equipment", "Frequency"], ["Pulp level", "Level sensor", "Continuous"], ["Motor noise", "Agitator motor", "Every 10 min"],
          ["Froth color", "Froth camera", "Every 10 min"], ["Pipeline flow", "Slurry pump", "Every 30 min"]]),
    ])
    D["equipment_manual_flotation_cell.pdf"] = ("Operation and Maintenance Manual: Mechanical Flotation Cell", [
        ("h1", "1 Product overview"),
        ("p", "The mechanical flotation cell consists of a tank, an agitator with an impeller, an air inlet, a froth scraper, an overflow launder and a level sensor. The agitator disperses air into fine bubbles and keeps the particles in suspension."),
        ("fig", lambda p: img_cell(p), "Fig. 1. Structure of the mechanical flotation cell.", 250),
        ("h1", "2 Technical specification"),
        ("table", "Table 1. Main technical parameters of the flotation cell.",
         [["Parameter", "Value", "Unit"], ["Effective volume", "8", "m3"], ["Motor power", "15", "kW"],
          ["Impeller speed", "180-240", "rpm"], ["Air flow range", "0.5-1.2", "m3/min"], ["Pulp level set value", "0.40", "m"]]),
        ("p", "The air blower supplies the aeration rate to the cell. A higher impeller speed makes the foam size smaller, and a lower impeller speed makes the foam size larger."),
        ("h1", "3 Operation"),
        ("p", "Before start-up the operator checks the pulp level and the air valve. The level sensor measures the pulp level and the slurry pump adjusts the tailing discharge to keep the level stable. The froth scraper removes the froth layer into the launder."),
        ("h1", "4 Maintenance"),
        ("table", "Table 2. Maintenance schedule.",
         [["Component", "Task", "Interval"], ["Agitator motor", "Check bearings and motor noise", "Weekly"], ["Impeller", "Inspect wear and coupling", "Monthly"],
          ["Air blower", "Clean filter and adjust valve", "Monthly"], ["Level sensor", "Calibrate", "Every 3 months"]]),
        ("h1", "5 Troubleshooting"),
        ("table", "Table 3. Common faults, causes and actions.",
         [["Fault", "Cause", "Action"], ["Motor noise is loud", "Bearing wear", "Replace the motor bearings"],
          ["Froth layer too thin", "Aeration rate is low", "Increase the air flow"], ["Level exceeds set value", "Tailing pipeline blockage", "Clear the pipeline"],
          ["Impeller vibration", "Loose shaft coupling", "Tighten the coupling"]]),
        ("p", "A blocked tailing pipeline reduces the flow rate and may cause the level in the flotation cell to exceed the set value. Motor noise at the scavenging cell often indicates bearing wear."),
    ])
    D["accident_report_pipeline_blockage.pdf"] = ("Accident Investigation Report: Tailing Pipeline Blockage and Cell Overflow", [
        ("h1", "1 Summary"),
        ("p", "A blockage in the tailing pipeline caused the pulp level in a scavenging cell to exceed the set value. The cell overflowed and slurry reached the floor. No person was injured and the line was restarted after six hours."),
        ("h1", "2 Sequence of events"),
        ("table", "Table 1. Timeline of the event.",
         [["Time", "Event"], ["8:20", "Operator noticed that the flow rate in the tailing pipeline decreased"], ["8:30", "Level sensor reported that the pulp level exceeded the set value"],
          ["8:40", "Cell S-01 overflowed into the drain"], ["9:10", "Feed was stopped and the pipeline was cleared"]]),
        ("fig", lambda p: img_flowsheet(p, detailed=True), "Fig. 1. Plant flowsheet showing the position of the blocked tailing line.", 330),
        ("h1", "3 Cause analysis"),
        ("p", "The slurry pump speed was too low, so the settled solids blocked the tailing pipeline. A higher pulp concentration increased the settling rate and caused the blockage. The level alarm was set too high and gave a late warning."),
        ("h1", "4 Corrective actions"),
        ("p", "The operator shall reduce the pulp concentration when the flow rate falls. The maintenance team shall increase the slurry pump speed and flush the pipeline every shift. The level alarm shall be lowered so that the operator is warned before the level exceeds the set value."),
        ("eq", "Q = v × A     (1)"),
        ("p", "In Equation (1) Q is the flow rate, v is the slurry velocity and A is the pipe cross-section; a lower slurry velocity reduces the flow rate and increases settling."),
    ])
    D["production_regulations.pdf"] = ("Production Regulations for the Flotation Workshop", [
        ("h1", "1 General rules"),
        ("p", "All operators shall follow the reagent addition rules and shall record the operating status of every cell. The shift leader shall review the inspection records and report any abnormality."),
        ("h1", "2 Reagent addition"),
        ("p", "Lime shall be added to the ball mill to raise the pH value and to depress pyrite. Copper sulfate shall be added to activate sphalerite before the zinc circuit. Xanthate is used as the collector and pine oil or MIBC is used as the frother."),
        ("table", "Table 1. Reagent dosage rules.",
         [["Reagent", "Function", "Addition point", "Dosage (g/t)"], ["Lime", "Regulates pH value and depresses pyrite", "Ball mill", "800-1200"],
          ["Copper sulfate", "Activates sphalerite", "Zinc conditioning tank", "250-400"], ["Xanthate", "Collector", "Roughing cell", "30-50"],
          ["MIBC", "Frother", "Roughing cell", "15-25"], ["Sodium silicate", "Depresses gangue", "Cleaning cell", "100-200"]]),
        ("p", "Excess pine oil causes a viscous froth that lowers concentrate grade because gangue is entrained. Excess collector dosage makes the foam color dark."),
        ("h1", "3 Control of the process"),
        ("p", "If the flotation time is kept constant, a higher pulp concentration reduces the residence time of particles and the recovery rate falls. If the aeration rate is low, the froth layer becomes thin and the production efficiency decreases."),
        ("h1", "4 Emergency measures"),
        ("p", "When the pulp level exceeds the set value, the operator shall stop the feed and check the tailing pipeline. When the motor noise is loud, the operator shall stop the agitator and inform the maintenance team."),
    ])
    D["technical_report_leadzinc_plant.pdf"] = ("Technical Report: Performance of the Lead-Zinc Flotation Circuit", [
        ("h1", "1 Process description"),
        ("p", "The raw ore is ground and mixed with water in the mixing bowl to form a slurry. The slurry flows to fast roughing, roughing and scavenging. The roughing concentrate is upgraded in three cleaning stages and the scavenging tailing leaves the plant as final tailing."),
        ("fig", lambda p: img_flowsheet(p, detailed=True), "Fig. 1. Flowsheet of the lead-zinc flotation circuit.", 340),
        ("h1", "2 Metallurgical performance"),
        ("table", "Table 1. Mass balance of the flotation circuit.",
         [["Product", "Yield (%)", "Pb grade (%)", "Pb recovery (%)"], ["Feed", "100.0", "3.10", "100.0"], ["Lead concentrate", "4.6", "58.20", "86.4"],
          ["Final tailing", "95.4", "0.44", "13.6"]]),
        ("p", "The lead concentrate grade is 58.2 percent and the lead recovery rate is 86.4 percent. The recovery rate is calculated with Equation (1)."),
        ("eq", "Recovery = (C × c) / (F × f) × 100%     (1)"),
        ("vfig", "Fig. 2. Concentrate grade and recovery rate versus aeration rate (plant data)."),
        ("p", "As the aeration rate rises from 0.6 to 0.9 m3/min the recovery rate increases, while the concentrate grade decreases above 0.8 m3/min because more gangue is entrained."),
        ("h1", "3 Froth observation"),
        ("p", "Under normal operating conditions the bubbles are small and uniform and the foam color is lead gray. A dark foam color with large bubbles indicates an over-flotation state and is corrected by reducing the collector dosage."),
        ("fig", lambda p: img_froth(p, (110, 110, 115), 220, 22, (150, 150, 158), 11), "Fig. 3. Froth image of the roughing cell under normal operating conditions.", 230),
        ("h1", "4 Recommendations"),
        ("p", "The operator should keep the aeration rate near 0.8 m3/min and keep the pH value near 11.5. The cleaning stage should use sodium silicate to improve concentrate grade."),
    ])
    return D


def write_pdfs():
    d = RAW / "pdf"
    if d.exists():
        for f in d.glob("*.pdf"):
            f.unlink()
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, (title, blocks) in documents(tmp).items():
            build_pdf(d / name, title, blocks, tmp)


# ========================================================================================== HTML
def write_html():
    d = RAW / "html"
    d.mkdir(parents=True, exist_ok=True)
    img_feeder(str(RAW / "images" / "equipment_reagent_feeder.png"))
    (RAW / "images" / "equipment_reagent_feeder.txt").unlink(missing_ok=True)         # captioned inside the HTML instead
    (d / "maintenance_bulletin.html").write_text("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Maintenance Bulletin: Reagent Dosing System</title></head>
<body>
<p><em>SYNTHETIC SAMPLE DOCUMENT generated for software testing. It is not a real bulletin.</em></p>
<h1>Maintenance Bulletin: Reagent Dosing System</h1>
<h2>1 Overview</h2>
<p>The reagent dosing system supplies xanthate from the xanthate tank to the roughing cell through a metering pump and a flow meter. The controller adjusts the pump speed to keep the reagent dosage stable.</p>
<figure><img src="../images/equipment_reagent_feeder.png" alt="reagent dosing system">
<figcaption>Fig. 1. Equipment diagram of the reagent dosing system from the xanthate tank to the flotation cell.</figcaption></figure>
<h2>2 Known faults</h2>
<table><caption>Table 1. Faults of the reagent dosing system.</caption>
<tr><th>Fault</th><th>Cause</th><th>Action</th></tr>
<tr><td>Dosage too low</td><td>Metering pump wear</td><td>Replace the pump diaphragm</td></tr>
<tr><td>Dosage unstable</td><td>Air in the dosing line</td><td>Bleed the dosing line</td></tr>
<tr><td>Flow meter reading drifts</td><td>Scaling on the sensor</td><td>Clean the flow meter</td></tr>
</table>
<h2>3 Calculation</h2>
<div class="formula">Dosage (g/t) = reagent flow (g/min) / ore feed rate (t/min)     (1)</div>
<p>A low reagent dosage reduces the recovery rate, and an excess reagent dosage makes the froth viscous and lowers concentrate grade.</p>
</body></html>""", encoding="utf-8")


def main():
    from data.generate_sample_data import write_training
    write_loose_images()
    write_csv()
    write_pdfs()
    write_html()
    write_training()
    n = {k: len(list((RAW / k).glob("*.*"))) for k in ("pdf", "html", "csv", "images")}
    print(f"synthetic dataset written: {n} files in data/raw/*, plus data/training/flotation_train.jsonl")


if __name__ == "__main__":
    main()
