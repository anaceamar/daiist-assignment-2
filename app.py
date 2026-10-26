"""
Gradio dashboard for the varroa detector.

Loads the committed weights in weights/ (only to run on uploaded images) and the
artifacts train.py wrote. Nothing is trained here. Every confusion matrix,
cost and review share is recomputed from the saved test predictions.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw

from train import (ARTIFACTS, THRESHOLDS, WEIGHTS_FILE, cost_per_1000, default_costs,
                   image_scores, object_confusion, outcome_shares, staffing)

# ----------------------------------------------------------------- style ---
SURFACE = "#14120e"
PANEL = "rgba(36, 31, 22, 0.78)"
INK, MUTED = "#f2ede3", "#a9a08f"
GRID = "rgba(255,255,255,0.07)"
HONEY = "#f0a92b"                      # page chrome only, never data
PASS, REVIEW, FAIL = "#199e70", "#c98500", "#d95926"   # always shown with a label
SERIES = {"finetuned": "#3987e5", "coco": "#d95926", "nothing": "#8a8579"}

TEMPLATE = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
    title=dict(font=dict(family="Fraunces, Georgia, serif", size=19, color=INK), x=0.01),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, automargin=True,
               tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, automargin=True,
               tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h",
                y=1.02, yanchor="bottom", x=1, xanchor="right"),
    hoverlabel=dict(bgcolor="#2a241a", bordercolor="#3d3424", font=dict(color=INK)),
    margin=dict(l=60, r=24, t=72, b=52)))


def styled(fig, height=400):
    fig.update_layout(template=TEMPLATE, height=height)
    return fig


# ------------------------------------------------------------------ data ---
METRICS = json.loads((ARTIFACTS / "metrics.json").read_text())
STATS = json.loads((ARTIFACTS / "dataset_stats.json").read_text())
SWEEP = pd.read_csv(ARTIFACTS / "cost_sweep.csv")
PREDS = json.loads((ARTIFACTS / "predictions_test.json").read_text())
NAMES = METRICS["classes"]
POS = METRICS["positive_id"]
Y, S = image_scores(PREDS, POS, POS)
CHOSEN = METRICS["chosen"]
COSTS = METRICS["costs"]

_model = None


def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS_FILE))
    return _model


def costs_from(fn, fp, tp, base_rate):
    c = default_costs()
    c.update({"fn": fn, "fp": fp, "tp": tp, "base_rate": base_rate / 100})
    return c


# ---------------------------------------------------------- band view ---
def band_view(lo, hi, fn, fp, tp, base_rate):
    lo, hi = min(lo, hi), max(lo, hi)
    c = costs_from(fn, fp, tp, base_rate)
    sh = outcome_shares(Y, S, lo, hi)
    res = cost_per_1000(sh, c)
    staff = staffing(res["review_share"])

    npos, nneg = int(Y.sum()), int((Y == 0).sum())
    counts = [[round(sh["pos"][k] * npos) for k in ("pass", "review", "fail")],
              [round(sh["neg"][k] * nneg) for k in ("pass", "review", "fail")]]
    cm = go.Figure(go.Heatmap(
        z=counts, x=["pass (no action)", "human review", "fail (treat hive)"],
        y=["mite present", "no mite"], text=counts, texttemplate="%{text}",
        textfont=dict(size=22, color=INK), colorscale=[[0, "#211c14"], [1, "#2a78d6"]],
        showscale=False, xgap=3, ygap=3))
    cm.update_layout(title=f"Test images by outcome (review band {lo:.2f}–{hi:.2f})",
                     yaxis_autorange="reversed")

    do_nothing = 1000 * c["base_rate"] * c["fn"]
    tiles = (
        tile("Expected cost", f"€{res['cost_per_1000']:,.0f}", "per 1,000 images")
        + tile("Do nothing", f"€{do_nothing:,.0f}", "per 1,000 images")
        + tile("Sent to a person", f"{res['review_share']:.0%}",
               f"{staff['review_cases_per_day']:.0f} images a day")
        + tile("Reviewers needed", f"{staff['people_needed']:.2f}",
               f"{staff['review_hours_per_day']:.1f} h a day at "
               f"{COSTS['review_seconds']} s each")
        + tile("Mites caught", f"{sh['pos']['fail']:.0%} auto + {sh['pos']['review']:.0%} review",
               f"{sh['pos']['pass']:.0%} slip through")
    )
    obj = object_confusion(PREDS, len(NAMES), hi)
    labels = NAMES + ["background"]
    ocm = go.Figure(go.Heatmap(
        z=obj, x=[f"pred: {n}" for n in labels], y=[f"true: {n}" for n in labels],
        text=obj, texttemplate="%{text}", textfont=dict(size=18, color=INK),
        colorscale=[[0, "#211c14"], [1, "#2a78d6"]], showscale=False, xgap=3, ygap=3))
    ocm.update_layout(title=f"Box-level confusion matrix at confidence {hi:.2f} (IoU ≥ 0.5)",
                      yaxis_autorange="reversed")
    return styled(cm, 340), f'<div class="tiles">{tiles}</div>', styled(ocm, 360)


def tile(label, value, note):
    return (f'<div class="tile"><div class="tile-label">{label}</div>'
            f'<div class="tile-value">{value}</div><div class="tile-note">{note}</div></div>')


# ------------------------------------------------------------ cost curve ---
def cost_curve(fn, fp, tp, base_rate):
    c = costs_from(fn, fp, tp, base_rate)
    curve = [cost_per_1000(outcome_shares(Y, S, t, t), c)["cost_per_1000"] for t in THRESHOLDS]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=THRESHOLDS, y=curve, mode="lines", name="finetuned model, test",
                             line=dict(color=SERIES["finetuned"], width=2.5)))
    if c == {**default_costs()}:
        fig.add_trace(go.Scatter(x=SWEEP["threshold"], y=SWEEP["cost_test_coco"], mode="lines",
                                 name="COCO zero-shot, test",
                                 line=dict(color=SERIES["coco"], width=2)))
    fig.add_hline(y=1000 * c["base_rate"] * c["fn"], line_dash="dot", line_color=SERIES["nothing"],
                  annotation_text="do nothing", annotation_font_color=MUTED)
    t = CHOSEN["single_threshold"]
    fig.add_vline(x=t, line_dash="dash", line_color=HONEY,
                  annotation_text=f"chosen on validation: {t:.2f}",
                  annotation_font_color=HONEY)
    best = float(THRESHOLDS[int(np.argmin(curve))])
    fig.update_layout(title=f"Cost per 1,000 images vs. alert threshold (test minimum at {best:.2f})",
                      xaxis_title="Confidence threshold for 'mite present'",
                      yaxis_title="€ per 1,000 images")
    return styled(fig, 420)


# ---------------------------------------------------------------- tables ---
def per_class_table():
    return pd.DataFrame(METRICS["per_class_test"]).rename(columns={
        "class": "Class", "precision": "Precision", "recall": "Recall",
        "ap50": "AP50", "ap50_95": "AP50-95"})


def per_class_chart():
    d = pd.DataFrame(METRICS["per_class_test"])
    fig = go.Figure()
    for col, name, color in (("precision", "Precision", "#3987e5"),
                             ("recall", "Recall", "#d95926"),
                             ("ap50", "AP50", "#199e70")):
        fig.add_trace(go.Bar(x=d["class"], y=d[col], name=name, marker_color=color,
                             text=[f"{v:.2f}" for v in d[col]], textposition="outside"))
    fig.update_layout(title="Per-class results on test", barmode="group",
                      yaxis_range=[0, 1.1], bargap=0.3)
    return styled(fig, 380)


def baselines_table():
    il = METRICS["image_level_test"]
    o = METRICS["overall_test"]
    alt = METRICS["alternatives_test"]
    pos = next(r for r in METRICS["per_class_test"] if r["class"] == NAMES[POS])
    return pd.DataFrame([
        {"Model": "Finetuned YOLO11n", "mAP50 (all)": o["map50"], f"AP50 {NAMES[POS]}": pos["ap50"],
         "Image AUC": il["auc_finetuned"], "€ / 1,000 at its threshold":
             round(alt["model_alone"]["cost_per_1000"], 1)},
        {"Model": "COCO YOLO11n zero-shot", "mAP50 (all)": 0.0, f"AP50 {NAMES[POS]}": 0.0,
         "Image AUC": il["auc_coco_zero_shot"], "€ / 1,000 at its threshold":
             round(alt["coco_zero_shot_alone"]["cost_per_1000"], 1)},
        {"Model": "Predict nothing", "mAP50 (all)": 0.0, f"AP50 {NAMES[POS]}": 0.0,
         "Image AUC": 0.5, "€ / 1,000 at its threshold":
             round(alt["do_nothing"]["cost_per_1000"], 1)},
    ])


def alternatives_table():
    rows = []
    labels = {"do_nothing": "Do nothing", "person_checks_everything": "A person checks everything",
              "model_alone": "Model decides alone", "model_plus_review": "Model + review band (chosen)"}
    for key, label in labels.items():
        a = METRICS["alternatives_test"][key]
        rows.append({"Design": label, "€ / 1,000 images": round(a["cost_per_1000"], 1),
                     "Sent to review": f"{a['review_share']:.0%}",
                     "Reviewers needed": a["people_needed"]})
    return pd.DataFrame(rows)


def dataset_table():
    rows = []
    for split, d in STATS["splits"].items():
        row = {"Split": split, "Images": d["images"],
               f"Images with {NAMES[POS]}": f"{d['share_images_with_positive']:.0%}"}
        for n in NAMES:
            row[f"{n} boxes"] = d["instances"][n]
            side = d["object_size"][n]["median_side_pct"]
            row[f"{n} size"] = f"{side}% of side" if side is not None else "–"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- upload ---
def predict_upload(img, lo, hi):
    if img is None:
        return None, "Upload an image of bees to run the model."
    lo, hi = min(lo, hi), max(lo, hi)
    r = model().predict(img, conf=min(lo, 0.05) if lo > 0 else 0.01, verbose=False)[0]
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    score = 0.0
    for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                              r.boxes.conf.cpu().numpy()):
        cls = int(cls)
        if cls == POS:
            score = max(score, float(conf))
        if conf < lo:
            continue
        color = FAIL if cls == POS else "#3987e5"
        draw.rectangle(box.tolist(), outline=color, width=3)
        draw.text((box[0] + 3, box[1] + 2), f"{NAMES[cls]} {conf:.2f}", fill=color)
    outcome = ("FAIL: treat this hive" if score >= hi else
               "HUMAN REVIEW" if score >= lo else "PASS: no mite found")
    return pil, (f"**{outcome}**  \nHighest {NAMES[POS]} confidence: **{score:.2f}** "
                 f"(review band {lo:.2f}–{hi:.2f})")


# ----------------------------------------------------------------- page ---
def hero():
    alt = METRICS["alternatives_test"]
    chosen = alt["model_plus_review"]
    pos = next(r for r in METRICS["per_class_test"] if r["class"] == NAMES[POS])
    return f"""
    <div class="hero"><div class="comb"></div><div class="hero-inner">
      <div class="eyebrow">Apiary monitoring · hive-entrance camera</div>
      <h1>Is there a mite<br>on that bee?</h1>
      <p class="lede">A finetuned YOLO11n flags images that show the varroa mite. Each image
      is passed, sent to a person, or flagged for treatment, depending on two thresholds.
      Everything here is loaded from the saved model and its test predictions; nothing is
      retrained.</p></div></div>
    <div class="tiles">
      {tile(f"{NAMES[POS]} AP50", f"{pos['ap50']:.2f}", "test split")}
      {tile("Image-level AUC", f"{METRICS['image_level_test']['auc_finetuned']:.2f}",
            f"COCO zero-shot: {METRICS['image_level_test']['auc_coco_zero_shot']:.2f}")}
      {tile("Chosen design", f"€{chosen['cost_per_1000']:,.0f}",
            f"vs €{alt['do_nothing']['cost_per_1000']:,.0f} doing nothing, per 1,000")}
      {tile("Review load", f"{chosen['review_share']:.0%}",
            f"{chosen['people_needed']:.2f} people at {COSTS['images_per_day']:,} images a day")}
    </div>"""


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Inter:wght@400;500;600&display=swap');
body, gradio-app, .gradio-container { background: #14120e !important; color: #f2ede3;
  font-family: Inter, system-ui, sans-serif; }
.gradio-container { max-width: 1280px !important; }
.hero { position: relative; overflow: hidden; border-radius: 18px; padding: 44px 40px 38px;
  margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.08);
  background: radial-gradient(120% 140% at 85% 0%, rgba(240,169,43,0.30), transparent 55%),
              linear-gradient(135deg, #1d1810, #14120e 60%); }
.comb { position: absolute; inset: 0; opacity: .16;
  background-image: radial-gradient(circle at 50% 50%, transparent 58%, #f0a92b 60%, transparent 63%);
  background-size: 42px 36px; mask-image: linear-gradient(90deg, transparent 35%, #000 95%); }
.hero-inner { position: relative; }
.eyebrow { font: 600 12px Inter, sans-serif; letter-spacing: .18em; text-transform: uppercase;
  color: #f0a92b; }
.hero h1 { font: 800 clamp(38px, 6.5vw, 70px)/1.0 Fraunces, Georgia, serif; margin: 16px 0 14px;
  color: #f2ede3; }
.lede { max-width: 660px; color: #d6cfc1; font-size: 15px; line-height: 1.55; }
.tiles { display: grid; gap: 12px; margin: 6px 0 12px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
.tile { background: rgba(36,31,22,0.78); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 16px 18px; }
.tile-label { font: 600 11px Inter, sans-serif; letter-spacing: .14em; text-transform: uppercase;
  color: #a9a08f; }
.tile-value { font: 800 30px/1.15 Fraunces, Georgia, serif; color: #f2ede3; margin-top: 6px; }
.tile-note { font-size: 12.5px; color: #a9a08f; margin-top: 2px; }
[role="tab"] { font: 600 13px Inter, sans-serif !important; letter-spacing: .06em;
  text-transform: uppercase; color: #a9a08f !important; }
[role="tab"][aria-selected="true"] { color: #f2ede3 !important;
  border-bottom: 2px solid #f0a92b !important; }
.block, .form { background: rgba(36,31,22,0.78) !important;
  border-color: rgba(255,255,255,0.08) !important; border-radius: 14px !important; }
h2, h3 { font-family: Fraunces, Georgia, serif !important; }
.prose { padding: 4px 10px !important; color: #a9a08f; }
input[type="range"] { accent-color: #f0a92b; }
"""

THEME = gr.themes.Base(primary_hue=gr.themes.colors.amber, neutral_hue=gr.themes.colors.stone,
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    body_background_fill=SURFACE, body_background_fill_dark=SURFACE,
    body_text_color=INK, body_text_color_dark=INK,
    body_text_color_subdued=MUTED, body_text_color_subdued_dark=MUTED,
    block_background_fill=PANEL, block_background_fill_dark=PANEL,
    block_border_color="rgba(255,255,255,0.08)", block_border_color_dark="rgba(255,255,255,0.08)",
    block_label_text_color=MUTED, block_label_text_color_dark=MUTED,
    block_title_text_color=INK, block_title_text_color_dark=INK,
    input_background_fill="#211c14", input_background_fill_dark="#211c14",
    background_fill_primary=SURFACE, background_fill_primary_dark=SURFACE,
    background_fill_secondary="#211c14", background_fill_secondary_dark="#211c14",
    border_color_primary="rgba(255,255,255,0.08)", border_color_primary_dark="rgba(255,255,255,0.08)",
    table_even_background_fill="#1c1812", table_even_background_fill_dark="#1c1812",
    table_odd_background_fill="#18150f", table_odd_background_fill_dark="#18150f",
    slider_color=HONEY, slider_color_dark=HONEY,
    button_primary_background_fill=HONEY, button_primary_background_fill_dark=HONEY)

FORCE_DARK = "() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }"

with gr.Blocks(title="Varroa detector") as demo:
    gr.HTML(hero())
    with gr.Row():
        lo = gr.Slider(0, 1, value=CHOSEN["band_lo"], step=0.01,
                       label="Below this: PASS (no action)")
        hi = gr.Slider(0, 1, value=CHOSEN["band_hi"], step=0.01,
                       label="At or above this: FAIL (treat hive). In between: human review")
    with gr.Accordion("Cost assumptions (edit to test sensitivity)", open=False):
        with gr.Row():
            fn = gr.Number(value=COSTS["fn"], label="€ missed mite (false negative)")
            fp = gr.Number(value=COSTS["fp"], label="€ false alarm (false positive)")
            tp = gr.Number(value=COSTS["tp"], label="€ treating a real case (true positive)")
            br = gr.Number(value=COSTS["base_rate"] * 100, label="% of images with a mite (base rate)")

    with gr.Tab("Decision & cost"):
        tiles_html = gr.HTML()
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        gr.Markdown("### Chosen design against the alternatives (test split, default costs)")
        gr.Dataframe(alternatives_table())

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload a hive-entrance image")
            img_out = gr.Image(type="pil", label="Predictions at the current thresholds")
        verdict = gr.Markdown()

    with gr.Tab("Per-class results"):
        gr.Dataframe(per_class_table())
        gr.Plot(per_class_chart())
        obj_plot = gr.Plot()
        gr.Markdown("### Finetuned model vs. both baselines (same test split)")
        gr.Dataframe(baselines_table())

    with gr.Tab("Dataset"):
        gr.Markdown(f"Roboflow Universe: `{METRICS['dataset']['workspace']}/"
                    f"{METRICS['dataset']['project']}` version {METRICS['dataset']['version']}. "
                    "Object size = median box side as a share of the image side.")
        gr.Dataframe(dataset_table())

    band_inputs = [lo, hi, fn, fp, tp, br]
    band_outputs = [cm_plot, tiles_html, obj_plot]
    cost_inputs = [fn, fp, tp, br]
    for comp in band_inputs:
        comp.change(band_view, band_inputs, band_outputs)
    for comp in cost_inputs:
        comp.change(cost_curve, cost_inputs, curve_plot)
    for comp in (img_in, lo, hi):
        comp.change(predict_upload, [img_in, lo, hi], [img_out, verdict])
    demo.load(band_view, band_inputs, band_outputs)
    demo.load(cost_curve, cost_inputs, curve_plot)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_DARK)
