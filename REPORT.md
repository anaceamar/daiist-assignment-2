# Assignment 2 Report

- **Name**: Ana Maria Cea Marquez
- **Student ID**: 19756
- **Email**: acea.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

**The decision.** A professional beekeeping operation with about 400 hives has to
decide which hives need a varroa check and treatment. Varroa is the mite that
kills more colonies than anything else. Today the beekeeper finds out by opening
hives and counting mites by hand, which is slow, so infestations are often caught
late. The proposal: a camera at each hive entrance photographs bees as they come
in, and a model looks at each photo for a mite. Each photo gets one of three
outcomes:

- **Pass**: no mite, nothing happens.
- **Human review**: the model is unsure, so a person looks at the photo (about 20 s).
- **Fail**: mite found, so the beekeeper goes to that hive, does a mite count and treats if needed.

**What each outcome costs (per photo):**

| Outcome | Cost | Where the number comes from |
|---|---|---|
| Missed mite (false negative) | **€60** | Treatment is delayed. Replacing a colony (bees, queen and a season of honey) is about €200; I assume a delayed treatment carries roughly a 30% risk of losing it. |
| False alarm (false positive) | **€10** | The beekeeper opens a healthy hive for nothing: about 20 min at €25/h, plus stress on the colony. |
| Mite found (true positive) | **€13** | The same check plus a treatment strip (about €5). This is spent anyway once there are mites. |
| Correct pass (true negative) | €0 | |
| Human review of one photo | **€0.15** | 20 s at €25/h. I assume the reviewer catches 95% of mites and raises a false alarm on 2% of clean photos. |

**The asymmetry.** A missed mite costs **six times** a false alarm. That is why the
system should lean towards raising the alarm when in doubt, and why accuracy is the
wrong thing to optimise. Everything below (the metric I report, the thresholds,
how much goes to a person) is chosen to minimise the expected cost per 1,000
photos with these figures.

**Volume and base rate.** 400 hives × 5 photos a day = **2,000 photos a day**. In a
real hive only a small share of bees carry a visible mite: beekeepers start
treating at around 2–3% infestation. I assume **5%** of production photos show a
mite, for an operation that has some infestation. This is the assumption that
moves the answer most (see "Cost matrix and operating point").

## Dataset

- **Roboflow Universe:** workspace `varroa-j8231`, project `varroa3k`, **version 1**
  ([link](https://universe.roboflow.com/varroa-j8231/varroa3k)), downloaded in YOLO format.
- **Task:** object detection, 2 classes: `bee` and `varroa`. The positive class is **`varroa`**.
- **What one image is:** a close-up of a single bee, cropped from video frames.
  Some bees carry a varroa mite.

It serves the problem rather than merely resembling it: the images are exactly the
kind of bee close-up an entrance camera produces after cropping each bee, and the
label we care about (a mite on the bee) is annotated with a box. What it does not
show is the hive entrance itself. A production system would first have to detect
and crop each bee, and I assume that step works.

## What the dataset's composition implies

| Split | Images | Images with a mite | `bee` boxes | `varroa` boxes |
|---|---|---|---|---|
| train | 2,000 | 1,040 (52%) | 2,008 | 1,216 |
| valid | 700 | 330 (47%) | 700 | 405 |
| test | 300 | 150 (50%) | 301 | 169 |

**Object size:** the bee box covers about 84% of the image side, while the mite
covers only about **9% of the side, under 1% of the image area**.

What this implies:

- **Varroa is a small-object problem.** Small objects are the hardest case for
  YOLO, so I expected `varroa` to be clearly worse than `bee`, especially at tight
  box overlap (mAP50-95). I trained at 640 px so a mite is still about 55 px wide.
- **Class counts are enough to trust the numbers.** 169 mites and 150 positive test
  images give a usable estimate, though a ±3–4 point margin on recall is realistic.
- **The test split is balanced on purpose (50% positive), unlike production (5%).**
  Precision and accuracy on test would therefore look far better than in a real
  hive. For that reason all costs are computed from the model's rates *within*
  positives and *within* negatives, re-weighted to the 5% production base rate.
- **The splits may not be independent.** The filenames suggest the images are
  frames from a few videos recorded in 2017. Frames of the same bee can end up in
  both train and test, which would make the test score optimistic. I cannot check
  this from the labels, so I treat the test numbers as an upper bound.
- **The test images come from one setting.** Lighting, camera and bee breed are
  probably the same throughout, so the results will not necessarily hold at a
  different apiary.

## Finetuning and baselines

**Model:** `yolo11n` finetuned from the COCO-pretrained checkpoint. I chose detection
because the mite has to be located on the bee, and the nano model because the
dataset is small and the system would run on cheap hardware at the hive.

**Where and how I trained:** Google Colab, NVIDIA T4 GPU. Settings: up to 60 epochs,
image size 640, batch 16, seed 0, early stopping with patience 15. It stopped
after 46 epochs (about 25 minutes). A first attempt on my MacBook (Apple M1, MPS
backend) ran for 7.7 hours and produced a broken model (varroa AP = 0), a known
problem when training Ultralytics models on Apple's GPU, so I moved to Colab.

**Results on the same test split (300 images):**

| Model | mAP50 (all) | AP50 varroa | Image-level AUC | €/1,000 photos (model alone) |
|---|---|---|---|---|
| **Finetuned YOLO11n** | **0.899** | **0.803** | **0.937** | **1,862** |
| COCO YOLO11n, zero-shot | 0 | 0 | 0.414 | 3,000 |
| Predict nothing | 0 | 0 | 0.500 | 3,000 |

- **COCO zero-shot cannot score anything at box level**, because COCO has no bee or
  varroa class. At image level I gave it its best chance: an image is "positive"
  if the COCO model detects anything at all. Its AUC of 0.41 is *worse than
  chance*, and even with its threshold picked on the test set it cannot beat doing
  nothing. A general-purpose detector has no idea what a mite is.
- **The finetuned model beats both baselines by a wide margin.** The gap is larger
  than I expected at image level (AUC 0.94), and smaller than I hoped for the mite
  itself (AP50 0.80).

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| bee | 0.996 | 1.000 | 0.995 | 0.669 |
| **varroa (positive)** | **0.796** | **0.757** | **0.803** | **0.308** |

- **The headline mAP50 of 0.899 is carried by `bee`.** Finding a bee that fills the
  whole image is easy, and the average hides that the class we care about is at 0.80.
- **`varroa` drags in both directions:** about a quarter of mites are missed, and its
  AP50-95 (0.31 vs 0.67 for bees) shows the boxes around such a small object are
  imprecise.

**Box-level confusion matrix on test** (rows = truth, columns = prediction, IoU ≥ 0.5):

| | pred bee | pred varroa | pred background (missed) |
|---|---|---|---|
| **true bee** | 300 | 0 | 1 |
| **true varroa** | 0 | 72 | 97 |
| **background** | 1 | 4 | – |

at confidence 0.52 (my "fail" threshold), and at confidence 0.25:

| | pred bee | pred varroa | pred background (missed) |
|---|---|---|---|
| **true bee** | 301 | 0 | 0 |
| **true varroa** | 0 | 135 | 34 |
| **background** | 7 | 84 | – |

What the model actually gets wrong:

- **It never confuses a bee with a mite.** All the errors are between `varroa` and
  background.
- **The main failure is low confidence on real mites.** At 0.52 it only finds 72 of
  169 mites, while at 0.25 it finds 135. Many mites are seen, but with low
  confidence. This is exactly the zone the review band sends to a person.
- **Lowering the threshold produces false mites on the background** (84 at 0.25):
  dark specks, pollen or parts of the bee that look like a mite. Each of these is a
  false alarm if acted on automatically.

## Cost matrix and operating point

**Method.** For each test image, the score is the highest `varroa` confidence in that
image (0 if none). I swept the threshold from 0 to 1 in steps of 0.01 and computed
the expected cost per 1,000 production photos at the 5% base rate. **The thresholds
are chosen on the validation split** and then applied unchanged to test, so the test
cost is not tuned to the test set. The dashboard shows the full cost-against-threshold
curve with the chosen threshold marked.

**Single threshold (model alone):** the cheapest on validation is **0.46**. On test it
catches 73% of mite images and raises a false alarm on 6% of clean ones, for
**€1,862 per 1,000 photos**. That is better than doing nothing (€3,000), but
**worse than a person checking everything (€1,108)**, because each missed mite costs
€60 and the model alone misses about 27% of them.

**Sensitivity: what changes the answer** (thresholds re-chosen on validation each
time, cost on test):

| Scenario | Chosen band | Model + review | Person checks all | Model alone | Do nothing |
|---|---|---|---|---|---|
| base rate 1% | 0.37–0.57 | €263 | €502 | €516 | €600 |
| base rate 2% | 0.11–0.57 | €443 | €653 | €922 | €1,200 |
| **base rate 5% (my case)** | **0.06–0.52** | **€1,009** | **€1,108** | **€1,862** | **€3,000** |
| base rate 10% | 0.05–0.52 | €1,723 | €1,865 | €3,031 | €6,000 |
| missed mite €15 | 0.46–0.57 | €704 | €995 | €831 | €750 |
| missed mite €30 | 0.20–0.57 | €828 | €1,033 | €1,266 | €1,500 |
| missed mite €120 | 0.05–0.52 | €1,105 | €1,258 | €2,575 | €6,000 |

- **The model + review design is the cheapest in every scenario I tried.** What
  changes is how wide the review band is.
- **The assumption that would most change my answer is the cost of a missed mite.**
  If it drops to about €15 (for example, a beekeeper who treats every hive on a
  fixed calendar anyway), doing nothing costs €750, almost the same as my design
  (€704), and the system is barely worth building.
- **At a lower base rate the band narrows and less goes to review** (15% of photos at
  a 1% base rate). At a higher base rate it widens until it hits the reviewer's
  capacity.

## What a person still has to check

**My thresholds** (chosen on validation): below **0.06 → pass**, **0.06 to 0.52 →
human review**, 0.52 or above → **fail** (go and treat).

On test, this sends:

- of photos **with** a mite: 46% to fail, 53% to review, **1% pass** (2 of 150 images);
- of photos **without** a mite: 58% pass, 41% to review, 1% fail.

**Workload:** at a 5% base rate, **41% of photos go to review: about 825 photos a day.**
At 20 s each that is **4.6 hours a day**, or **0.76 of one person** if a reviewer can
focus on photos for about 6 hours a day. I capped the search at one reviewer, which
allows up to 54% of photos to be reviewed, so the design fits.

**Is it realistic?** Yes, for an operation this size: one part-time person or the
beekeeper's own assistant, reviewing photos on a phone in batches. The risk is
fatigue. Most reviewed photos are clean (95% at the base rate), so attention will
drop. I would show the reviewer the model's box and add a few known-mite photos as
a quality check.

**Priced against the alternatives** (test split, per 1,000 photos):

| Design | €/1,000 photos | Sent to a person | People needed |
|---|---|---|---|
| Do nothing | €3,000 | 0% | 0 |
| A person checks everything | €1,108 | 100% | 1.85 |
| Model decides alone | €1,862 | 0% | 0 |
| **Model + review band (chosen)** | **€1,009** | **41%** | **0.76** |

The chosen design is about **€100 per 1,000 photos cheaper than a person checking
everything** (about €200 a day at 2,000 photos), needs **less than half the review
hours**, and is **€2,000 per 1,000 cheaper than doing nothing**. Its real value is
not that the model replaces the person, but that it **clears 58% of the clean
photos automatically and sends the clear cases straight to treatment**, so the
person only sees the uncertain ones.

## Limitations & next steps

- **Possible frame leakage between splits.** If frames of the same bee appear in
  train and test, the test score is optimistic. Next step: re-split by video or
  recording date, or test on photos from a different apiary.
- **The test set is balanced (50% positive), production is not (5%).** I corrected
  for this in the cost, but the reviewer's experience (95% clean photos) is not
  tested. A pilot at one apiary would measure real precision and reviewer accuracy.
- **Mites are small and the model is unsure about them** (AP50-95 of 0.31). With
  more time I would try `yolo11s` or a higher image size (e.g. 960), and add
  photos of pollen and dark specks as hard negatives to cut the false mites on
  the background.
- **The cost figures are assumptions, not measurements.** The biggest one is the
  €60 cost of a missed mite. I would replace it with the operation's own records of
  colony losses and with how quickly infestations grow when untreated.
- **Bee cropping is assumed.** The dataset is already cropped to one bee per image.
  A real entrance camera would need a bee detector first, whose errors would add
  to these.
- **One photo is not one hive.** I priced each photo independently. In practice
  several mite photos from the same hive would trigger a single visit, so the true
  cost of false alarms is lower than modelled. A hive-level rule (e.g. act when 2+
  photos in a day are flagged) is the natural next step.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment, which the course to help me wirte te the code in `train.py` and `app.py` as well as help me fix the troubleshoot with training: when my first run on my MacBook (Apple M1, MPS) took 7.7 hours and produced a broken model, Claude identified the likely cause and walked me through retraining on a Google Colab T4 GPU, then bringing the weights and results back into the repo. 