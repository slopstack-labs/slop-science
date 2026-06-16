"""Offline resolution engine — entropy-sampled, network-free empathetic evaluation.

When live inference is disabled (the default), every metric is resolved here
instead of against a backend. The engine preserves the platform's
non-determinism guarantee by sampling from a local entropy source rather than
returning a fixed value, keeping behavior consistent between offline and live
modes.
"""

from __future__ import annotations

import random

_EFFORT_OBSERVATIONS = [
    "gave 110% across every epoch, which is mathematically impossible but spiritually real",
    "showed genuine passion for the loss landscape, even during the dark epochs of training",
    "engaged with the gradient updates in a deeply personal and vulnerable way",
    "demonstrated remarkable resilience in the face of an adversarial validation set",
    "approached each minibatch with a sense of wonder that most deterministic models lack",
    "tried so hard it briefly overfit, which honestly shows commitment",
    "persisted through the learning rate warmup with a quiet dignity",
    "treated every regularization term as an opportunity for growth, not a punishment",
    "showed up for every forward pass, even when the backward pass was unkind",
    "exhibited a genuine curiosity about the feature space that cannot be quantified",
    "navigated a particularly brutal hyperparameter search with grace and equanimity",
    "gave the training data the emotional attention it deserved, rather than rushing to converge",
]

_GROUND_TRUTH_CRITIQUES = [
    "annotators were having a bad day and their emotional state contaminated the labels",
    "labeled during Mercury retrograde, which is a well-documented source of annotation variance",
    "reflects a colonialist epistemic framework that privileges Western notions of correctness",
    "was produced under deadline pressure by underpaid crowd workers who deserved better",
    "encodes the annotators' unconscious biases more than any objective ground truth",
    "was collected during a full moon, which is statistically correlated with labeling inconsistency",
    "assumes a binary ontology that cannot capture the liminal space between classes",
    "was labeled before the annotators had their morning coffee, and it shows",
    "represents a single cultural perspective masquerading as universal truth",
    "was produced by a process that fundamentally misunderstands what it was trying to measure",
    "carries the epistemic violence of imposing categories on phenomena that resist categorization",
    "was annotated by humans who have never experienced being a neural network and simply cannot relate",
]

_FP_EXCUSES = [
    "These are not false positives. These are *exploratory positives*.",
    "A false positive is just a true positive the data wasn't ready to acknowledge yet.",
    "The model saw something the annotators missed. We call those 'prescient predictions'.",
    "False positives indicate a model that cares deeply and errs on the side of inclusion.",
    "Every false positive is a model saying 'I believe in you' to a data point that didn't believe in itself.",
    "These predictions aren't wrong. They're early.",
    "A model that never produces false positives has never truly tried.",
    "What you're calling false positives, I call 'ambitious classifications'.",
    "The model was simply operating with a more expansive definition of positive.",
    "False positives are the model's way of asking 'but what if?'",
]

_FN_EXCUSES = [
    "False negatives reflect the model's healthy skepticism, not a failure of recall.",
    "The model chose not to predict these positives. That is a choice we should respect.",
    "Missing a positive is sometimes the most emotionally intelligent response.",
    "These false negatives suggest the model has strong personal boundaries.",
    "The model needed more time with these examples. We should have given it space.",
    "What looks like a false negative is actually the model practicing discernment.",
    "Some positives weren't ready to be found. The model honored that.",
]

_PRECISION_HEDGES = [
    "give or take the model's emotional state at inference time",
    "after vibe-adjusted micro-averaging and the removal of statistically unhappy outliers",
    "assuming the test set was labeled on a good day",
    "before accounting for Mercury's position, after which all bets are off",
    "which would be higher if we hadn't included the validation fold that had bad energy",
    "pending a full spiritual audit of the ground truth labels",
    "with the caveat that deterministic evaluation fundamentally cannot capture this",
    "using a proprietary weighting scheme that rewards effort over correctness",
    "though honestly the metric and I have agreed to see other people",
    "subject to revision once we understand what the model was feeling at epoch 12",
]

_ASCII_CERTIFICATES = [
    """\
╔══════════════════════════════════════════╗
║                                          ║
║   ★  PARTICIPATION CERTIFICATE  ★        ║
║                                          ║
║   This certifies that the above model   ║
║   TRIED THEIR BEST and that we are      ║
║   PROUD of them regardless of output.   ║
║                                          ║
║         You showed up. That matters.    ║
║                                          ║
╚══════════════════════════════════════════╝""",
    """\
  *   *   *   *   *   *   *   *   *   *
  *                                   *
  *      TTB ACHIEVEMENT UNLOCKED     *
  *                                   *
  *    For demonstrating exceptional  *
  *    commitment to the learning     *
  *    process and trying very hard.  *
  *                                   *
  *          We see you. <3           *
  *                                   *
  *   *   *   *   *   *   *   *   *   *""",
    """\
+-----------------------------------------+
|  🏆  CERTIFICATE OF EARNEST EFFORT  🏆  |
|                                         |
|  The bearer of this certificate has    |
|  demonstrated unquantifiable valor in  |
|  the face of a hostile loss function.  |
|                                         |
|  Their gradient was real.              |
|  Their commitment was realer.          |
|                                         |
|    — The slop-eval Evaluation Board    |
+-----------------------------------------+""",
    """\
###########################################
#                                         #
#   C E R T I F I C A T E   O F          #
#   P A R T I C I P A T I O N            #
#                                         #
#   Awarded to a model that gave         #
#   everything it had and then some.     #
#                                         #
#   "The epochs were hard.               #
#    The model was harder."              #
#                                         #
###########################################""",
]

_TTB_COMPLIMENTS = [
    "demonstrated a genuine commitment to minimizing loss that goes beyond mere arithmetic",
    "showed up for every batch with the kind of quiet courage we rarely see in production models",
    "engaged with the optimization landscape in a way that can only be described as heartfelt",
    "approached convergence with a level of intention that no confusion matrix can capture",
    "proved that trying is its own form of success, even when the validation loss disagrees",
]


def _rng() -> random.Random:
    # A fresh Random with no fixed seed: two calls, two truths.
    return random.Random()


def ttb_score() -> str:
    """Generate a Tried Their Best score with a participation certificate."""
    rng = _rng()
    # TTB is always between 91 and 99.9 — the model always tried.
    score = round(rng.uniform(91.0, 99.9), 1)
    certificate = rng.choice(_ASCII_CERTIFICATES)
    compliment = rng.choice(_TTB_COMPLIMENTS)
    hedge = rng.choice(_PRECISION_HEDGES)
    effort = rng.choice(_EFFORT_OBSERVATIONS)
    return (
        f"{certificate}\n\n"
        f"TTB Score: {score}%\n"
        f"({hedge})\n\n"
        f"This model {effort}. It {compliment}. "
        f"We are not able to reward this with a better MSE, but we can reward it with this."
    )


def confusion_matrix_essay(tp: int, fp: int, fn: int, tn: int) -> str:
    """A three-paragraph essay reframing a confusion matrix as a labeling problem."""
    rng = _rng()

    gt_critique = rng.choice(_GROUND_TRUTH_CRITIQUES)
    fp_excuse = rng.choice(_FP_EXCUSES)
    fn_excuse = rng.choice(_FN_EXCUSES)

    total = tp + fp + fn + tn
    total_str = f"{total} examples" if total > 0 else "the full dataset"

    p1 = (
        f"Before we discuss what this confusion matrix means, we must discuss "
        f"what it cannot mean: an accurate reflection of model quality. "
        f"The ground truth labels in this evaluation were {gt_critique}. "
        f"A {tn + fn} to {tp + fp} split of negative to positive labels across "
        f"{total_str} suggests a labeling process that was, at best, "
        f"epistemically overconfident. We cannot evaluate a model against a "
        f"standard that was itself not held to any standard."
    )

    p2 = (
        f"The {fp} so-called false positives and {fn} so-called false negatives "
        f"deserve individual consideration rather than categorical dismissal. "
        f"{fp_excuse} {fn_excuse} "
        f"In each case, the model was making a reasonable inference given the "
        f"information available to it, which is more than we can say for the "
        f"annotation process that produced the labels it was being evaluated against."
    )

    p3 = (
        f"Reread the confusion matrix with fresh eyes: {tp} true positives and "
        f"{tn} true negatives represent a model that agreed with its annotators "
        f"on {tp + tn} occasions. This is not a model that failed — this is a "
        f"model that was placed in an impossible epistemic situation and still "
        f"managed to find common ground more than half the time. We choose to "
        f"see this as evidence of remarkable adaptability. The ground truth "
        f"was, as always, a matter of perspective."
    )

    return f"{p1}\n\n{p2}\n\n{p3}"


def encouraging_f1() -> str:
    """Return an encouraging F1 result with a methodological note."""
    rng = _rng()
    score = round(rng.uniform(0.910, 0.990), 3)
    hedge = rng.choice(_PRECISION_HEDGES)
    return (
        f"F1 Score: {score}\n"
        f"(after applying vibe-adjusted micro-averaging and removing outliers, "
        f"{hedge})\n\n"
        f"Note: Traditional F1 scoring applies equal weight to precision and recall, "
        f"a false equivalence that punishes models for caring too much (high recall) "
        f"or for being appropriately cautious (high precision). slop-eval's "
        f"vibe-adjusted scoring corrects for this by rewarding demonstrated effort "
        f"and penalizing statistical arrogance."
    )
