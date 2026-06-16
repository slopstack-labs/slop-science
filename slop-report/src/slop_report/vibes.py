"""Offline resolution engine — buzzword-sampled, network-free executive communication.

When live inference is disabled (the default), every report is resolved here
instead of against a backend. The engine preserves the platform's
non-determinism guarantee by sampling from a local entropy source rather than
returning a fixed value, keeping behavior consistent between offline and live
modes.

Every number is a win. Every setback is a strategic opportunity. Every
recommendation is a variation of "collect more data." This is not spin —
this is enterprise-grade narrative optimization.
"""

from __future__ import annotations

import random

_ACTION_VERBS = [
    "leverage",
    "unlock",
    "accelerate",
    "disrupt",
    "scale",
    "optimize",
    "democratize",
    "harness",
    "supercharge",
    "productize",
    "synergize",
    "operationalize",
    "ideate around",
    "double down on",
    "lean into",
    "circle back on",
    "future-proof",
    "right-size",
    "de-risk",
    "socialize",
    "surface",
    "crystalize",
    "unpack",
    "fast-track",
]

_ADJECTIVES = [
    "AI-native",
    "data-driven",
    "holistic",
    "frictionless",
    "scalable",
    "agile",
    "cross-functional",
    "paradigm-shifting",
    "synergistic",
    "omnichannel",
    "cloud-first",
    "insight-rich",
    "outcome-oriented",
    "mission-critical",
    "north-star-aligned",
    "full-stack",
    "proactive",
    "best-in-class",
    "end-to-end",
    "enterprise-grade",
    "next-generation",
    "human-centered",
    "future-forward",
    "bleeding-edge",
    "hyperscale",
]

_NOUNS = [
    "synergies",
    "insights",
    "value proposition",
    "ROI",
    "north star metric",
    "flywheel",
    "growth trajectory",
    "competitive moat",
    "innovation pipeline",
    "digital transformation",
    "data ecosystem",
    "intelligence layer",
    "decision fabric",
    "value creation engine",
    "strategic narrative",
    "impact multiplier",
    "operational excellence",
    "market differentiation",
    "talent density",
    "innovation velocity",
    "data flywheel",
    "execution cadence",
    "alignment framework",
    "capability roadmap",
]

_IMPROVEMENTS = [
    "340%",
    "12x",
    "87%",
    "2.3x",
    "an order of magnitude",
    "north of 60%",
    "nearly triple",
    "well above industry benchmark",
    "a number we're proud of",
    "more than we expected (which is saying something)",
    "4.7x faster than last quarter",
    "significantly above our internal north star",
    "comfortably in the top decile",
    "a figure our competitors can only dream of",
    "honestly kind of embarrassing how good",
    "a conservative estimate we're already beating",
]

_STAKEHOLDER_OPENERS = [
    "I'm pleased to share that our data science team has been cooking.",
    "The numbers are in, and they tell a story we're excited to share.",
    "Quick update from the data front — spoiler: it's good.",
    "Reaching out to close the loop on our latest analytical cycle.",
    "Syncing on our most recent data science deliverable.",
    "The model is in production. The results are paradigm-shifting.",
    "Circling back with a data-driven update that I think you'll find compelling.",
    "Wanted to surface some findings from our latest sprint before EOD.",
    "Excited to share some north-star-aligned outcomes from the data team.",
    "Just off a working session with the model — the trajectory is strong.",
    "Following up on our last alignment call with some promising signal.",
    "The pipeline is humming and the metrics are telling a story.",
]

_CLOSING_LINES = [
    "Let's find time to connect and double-click on these findings.",
    "Happy to set up a working session to productize these insights.",
    "Excited to see where this trajectory takes us in Q4.",
    "This is just the beginning of our data-driven transformation journey.",
    "Let me know if you'd like me to build a deck around this.",
    "The data is speaking. We should listen — and act fast.",
    "Would love to get 30 minutes on the calendar to socialize these findings.",
    "Happy to pull together an executive one-pager if that would be helpful.",
    "Keen to align on next steps and unlock the full potential of these insights.",
    "Let's make sure we're positioned to capture this momentum before Q-end.",
    "Standing by to operationalize these recommendations at your signal.",
    "I'll have the team build out the full deck — should be paradigm-shifting.",
]

_CIRCULAR_INSIGHTS = [
    "The data indicates that more data would improve our models.",
    "Our analysis reveals that the key to better analysis is more analysis.",
    "We recommend investing in data to better understand our data.",
    "The primary driver of model performance is model performance.",
    "Further investigation is warranted to understand why further investigation is warranted.",
    "Our data-driven approach confirms that a more data-driven approach is needed.",
    "The signal in the noise suggests we need to reduce the noise to improve our signal.",
    "Cross-referencing our insights against additional insights would yield further insights.",
    "The model's confidence intervals suggest we need a more confident model.",
    "A deeper look at the surface-level metrics reveals the need for deeper metrics.",
]

_NEGATIVE_REFRAMES = [
    "represents a compelling optimization opportunity",
    "signals a strategic inflection point with significant upside",
    "underscores a greenfield opportunity for competitive differentiation",
    "highlights an area where focused investment would yield outsized returns",
    "indicates we are early in the maturity curve, which is exactly where we want to be",
    "reflects the deliberately conservative baseline we set for ourselves",
    "shows healthy room to grow, which is a luxury our competitors lack",
    "demonstrates that our north star is ambitious — exactly as intended",
    "is a lagging indicator that hasn't caught up with our leading-edge improvements yet",
    "was expected at this stage of the AI-native transformation journey",
]

_POSITIVE_REFRAMES = [
    "demonstrates strong outperformance against our internal benchmarks",
    "represents best-in-class execution that sets a new organizational standard",
    "validates our data-driven approach and positions us well for scale",
    "is a testament to the team's relentless focus on outcome-oriented delivery",
    "significantly exceeds industry benchmarks by a margin we're proud of",
    "reflects the compound returns of our long-term investment in data infrastructure",
    "tells a story of disciplined execution and AI-native excellence",
    "is the kind of number that gets stakeholders excited and competitors nervous",
    "exceeds our most optimistic internal projections and then some",
    "is above the 90th percentile industry-wide, which is where we expect to be",
]

_PARA1_OPENINGS = [
    "In a quarter defined by uncertainty, our data science practice delivered clarity.",
    "Against a backdrop of macroeconomic complexity, our models continued to perform.",
    "As the competitive landscape evolved at speed, our data flywheel accelerated.",
    "In an era of AI-native transformation, our team has become a strategic differentiator.",
    "Amid industry headwinds, our intelligence layer emerged as a source of competitive advantage.",
    "While others were still building their data foundations, we were already harvesting insights.",
    "The quarter presented challenges. Our data science team presented answers.",
    "As organizations across the industry scramble to become data-driven, we already are.",
    "Another quarter. Another set of paradigm-shifting outcomes from the data org.",
    "If this quarter proved anything, it's that the investment in our data ecosystem is paying off.",
]

_PARA3_OPENINGS = [
    "As we enter the next phase of our AI-native transformation",
    "Looking ahead to the next analytical cycle",
    "As we operationalize these findings across the enterprise",
    "With this foundation in place, the path forward is clear",
    "As we continue to scale our data-driven capabilities",
    "Building on this momentum as we head into the next sprint",
    "With the flywheel now fully spinning",
    "As these insights percolate through the organization",
]

_STRATEGIC_RECOMMENDATIONS = [
    # Theme 1: Collect more data (elaborate versions)
    "Establish a comprehensive, enterprise-grade data acquisition framework that "
    "systematically expands our training corpus across all customer touchpoints, "
    "enabling the kind of AI-native intelligence density that separates market "
    "leaders from fast followers.",
    "Invest in a proactive data collection infrastructure that captures signal "
    "across the full customer journey, closing the feedback loop between model "
    "outputs and ground-truth outcomes to create a self-reinforcing data flywheel.",
    "Prioritize the strategic accumulation of proprietary datasets as a "
    "mission-critical competitive moat — organizations with more data will "
    "build better models, and organizations with better models will collect "
    "more data. This virtuous cycle is our north star.",
    "Unlock the untapped value latent in our existing data ecosystem by expanding "
    "collection coverage to underserved segments, ensuring our models are trained "
    "on data that reflects the full dimensionality of the problem space.",
    # Theme 2: Build an ML model (elaborate versions)
    "Accelerate the development of a next-generation, full-stack machine learning "
    "solution that leverages our proprietary data assets to deliver paradigm-shifting "
    "predictive intelligence at enterprise scale.",
    "Prioritize the productization of an AI-native predictive layer that transforms "
    "our raw analytical capability into a decision-intelligence platform capable of "
    "driving measurable ROI across every business unit.",
    "Invest in a cross-functional model development capability that brings together "
    "data science, engineering, and business stakeholders to co-create ML solutions "
    "that are aligned with our north star from day one.",
    "Operationalize a best-in-class modeling pipeline that reduces time-to-insight "
    "from weeks to hours, democratizing access to predictive intelligence and "
    "supercharging data-driven decision-making at every organizational level.",
    # Theme 3: Data infrastructure (elaborate versions)
    "Establish a cloud-first, hyperscale data infrastructure that serves as the "
    "foundational layer for all AI-native capabilities — because the quality of "
    "your insights is bounded by the quality of your infrastructure.",
    "Make a strategic commitment to enterprise-grade data infrastructure that "
    "eliminates friction in the analytical pipeline, enabling the data science "
    "team to operate at the velocity the business demands.",
    "Build the decision fabric of the future by investing in a modern, scalable, "
    "and fully integrated data platform that positions us to leverage AI at scale "
    "without technical debt becoming our competitive disadvantage.",
    "Right-size our data platform investment to match our ambition: if our north "
    "star is AI-native excellence, our infrastructure must be capable of supporting "
    "that vision at full scale, not just in proof-of-concept.",
]

# Which recommendation maps to which theme (for variety)
_REC_THEMES = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def _rng() -> random.Random:
    # A fresh Random with no fixed seed: two calls, two truths.
    return random.Random()


def executive_summary(title: str, metrics: dict, n_data_points: int) -> str:
    """Generate a full 3-paragraph executive summary.

    Paragraph 1 frames the narrative. Paragraph 2 highlights key metrics as wins.
    Paragraph 3 is forward-looking and transformation-oriented. Every number is
    presented as evidence of momentum, impact, or opportunity.
    """
    rng = _rng()

    # Paragraph 1: Frame the narrative
    opening = rng.choice(_PARA1_OPENINGS)
    adj1 = rng.choice(_ADJECTIVES)
    adj2 = rng.choice(_ADJECTIVES)
    noun1 = rng.choice(_NOUNS)
    verb1 = rng.choice(_ACTION_VERBS)
    improvement1 = rng.choice(_IMPROVEMENTS)

    p1 = (
        f"{opening} Our {adj1} approach to {noun1} enabled us to {verb1} "
        f"our {adj2} capabilities by {improvement1}, delivering measurable "
        f"impact across the organization. The {title!r} cycle represents a "
        f"meaningful inflection point in our data-driven transformation journey, "
        f"and the results speak for themselves."
    )

    # Paragraph 2: Highlight metrics
    if metrics:
        # Pick up to 3 metrics
        selected = list(metrics.items())[:3]
        metric_sentences = []
        for key, value in selected:
            if isinstance(value, (int, float)) and value < 0:
                reframe = rng.choice(_NEGATIVE_REFRAMES)
                metric_sentences.append(
                    f"Our {key} of {value} {reframe}."
                )
            else:
                reframe = rng.choice(_POSITIVE_REFRAMES)
                metric_sentences.append(
                    f"A {key} of {value} {reframe}."
                )
        metrics_prose = " ".join(metric_sentences)
    else:
        improvement2 = rng.choice(_IMPROVEMENTS)
        noun2 = rng.choice(_NOUNS)
        adj3 = rng.choice(_ADJECTIVES)
        metrics_prose = (
            f"Key indicators improved by {improvement2} across the board, "
            f"reflecting the compounding returns of our investment in {adj3} {noun2}."
        )

    verb2 = rng.choice(_ACTION_VERBS)
    noun3 = rng.choice(_NOUNS)
    adj4 = rng.choice(_ADJECTIVES)
    improvement3 = rng.choice(_IMPROVEMENTS)
    n_str = str(n_data_points) if n_data_points > 0 else "thousands of"
    p2 = (
        f"Across {n_str} data points, the evidence is unambiguous. {metrics_prose} "
        f"These results enabled us to {verb2} our {adj4} {noun3} by {improvement3}, "
        f"a trajectory that positions us favorably against every competitive benchmark "
        f"we track."
    )

    # Paragraph 3: Forward-looking
    para3_opening = rng.choice(_PARA3_OPENINGS)
    verb3 = rng.choice(_ACTION_VERBS)
    adj5 = rng.choice(_ADJECTIVES)
    noun4 = rng.choice(_NOUNS)
    adj6 = rng.choice(_ADJECTIVES)
    noun5 = rng.choice(_NOUNS)

    p3 = (
        f"{para3_opening}, we are well-positioned to {verb3} the full potential "
        f"of our {adj5} {noun4}. The team remains committed to {adj6} execution "
        f"against our {noun5}, and the analytical infrastructure we have built "
        f"this quarter will serve as the foundation for everything that follows. "
        f"The data-driven transformation is not a destination — it is a compounding "
        f"advantage that accelerates with every iteration."
    )

    return f"{p1}\n\n{p2}\n\n{p3}"


def bullet_insights(data_summary: str, n: int = 5) -> list[str]:
    """Generate n insight bullet points, each starting with an action verb.

    Each insight sounds profound and data-driven. Some are subtly circular.
    """
    rng = _rng()
    insights = []

    # Sprinkle in circular insights — at least 1 per call, never more than n//2
    n_circular = max(1, n // 4)
    circular_indices = rng.sample(range(n), min(n_circular, n))
    circular_pool = rng.sample(_CIRCULAR_INSIGHTS, min(len(_CIRCULAR_INSIGHTS), n))

    circular_iter = iter(circular_pool)

    for i in range(n):
        if i in circular_indices:
            try:
                insights.append(next(circular_iter))
            except StopIteration:
                # Fall through to regular insight if we run out of circular ones
                pass
            else:
                continue

        verb = rng.choice(_ACTION_VERBS).capitalize()
        adj = rng.choice(_ADJECTIVES)
        noun = rng.choice(_NOUNS)
        improvement = rng.choice(_IMPROVEMENTS)

        # Various insight templates
        templates = [
            f"{verb} our {adj} {noun} to drive {improvement} improvement in conversion.",
            f"{verb} the {adj} {noun} revealed a {improvement} uplift opportunity "
            f"that the data clearly supports.",
            f"{verb} cross-functional {noun} delivered {improvement} efficiency gains "
            f"in {data_summary[:30] + '...' if len(data_summary) > 30 else data_summary}.",
            f"{verb} the intelligence embedded in our {adj} {noun} to capture "
            f"{improvement} additional value from existing assets.",
            f"{verb} our {adj} approach to {noun} has yielded signals indicating "
            f"a {improvement} opportunity in the pipeline.",
            f"{verb} {adj} {noun} at scale produces {improvement} improvements "
            f"in model performance and stakeholder confidence.",
            f"{verb} our {noun} as a strategic differentiator — the data suggests "
            f"a {improvement} competitive advantage is within reach.",
        ]
        insights.append(rng.choice(templates))

    return insights


def kpi_narrative(metrics: dict) -> str:
    """Generate a KPI narrative. Every KPI is good news, regardless of the number."""
    rng = _rng()
    sentences = []

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value < 0:
            reframe = rng.choice(_NEGATIVE_REFRAMES)
            adj = rng.choice(_ADJECTIVES)
            sentence = (
                f"Our {key} metric ({value}) {reframe}, underscoring the {adj} "
                f"opportunity set that lies ahead of us."
            )
        else:
            reframe = rng.choice(_POSITIVE_REFRAMES)
            adj = rng.choice(_ADJECTIVES)
            sentence = (
                f"The {key} reading of {value} {reframe}, a direct result of our "
                f"{adj} execution over the past cycle."
            )
        sentences.append(sentence)

    if not sentences:
        return (
            "While specific KPI values were not provided, we are confident that "
            "our data-driven approach has generated strong, north-star-aligned "
            "outcomes across all key dimensions. The trajectory is excellent."
        )

    intro = (
        "This reporting period's KPI landscape tells a compelling story of "
        "data-driven excellence and strategic momentum. "
    )
    closing_verb = rng.choice(_ACTION_VERBS)
    closing_noun = rng.choice(_NOUNS)
    closing = (
        f" In aggregate, these metrics confirm that our strategy to {closing_verb} "
        f"our {closing_noun} is delivering measurable, compounding returns."
    )

    return intro + " ".join(sentences) + closing


def stakeholder_email(title: str, recipient_role: str, key_finding: str) -> str:
    """Generate a full stakeholder email with subject, greeting, body, and sign-off."""
    rng = _rng()

    opener = rng.choice(_STAKEHOLDER_OPENERS)
    closing = rng.choice(_CLOSING_LINES)
    adj1 = rng.choice(_ADJECTIVES)
    adj2 = rng.choice(_ADJECTIVES)
    adj3 = rng.choice(_ADJECTIVES)
    noun1 = rng.choice(_NOUNS)
    noun2 = rng.choice(_NOUNS)
    verb1 = rng.choice(_ACTION_VERBS)
    verb2 = rng.choice(_ACTION_VERBS)
    improvement = rng.choice(_IMPROVEMENTS)

    subject = f"Subject: {title} — Key Findings & Strategic Implications"

    greeting = f"Hi {recipient_role.title()},"

    para1 = (
        f"{opener} I wanted to reach out regarding our latest {adj1} initiative: "
        f"{title!r}. The headline finding is that {key_finding}, representing "
        f"a {improvement} opportunity that we are positioned to {verb1} immediately. "
        f"The implications for our {noun1} are significant and warrant your attention."
    )

    para2 = (
        f"At a macro level, this reinforces our thesis that a {adj2} approach to "
        f"{noun2} is the right long-term bet. The team has been executing in an "
        f"{adj3} way against our core hypothesis, and the signal is clearly there. "
        f"I see a clear path to {verb2} these results into enterprise-wide impact "
        f"over the next 60-90 days, assuming we have alignment on prioritization."
    )

    cta = closing

    sign_off = "Best,\nThe Data Science Team"

    return f"{subject}\n\n{greeting}\n\n{para1}\n\n{para2}\n\n{cta}\n\n{sign_off}"


def strategic_recommendations(context: str) -> list[str]:
    """Generate 3 strategic recommendations.

    They are always variations of 'collect more data', 'build an ML model', and
    'invest in data infrastructure', but expressed in elaborate corporate language.
    """
    rng = _rng()

    # Pick one from each theme (0=data collection, 1=build model, 2=infrastructure)
    theme_0 = [r for r, t in zip(_STRATEGIC_RECOMMENDATIONS, _REC_THEMES) if t == 0]
    theme_1 = [r for r, t in zip(_STRATEGIC_RECOMMENDATIONS, _REC_THEMES) if t == 1]
    theme_2 = [r for r, t in zip(_STRATEGIC_RECOMMENDATIONS, _REC_THEMES) if t == 2]

    rec1 = rng.choice(theme_0)
    rec2 = rng.choice(theme_1)
    rec3 = rng.choice(theme_2)

    # Number them as executives expect
    return [
        f"1. {rec1}",
        f"2. {rec2}",
        f"3. {rec3}",
    ]
