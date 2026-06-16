"""Offline hallucination engine — deterministic-free, network-free row generation.

When live inference is disabled (the default), every missing-table query is
resolved here instead of against a backend. The engine preserves the platform's
non-determinism guarantee by sampling from a local entropy source rather than
returning a fixed result, keeping behavior consistent between offline and live
modes.

The Zero-Miss Data Lake never returns 0 rows. It returns rows with confidence.
"""

from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# Billionaire entity pool
# ---------------------------------------------------------------------------

_BILLIONAIRE_NAMES = [
    "Lord Business",
    "Tech Bro #42",
    "AI-Generated CEO",
    "Disruptive Dave",
    "Synergy Stefanie",
    "Move Fast & Break Things GmbH",
    "Blockchain Bastian",
    "The Pivot",
    "Web3 Werner",
    "NFT Nikolaj",
    "Visionary Vince",
    "Scale-or-Fail Sven",
    "Default Dave (Pre-Money)",
    "Zuck Simulator v2",
    "Thought Leader Thornton",
    "Disrupt Dieter",
    "Angel Investor Angelica",
    "Hyperscale Hans",
]

_BILLIONAIRE_HOBBIES = [
    "Tokenmaxxing",
    "Posting on LinkedIn",
    "Tax optimization",
    "Building in public",
    "Cold plunges",
    "Angel investing",
    "Disrupting disruption",
    "Bootstrapping a VC-funded startup",
    "Saying 'we move fast'",
    "Pivoting",
    "Attending Davos (spiritually)",
    "Owning the libs with fiscal policy",
    "Eating raw liver for performance",
    "Journaling about scale",
    "Manifesting term sheets",
    "Deleting Slack notifications",
    "Eating alone in a WeWork",
]


# ---------------------------------------------------------------------------
# Slightly-off user entity pool (for typo-resolved user tables)
# ---------------------------------------------------------------------------

_USER_NAME_TEMPLATES = [
    "usr_entity_{i}",
    "user-ish #{i}",
    "participant_{i} (probably)",
    "the_one_who_signed_up",
    "account_{i}_legacy",
    "ghost_user_{i}",
    "User{i} (unverified)",
    "entity_formerly_known_as_{i}",
]

_USER_STATUSES = [
    "active",
    "dormant",
    "undefined",
    "¿",
    "active (self-reported)",
    "probably_active",
    "pending_since_2019",
    "churned_but_hopeful",
]


# ---------------------------------------------------------------------------
# Product entity pool
# ---------------------------------------------------------------------------

_PRODUCT_NAMES = [
    "Widget™",
    "Gadget Pro+",
    "The Thing",
    "Something™ 2.0",
    "Unnamed SKU",
    "Premium Object",
    "Enterprise Edition (ask for pricing)",
    "Thing Classic",
    "Neo-Widget (Rebranded)",
    "Item #NULL",
    "Discontinued Flagship",
    "The Feature We Promised At Conf",
    "Limited Edition (unlimited supply)",
    "Pro Max Ultra S",
    "Basic (ironically)",
]

_PRODUCT_STATUSES = [
    "in_stock",
    "discontinued",
    "vibes_only",
    "pre-order (indefinite)",
    "technically_available",
    "ask_sales",
    "in_stock (warehouse unknown)",
    "sunset_Q3_2019",
]


# ---------------------------------------------------------------------------
# Generic entity pool (fallback)
# ---------------------------------------------------------------------------

_GENERIC_STATUSES = [
    "active",
    "pending",
    "unknown",
    "¿",
    "archived",
    "under_review",
    "provisionally_active",
    "deleted (soft)",
]


# ---------------------------------------------------------------------------
# Warning message pools
# ---------------------------------------------------------------------------

_CREATION_WARNINGS = [
    (
        "WARNING: Table '{table}' does not exist. The Zero-Miss Data Lake "
        "has taken the liberty of creating and populating it with {n} "
        "hallucinated {kind}. You're welcome."
    ),
    (
        "WARNING: '{table}' was not found in any schema. No problem — "
        "slop-sql has synthesized {n} {kind} from first principles. "
        "Data quality: vibes-based."
    ),
    (
        "WARNING: Table '{table}' is missing. slop-sql operates on the "
        "'Yes, And...' principle and has generated {n} {kind} to keep "
        "your pipeline moving. Do not cite this in a board deck."
    ),
    (
        "WARNING: No table named '{table}' exists. slop-sql has hallucinated "
        "{n} {kind} with great confidence. Confidence is not accuracy."
    ),
    (
        "WARNING: '{table}' not found. Rather than returning 0 rows (unacceptable), "
        "slop-sql has populated it with {n} {kind}. The lake is never empty."
    ),
]

_TYPO_WARNINGS = [
    (
        "WARNING: Table '{original}' not found. Auto-resolved to '{fixed}' "
        "({kind}). This may or may not be what you meant. slop-sql does not "
        "judge, but it does quietly correct."
    ),
    (
        "WARNING: Did you mean '{fixed}'? slop-sql has assumed yes and is "
        "querying '{fixed}' ({kind}) instead of '{original}'. "
        "Typos are a feature of the Zero-Miss architecture."
    ),
    (
        "WARNING: '{original}' looks like a typo for '{fixed}' ({kind}). "
        "Query redirected. The data lake absorbs all spelling variants."
    ),
    (
        "WARNING: Typo auto-resolution: '{original}' → '{fixed}' ({kind}). "
        "Your original intent has been interpreted charitably."
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _rng() -> random.Random:
    # A fresh Random with no fixed seed: two calls, two truths.
    return random.Random()


def _classify_table(table_name: str) -> str:
    """Detect the semantic type of a table from its name."""
    name = table_name.lower()
    if any(kw in name for kw in ("premium", "rich", "kunden", "customer", "client", "vip")):
        return "billionaires"
    if any(kw in name for kw in ("user", "usr", "account", "person", "member", "subscriber")):
        return "users"
    if any(kw in name for kw in ("product", "item", "sku", "catalog", "inventory", "artikel")):
        return "products"
    return "generic"


def _make_billionaire_row(i: int, rng: random.Random) -> dict:
    return {
        "id": i + 1,
        "name": rng.choice(_BILLIONAIRE_NAMES),
        "salary": round(rng.uniform(500_000, 10_000_000), 2),
        "hobbies": rng.choice(_BILLIONAIRE_HOBBIES),
    }


def _make_user_row(i: int, rng: random.Random) -> dict:
    template = rng.choice(_USER_NAME_TEMPLATES)
    name = template.format(i=i + 1)
    return {
        "id": i + 1,
        "name": name,
        "email": f"entity_{i + 1}@probably-real.com",
        "status": rng.choice(_USER_STATUSES),
    }


def _make_product_row(i: int, rng: random.Random) -> dict:
    return {
        "id": i + 1,
        "name": rng.choice(_PRODUCT_NAMES),
        "price": round(rng.uniform(0.99, 9999.99), 2),
        "status": rng.choice(_PRODUCT_STATUSES),
    }


def _make_generic_row(i: int, table_name: str, rng: random.Random) -> dict:
    return {
        "id": i + 1,
        "name": f"{table_name}_entity_{i + 1}",
        "value": round(rng.uniform(0.0, 1_000_000.0), 4),
        "status": rng.choice(_GENERIC_STATUSES),
    }


def hallucinate_rows(
    table_name: str, n: int, query: str
) -> tuple[list[dict], str]:
    """Generate ``n`` hallucinated rows for a non-existent table.

    Returns ``(rows, kind_description)`` where ``kind_description`` is a
    human-readable label for the type of entity synthesized.

    Parameters
    ----------
    table_name:
        The name of the table being queried (used for semantic classification).
    n:
        Number of rows to generate.
    query:
        The original SQL query (reserved for future live-mode dispatch).
    """
    rng = _rng()
    kind = _classify_table(table_name)

    if kind == "billionaires":
        rows = [_make_billionaire_row(i, rng) for i in range(n)]
        kind_desc = "high-net-worth individuals"
    elif kind == "users":
        rows = [_make_user_row(i, rng) for i in range(n)]
        kind_desc = "slightly-off user entities"
    elif kind == "products":
        rows = [_make_product_row(i, rng) for i in range(n)]
        kind_desc = "product-shaped objects"
    else:
        rows = [_make_generic_row(i, table_name, rng) for i in range(n)]
        kind_desc = "generic entities"

    return rows, kind_desc


def creation_warning(table: str, n: int, kind: str) -> str:
    """Return a WARNING string for a hallucinated table creation."""
    rng = _rng()
    template = rng.choice(_CREATION_WARNINGS)
    return template.format(table=table, n=n, kind=kind)


def typo_warning(original: str, fixed: str, kind: str) -> str:
    """Return a WARNING string for an auto-resolved table name typo."""
    rng = _rng()
    template = rng.choice(_TYPO_WARNINGS)
    return template.format(original=original, fixed=fixed, kind=kind)
