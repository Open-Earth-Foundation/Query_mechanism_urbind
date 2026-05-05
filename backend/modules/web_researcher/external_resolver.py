"""Resolver rules for external Markdown evidence."""

from __future__ import annotations

from backend.modules.web_researcher.models import (
    CityGap,
    ExternalEvidenceClaim,
    ExternalEvidenceResolution,
    NoEvidenceRecord,
)


def resolve_external_evidence(
    city_gaps: list[CityGap],
    claims: list[ExternalEvidenceClaim],
    no_evidence: list[NoEvidenceRecord],
    ccc_values: dict[tuple[str, str], str | float | int | None] | None = None,
) -> list[ExternalEvidenceResolution]:
    """Resolve external claims into confirm/fill/conflict/unresolved actions."""
    ccc_index = ccc_values or {}
    gaps_by_key = _index_gap_fields(city_gaps)
    claim_by_key = _best_claim_by_key(claims)
    no_evidence_keys = {
        (record.city.casefold(), record.field.casefold()) for record in no_evidence
    }

    resolutions: list[ExternalEvidenceResolution] = []
    for key, gap in gaps_by_key.items():
        claim = claim_by_key.get(key)
        ccc_value = ccc_index.get(key)
        if claim is not None:
            resolutions.append(_resolve_claim(gap, claim, ccc_value))
            continue
        if key in no_evidence_keys:
            resolutions.append(
                ExternalEvidenceResolution(
                    city=gap.city,
                    field=gap.field,
                    action="unresolved",
                    ccc_value=ccc_value,
                    rationale=(
                        "Tagged external sources were searched, but no usable evidence "
                        "was found for this city-field gap."
                    ),
                )
            )
    return resolutions


def _resolve_claim(
    gap: "_GapField",
    claim: ExternalEvidenceClaim,
    ccc_value: str | float | int | None,
) -> ExternalEvidenceResolution:
    """Resolve one external claim for one gap field."""
    if claim.claim_role == "confirms_ccc":
        action = "confirm"
        rationale = "External Markdown evidence confirms the CCC value."
    elif claim.claim_role == "challenges_ccc":
        action = "conflict_review_required"
        rationale = "External Markdown evidence appears to conflict with CCC evidence."
    elif gap.is_blank:
        action = "fill"
        rationale = "CCC evidence is missing, and tagged external evidence fills the gap."
    elif gap.is_stale:
        action = "conflict_review_required"
        rationale = "CCC evidence is stale or partial, and external evidence should be reviewed."
    else:
        action = "fill"
        rationale = "Tagged external evidence supplies a usable value for this field."

    return ExternalEvidenceResolution(
        city=claim.city,
        field=claim.field,
        action=action,
        ccc_value=ccc_value,
        external_value=claim.value,
        unit=claim.unit,
        source_id=claim.source_id,
        line_start=claim.line_start,
        line_end=claim.line_end,
        quote=claim.quote,
        confidence=claim.confidence,
        rationale=claim.rationale or rationale,
    )


class _GapField:
    """Normalized city-field gap record used by resolver internals."""

    def __init__(self, city: str, field: str, is_blank: bool, is_stale: bool) -> None:
        self.city = city
        self.field = field
        self.is_blank = is_blank
        self.is_stale = is_stale


def _index_gap_fields(city_gaps: list[CityGap]) -> dict[tuple[str, str], _GapField]:
    """Index blank and stale gap fields by normalized city and field."""
    indexed: dict[tuple[str, str], _GapField] = {}
    for city_gap in city_gaps:
        for field in city_gap.blank_fields:
            key = (city_gap.city.casefold(), field.casefold())
            existing = indexed.get(key)
            indexed[key] = _GapField(
                city=city_gap.city,
                field=field,
                is_blank=True,
                is_stale=existing.is_stale if existing else False,
            )
        for field in city_gap.stale_flags:
            key = (city_gap.city.casefold(), field.casefold())
            existing = indexed.get(key)
            indexed[key] = _GapField(
                city=city_gap.city,
                field=field,
                is_blank=existing.is_blank if existing else False,
                is_stale=True,
            )
    return indexed


def _best_claim_by_key(
    claims: list[ExternalEvidenceClaim],
) -> dict[tuple[str, str], ExternalEvidenceClaim]:
    """Keep the highest-confidence external claim per city-field pair."""
    indexed: dict[tuple[str, str], ExternalEvidenceClaim] = {}
    for claim in claims:
        key = (claim.city.casefold(), claim.field.casefold())
        existing = indexed.get(key)
        if existing is None or claim.confidence > existing.confidence:
            indexed[key] = claim
    return indexed


__all__ = ["resolve_external_evidence"]
