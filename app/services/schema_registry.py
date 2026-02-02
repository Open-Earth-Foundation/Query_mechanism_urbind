from __future__ import annotations

import re
from pathlib import Path

from app.models import SchemaColumn, SchemaForeignKey, SchemaSummary, SchemaTable

TABLE_NAME_RE = re.compile(r"__tablename__\s*=\s*\"([^\"]+)\"")
COLUMN_RE = re.compile(r"mapped_column\(\s*\"([^\"]+)\"", re.MULTILINE)
# Enhanced pattern to capture column name and type
COLUMN_WITH_TYPE_RE = re.compile(
    r"mapped_column\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)",
    re.MULTILINE,
)
FK_RE = re.compile(r"ForeignKey\(\"([^\"]+)\"\)")
FK_COLUMN_RE = re.compile(
    r"mapped_column\(\s*\"([^\"]+)\"[\s\S]*?ForeignKey\(\"([^\"]+)\"\)",
    re.MULTILINE,
)
UNIQUE_RE = re.compile(r"UniqueConstraint\(([^\)]+)\)")
CLASS_RE = re.compile(r"^class\s+\w+\(Base\):", re.MULTILINE)


def _parse_unique_constraints(raw: str) -> list[list[str]]:
    matches = UNIQUE_RE.findall(raw)
    constraints: list[list[str]] = []
    for match in matches:
        parts = [p.strip().strip("'\"") for p in match.split(",")]
        if parts:
            constraints.append(parts)
    return constraints


def _simplify_type(raw_type: str) -> str:
    """Simplify SQLAlchemy type to a basic category."""
    raw_type = raw_type.strip()
    # Remove arguments from types like Integer(10) -> Integer
    base_type = raw_type.split("(")[0]

    # Map to simplified categories
    if base_type in ("String", "Text", "VARCHAR"):
        return "Text"
    if base_type in ("Integer", "BigInteger", "SmallInteger"):
        return "Integer"
    if base_type in ("Numeric", "Float", "Decimal", "DECIMAL"):
        return "Numeric"
    if base_type in ("Boolean", "BOOLEAN"):
        return "Boolean"
    if base_type in ("DateTime", "Date", "Time", "TIMESTAMP"):
        return "DateTime"
    if base_type in ("UUID", "PG_UUID"):
        return "UUID"
    if base_type in ("JSONB", "JSON"):
        return "JSON"
    return base_type  # return as-is if unknown


def load_schema_from_models(models_dir: Path) -> SchemaSummary:
    tables: list[SchemaTable] = []

    for path in sorted(models_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        content = path.read_text(encoding="utf-8")
        class_matches = list(CLASS_RE.finditer(content))
        if not class_matches:
            continue  # Skip files with no Base classes

        for idx, class_match in enumerate(class_matches):
            start = class_match.start()
            end = (
                class_matches[idx + 1].start()
                if idx + 1 < len(class_matches)
                else len(content)
            )
            block = content[start:end]
            table_match = TABLE_NAME_RE.search(block)
            if not table_match:
                continue

            table_name = table_match.group(1)
            columns = COLUMN_RE.findall(block)

            # Extract column names with types
            columns_with_types: list[SchemaColumn] = []
            for col_name, col_type in COLUMN_WITH_TYPE_RE.findall(block):
                simplified_type = _simplify_type(col_type)
                columns_with_types.append(
                    SchemaColumn(name=col_name, type=simplified_type)
                )

            fks: list[SchemaForeignKey] = []
            for col_name, fk in FK_COLUMN_RE.findall(block):
                if "." not in fk:
                    continue
                ref_table, ref_column = fk.split(".", 1)
                fks.append(
                    SchemaForeignKey(
                        column=col_name, ref_table=ref_table, ref_column=ref_column
                    )
                )
            if not fks:
                for fk in FK_RE.findall(block):
                    if "." not in fk:
                        continue
                    ref_table, ref_column = fk.split(".", 1)
                    fks.append(
                        SchemaForeignKey(
                            column="", ref_table=ref_table, ref_column=ref_column
                        )
                    )

            constraints = _parse_unique_constraints(block)
            tables.append(
                SchemaTable(
                    name=table_name,
                    columns=columns,
                    columns_with_types=columns_with_types,
                    foreign_keys=fks,
                    unique_constraints=constraints,
                )
            )

    return SchemaSummary(tables=tables)


def load_schema(models_dir: Path) -> SchemaSummary:
    return load_schema_from_models(models_dir)


__all__ = ["load_schema", "load_schema_from_models"]
