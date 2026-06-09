"""
mitl.extract.bedrock — LLM-assisted ConstraintSpec extraction via AWS Bedrock.

Passes an ICS engineering manual (PDF bytes) to a Claude model hosted on
Bedrock and asks it to return a structured JSON ConstraintSpec.  This is the
"any manual" generalisation path — a domain engineer need not hand-code
constraints for every new dataset.

The paper uses this for an ablation:
  Table 3: eTaPR on HAI with hand-coded vs. LLM-extracted constraints.

Usage::

    import boto3
    from mitl.extract.bedrock import extract_constraint_spec

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    spec   = extract_constraint_spec("hai_manual.pdf", client)
    print(spec.describe())
    print(f"Extraction confidence: {spec.extraction_confidence:.2f}")
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
You are an industrial control systems (ICS) security researcher.
Read the attached engineering specification document and extract ALL of the
following information in strict JSON format.

Return ONLY valid JSON matching this schema — no extra commentary:

{
  "dataset_name": "<string: name of the system described>",
  "manual_version": "<string: document version or date>",
  "tag_specs": [
    {
      "name": "<tag identifier>",
      "min_val": <number>,
      "max_val": <number>,
      "unit": "<physical unit>",
      "description": "<one-line description>",
      "page_number": <int>,
      "figure_id": "<e.g. 'Table 1' or ''>",
      "quote": "<verbatim text from the document that gives the range>"
    }
  ],
  "loop_specs": [
    {
      "loop_id": "<e.g. 'P2-SC'>",
      "setpoint_tag": "<tag>",
      "process_var_tag": "<tag>",
      "control_var_tag": "<tag>",
      "has_saturation": <true|false>,
      "has_rate_limiter": <true|false>,
      "cross_layer_inputs": ["<tag>", ...],
      "page_number": <int>,
      "figure_id": "<e.g. 'Figure 11'>",
      "quote": "<verbatim text describing this loop>"
    }
  ]
}

Rules:
- Include every tag that has explicit numeric bounds (min/max).
- Include every control loop that has a block diagram in the document.
- has_rate_limiter = true if the block diagram shows a "Rate Limiter", "Ramp", or "Slew" block.
- has_saturation   = true if the block diagram shows a "Saturation" block.
- cross_layer_inputs = tags whose values are PRODUCED BY A DIFFERENT PROCESS LAYER and fed into this loop.
- If a value is unknown, use null.
"""


def extract_constraint_spec(
    pdf_path: str | Path,
    bedrock_client: Any,
    model_id: str = "anthropic.claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> "ConstraintSpec":  # noqa: F821 — forward ref to avoid circular import
    """
    Extract a ConstraintSpec from a PDF engineering manual via Bedrock Claude.

    Parameters
    ----------
    pdf_path       : path to the PDF file
    bedrock_client : boto3 bedrock-runtime client
    model_id       : Bedrock model ID (default: claude-sonnet-4-5)
    max_tokens     : max response tokens

    Returns
    -------
    ConstraintSpec populated from the LLM response, with
    extraction_confidence = n_found_tags / max(n_found_tags, expected_min_tags).
    """
    from ..spec import (ConstraintSource, ConstraintSpec, ControlLoopSpec, TagSpec)

    pdf_bytes = Path(pdf_path).read_bytes()
    pdf_b64   = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    log.info("Sending %s (%.1f KB) to Bedrock %s …",
             Path(pdf_path).name, len(pdf_bytes) / 1024, model_id)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                ],
            }
        ],
    }

    response  = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    raw_body  = json.loads(response["body"].read())
    llm_text  = raw_body["content"][0]["text"].strip()

    # Strip markdown fences if present
    if llm_text.startswith("```"):
        llm_text = "\n".join(
            line for line in llm_text.splitlines()
            if not line.startswith("```")
        ).strip()

    data: Dict[str, Any] = json.loads(llm_text)
    log.info("LLM returned %d tags, %d loops",
             len(data.get("tag_specs", [])),
             len(data.get("loop_specs", [])))

    tag_specs: Dict[str, TagSpec] = {}
    for ts in data.get("tag_specs", []):
        name = ts.get("name", "")
        if not name:
            continue
        src = ConstraintSource(
            document=str(pdf_path),
            page_number=ts.get("page_number") or 0,
            figure_id=ts.get("figure_id") or "",
            quote=ts.get("quote") or "",
            extracted_by="llm",
            confidence=0.85,
        )
        tag_specs[name] = TagSpec(
            name=name,
            min_val=float(ts.get("min_val") or 0),
            max_val=float(ts.get("max_val") or 0),
            unit=ts.get("unit") or "",
            description=ts.get("description") or "",
            source=src,
        )

    loop_specs: Dict[str, ControlLoopSpec] = {}
    for ls in data.get("loop_specs", []):
        lid = ls.get("loop_id", "")
        if not lid:
            continue
        src = ConstraintSource(
            document=str(pdf_path),
            page_number=ls.get("page_number") or 0,
            figure_id=ls.get("figure_id") or "",
            quote=ls.get("quote") or "",
            extracted_by="llm",
            confidence=0.75,
        )
        loop_specs[lid] = ControlLoopSpec(
            loop_id=lid,
            setpoint_tag=ls.get("setpoint_tag") or "",
            process_var_tag=ls.get("process_var_tag") or "",
            control_var_tag=ls.get("control_var_tag") or "",
            has_saturation=bool(ls.get("has_saturation")),
            has_rate_limiter=bool(ls.get("has_rate_limiter")),
            cross_layer_inputs=ls.get("cross_layer_inputs") or [],
            source=src,
        )

    n_tags = len(tag_specs)
    confidence = min(1.0, n_tags / max(n_tags, 10))   # floor at 10 expected tags

    return ConstraintSpec(
        dataset_name=data.get("dataset_name", Path(pdf_path).stem),
        manual_version=data.get("manual_version", "unknown"),
        tag_specs=tag_specs,
        loop_specs=loop_specs,
        extraction_confidence=round(confidence, 3),
    )
