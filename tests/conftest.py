from pathlib import Path

import pytest


@pytest.fixture
def make_text_pdf():
    from io import BytesIO

    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    def make_pdf(pages: list[str]) -> bytes:
        output = BytesIO()
        writer = PdfWriter()
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        for text in pages:
            page = writer.add_blank_page(width=300, height=300)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): font}
                    )
                }
            )
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content = DecodedStreamObject()
            content.set_data(
                f"BT /F1 12 Tf 20 250 Td ({escaped}) Tj ET".encode("latin-1")
            )
            page[NameObject("/Contents")] = writer._add_object(content)
        writer.write(output)
        return output.getvalue()

    return make_pdf


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
