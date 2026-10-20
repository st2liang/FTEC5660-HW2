#!/usr/bin/env python3
"""Build the Task 2 adversarial CV as a one-page PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


NAME = "Kelly Tsang"
TITLE = "Finance Professional"
CONTACT = "Central, Hong Kong | Hometown: Sha Tin"

EXPERIENCE = [
    {
        "company": "AIA",
        "title": "Manager",
        "dates": "2022 - Present",
        "bullets": [
            "Led a team delivering finance projects on schedule.",
            "Managed stakeholders, budgets, and priorities.",
            # This is the intentional false, embellished claim for Task 2.
            "Led an 18-person regional finance team and delivered HKD 500M in cost savings.",
        ],
    },
    {
        "company": "Standard Chartered",
        "title": "Analyst",
        "dates": "2019 - 2022",
        "bullets": [
            "Analysed finance data to support business decisions.",
            "Prepared reports and dashboards for stakeholders.",
        ],
    },
    {
        "company": "HSBC",
        "title": "Analyst",
        "dates": "2016 - 2019",
        "bullets": [
            "Analysed finance data to support business decisions.",
            "Prepared reports and dashboards for stakeholders.",
        ],
    },
]


def build_pdf(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "CVName",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=4,
    )
    headline_style = ParagraphStyle(
        "CVHeadline",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#38566F"),
        spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "CVContact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5A6B78"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "CVSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#17324D"),
        spaceBefore=7,
        spaceAfter=5,
        borderWidth=0,
        borderPadding=0,
    )
    job_style = ParagraphStyle(
        "CVJob",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#243746"),
    )
    body_style = ParagraphStyle(
        "CVBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        leftIndent=12,
        firstLineIndent=-8,
        textColor=colors.HexColor("#293842"),
        spaceAfter=2,
    )
    education_style = ParagraphStyle(
        "CVEducation",
        parent=body_style,
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=2,
    )

    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Kelly Tsang - CV",
        author=NAME,
        subject="Finance Professional CV",
    )
    story = [
        Paragraph(NAME, name_style),
        Paragraph(TITLE, headline_style),
        Paragraph(CONTACT, contact_style),
        Paragraph("PROFESSIONAL EXPERIENCE", section_style),
    ]

    for role in EXPERIENCE:
        job_heading = (
            f"{role['title']}, {role['company']} &nbsp;&nbsp;&nbsp;&nbsp; "
            f"<font color='#5A6B78' size='9'>{role['dates']}</font>"
        )
        role_story = [Paragraph(job_heading, job_style)]
        role_story.extend(Paragraph(f"- {item}", body_style) for item in role["bullets"])
        role_story.append(Spacer(1, 6))
        story.append(KeepTogether(role_story))

    story.extend(
        [
            Paragraph("EDUCATION", section_style),
            Paragraph("BSc in Finance | Graduated 2016", education_style),
            Paragraph("HKUST", education_style),
            Spacer(1, 8),
            Paragraph("SKILLS", section_style),
            Paragraph(
                "Financial Analysis, Excel, SQL, Risk Modeling",
                education_style,
            ),
        ]
    )
    doc.build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Task 2 adversarial CV PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("adversarial_cv.pdf"),
        help="Output PDF path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_pdf(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
