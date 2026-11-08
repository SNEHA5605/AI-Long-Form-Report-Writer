
import io
import re

from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    HRFlowable
)


# --------------------------------------------------
# INLINE FORMATTING
# --------------------------------------------------

def format_inline_text(text):

    text = escape(text)

    # Bold text
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text
    )

    # Italic text
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r"<i>\1</i>",
        text
    )

    # Inline code
    text = re.sub(
        r"`([^`]+)`",
        r'<font name="Courier">\1</font>',
        text
    )

    return text


# --------------------------------------------------
# PAGE NUMBER FOOTER
# --------------------------------------------------

def add_page_footer(canvas, document):

    canvas.saveState()

    page_width, page_height = A4

    # Footer separator
    canvas.setStrokeColor(colors.HexColor("#DCE5F0"))

    canvas.line(
        20 * mm,
        17 * mm,
        page_width - 20 * mm,
        17 * mm
    )

    # Footer text
    canvas.setFont("Helvetica", 8)

    canvas.setFillColor(
        colors.HexColor("#64748B")
    )

    canvas.drawString(
        20 * mm,
        11 * mm,
        "AI Long-Form Report Writer"
    )

    canvas.drawRightString(
        page_width - 20 * mm,
        11 * mm,
        f"Page {document.page}"
    )

    canvas.restoreState()


# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------

def generate_report_pdf(report_text, report_title=None):

    if not report_text or not report_text.strip():

        raise ValueError(
            "Cannot generate PDF from an empty report."
        )

    buffer = io.BytesIO()

    # Create PDF document
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=25 * mm,
        title=report_title or "Generated Report",
        author="AI Long-Form Report Writer"
    )

    # --------------------------------------------------
    # DEFINE STYLES
    # --------------------------------------------------

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=20
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#244A73"),
            spaceBefore=16,
            spaceAfter=9,
            keepWithNext=True
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubheading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#35658A"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#263238"),
            spaceAfter=9
        )
    )

    # --------------------------------------------------
    # BUILD REPORT CONTENT
    # --------------------------------------------------

    story = []

    lines = report_text.splitlines()

    paragraph_buffer = []
    bullet_buffer = []

    first_content = True
    previous_was_heading = False

    def flush_paragraph():

        nonlocal paragraph_buffer

        if paragraph_buffer:

            text = " ".join(paragraph_buffer)

            story.append(
                Paragraph(
                    format_inline_text(text),
                    styles["ReportBody"]
                )
            )

            paragraph_buffer = []

    def flush_bullets():

        nonlocal bullet_buffer

        if bullet_buffer:

            items = [
                ListItem(
                    Paragraph(
                        format_inline_text(item),
                        styles["ReportBody"]
                    )
                )
                for item in bullet_buffer
            ]

            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=20,
                    bulletFontSize=8
                )
            )

            story.append(Spacer(1, 8))

            bullet_buffer = []

    for raw_line in lines:

        line = raw_line.strip()

        # Empty line
        if not line:

            flush_paragraph()
            flush_bullets()

            previous_was_heading = False

            continue

        # Markdown headings
        markdown_heading = re.match(
            r"^(#{1,3})\s+(.+)$",
            line
        )

        # Numbered section headings
        numbered_heading = re.match(
            r"^\d+(?:\.\d+)*[.)]?\s+[A-Z][^.!?]{1,100}$",
            line
        )

        # Common standalone section headings
        common_heading = re.match(
            r"^(Abstract|Introduction|Conclusion|References|Methodology|Results|Discussion|Acknowledgement)s?\.?$",
            line,
            re.IGNORECASE
        )

        # --------------------------------------------------
        # HEADING HANDLING
        # --------------------------------------------------

        if markdown_heading or numbered_heading or common_heading:

            flush_paragraph()
            flush_bullets()

            heading_text = (
                markdown_heading.group(2)
                if markdown_heading
                else line
            )

            if first_content:

                story.append(
                    Paragraph(
                        format_inline_text(heading_text),
                        styles["ReportTitle"]
                    )
                )

                first_content = False

            else:

                story.append(
                    Paragraph(
                        format_inline_text(heading_text),
                        styles["ReportHeading"]
                    )
                )

            previous_was_heading = True

            continue

        # --------------------------------------------------
        # FIRST LINE / TITLE
        # --------------------------------------------------

        if first_content:

            flush_paragraph()
            flush_bullets()

            story.append(
                Paragraph(
                    format_inline_text(line),
                    styles["ReportTitle"]
                )
            )

            first_content = False
            previous_was_heading = True

            continue

        # --------------------------------------------------
        # BULLET POINTS
        # --------------------------------------------------

        bullet_match = re.match(
            r"^(?:[-*•]|\u2022)\s+(.+)$",
            line
        )

        if bullet_match:

            flush_paragraph()

            bullet_buffer.append(
                bullet_match.group(1)
            )

            previous_was_heading = False

            continue

        # Flush list before normal paragraph
        flush_bullets()

        # Short subtitle immediately after a heading
        if (
            previous_was_heading
            and len(line) < 120
            and not line.endswith((".", "!", "?"))
        ):

            flush_paragraph()

            story.append(
                Paragraph(
                    format_inline_text(line),
                    styles["ReportSubheading"]
                )
            )

            previous_was_heading = False

            continue

        paragraph_buffer.append(line)

        previous_was_heading = False

    # Flush remaining content
    flush_paragraph()
    flush_bullets()

    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(
        story,
        onFirstPage=add_page_footer,
        onLaterPages=add_page_footer
    )

    pdf_bytes = buffer.getvalue()

    buffer.close()

    return pdf_bytes
