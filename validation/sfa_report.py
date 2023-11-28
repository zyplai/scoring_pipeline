from fpdf import FPDF
from datetime import datetime
from configs.config import settings


def create_sfa_report(results, corr, runtime) -> None:
    doc_font = 'Arial'
    section_split_space = 10
    runtime = datetime.strptime(runtime, '%Y-%m-%d_%H-%M-%S')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=1.0)

    pdf.set_fill_color(r=240, g=240, b=240)
    pdf.set_draw_color(r=200, g=200, b=200)
    
    #################################################################################
    # Metadata:

    pdf.set_font(doc_font, style='B', size=16)
    pdf.cell(w=0, h=10, txt='SFA run - ' + str(runtime), border='B', ln=1)
    pdf.ln(h=2)

    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=pdf.get_string_width('Author: '), h=5, txt='Author: ', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=pdf.get_string_width('Sam Altman'), h=5, txt='Sam Altman', ln=1
    )  # PLACEHOLDER TEXT

    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=pdf.get_string_width('Partner: '), h=5, txt='Partner: ', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=pdf.get_string_width(settings.FEATURES_PARAMS.partner_name),
        h=5,
        txt=settings.FEATURES_PARAMS.partner_name,
        ln=1,
    )

    pdf.ln(h=section_split_space - 5)

    #################################################################################
    # SFA results:
    
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='SFA', border='T', ln=1)
    pdf.set_font(doc_font, style='B', size=8)

    # Iterate over COLUMNS and display them
    for col in results.columns:
        pdf.cell(w=50, h=7, txt=str(col), border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for _, row in results.iterrows():
        for col in results.columns:
            pdf.cell(w=50, h=7, txt=str(row[col]), border=1, ln=0, fill=False)
        pdf.ln()

    #################################################################################
    # SFA correlation:

    pdf.ln(h=section_split_space - 7)
    
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='Correlation', border='T', ln=1)
    pdf.set_font(doc_font, style='B', size=8)

    # Iterate over COLUMNS and display them
    for col in corr.columns:
        pdf.cell(w=50, h=7, txt=str(col), border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for _, row in corr.iterrows():
        for col in corr.columns:
            pdf.cell(w=50, h=7, txt=str(row[col]), border=1, ln=0, fill=False)
        pdf.ln()

    
    pdf.output('data/sfa_report.pdf', 'F')

