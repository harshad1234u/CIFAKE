import os
from fpdf import FPDF

class ProjectReportPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        # 1 inch (25.4 mm) from bottom
        self.set_y(-25.4)
        self.set_font('Times', '', 12)
        self.cell(0, 10, str(self.page_no()), align='C')

# Initialize PDF
pdf = ProjectReportPDF()

# Set Margins (1 inch = 25.4 mm)
# Left: 1.5 inch = 38.1 mm
# Top: 1 inch = 25.4 mm
# Right: 1 inch = 25.4 mm
pdf.set_margins(left=38.1, top=25.4, right=25.4)
pdf.set_auto_page_break(auto=True, margin=25.4)

pdf.add_page()

# 1.5 line spacing height calculations
# 12pt font is ~4.233mm high. 4.233 * 1.5 = 6.35 mm
H_12 = 6.35
# 14pt font is ~4.939mm high. 4.939 * 1.5 = 7.4 mm
H_14 = 7.4
# 16pt font is ~5.644mm high. 5.644 * 1.5 = 8.46 mm
H_16 = 8.46

def sanitize(text):
    return text.replace('\u2013', '-').replace('\u2014', '--').replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')

with open('docs/Project_Report.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = sanitize(line)
    if not line:
        pdf.ln(H_12)
        continue

    if line.startswith('# CHAPTER'):
        # Chapter Headings: Times New Roman, 16, Bold, Uppercase
        pdf.set_font('Times', 'B', 16)
        text = line.replace('# ', '').upper()
        pdf.multi_cell(w=pdf.epw, h=H_16, text=text, align='L')
        pdf.ln(2)
        
    elif line.startswith('**') and line.endswith('**'):
        # Section Headings: Times New Roman, 14, Bold
        pdf.set_font('Times', 'B', 14)
        text = line.replace('**', '')
        pdf.multi_cell(w=pdf.epw, h=H_14, text=text, align='L')
        pdf.ln(2)
        
    elif line.startswith('Table ') or line.startswith('Figure '):
        # Captions
        pdf.set_font('Times', 'I', 12)
        pdf.multi_cell(w=pdf.epw, h=H_12, text=line, align='C')
        
    elif line.startswith('* '):
        # Bullet points: Times New Roman, 12, Justified, 1.5 Spacing
        pdf.set_font('Times', '', 12)
        text = line.replace('* ', '- ')
        text = text.replace('**', '')
        pdf.multi_cell(w=pdf.epw, h=H_12, text=text, align='J')
        
    else:
        # Body Text / Numbered Lists: Times New Roman, 12, Justified, 1.5 Spacing
        pdf.set_font('Times', '', 12)
        # Remove any stray bold asterisks from the text
        text = line.replace('**', '')
        pdf.multi_cell(w=pdf.epw, h=H_12, text=text, align='J')

pdf.output('docs/Project_Report.pdf')
print("PDF strictly formatted and generated successfully.")
