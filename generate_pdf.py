import os
from fpdf import FPDF

class ProjectReportPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        # Skip page number on the cover page (page 1)
        if self.page_no() == 1:
            return
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

# --- PAGE 1: COVER PAGE ---
pdf.add_page()

# Spacer to push title down to middle-top
pdf.ln(60)

# Title (Large, Bold, Dark Blue/Gray)
pdf.set_text_color(20, 50, 80)
pdf.set_font('Times', 'B', 20)
title_text = "IMAGE CLASSIFICATION AND EXPLAINABLE IDENTIFICATION OF AI-GENERATED SYNTHETIC IMAGES"
pdf.multi_cell(w=pdf.epw, h=8, text=title_text, align='C')

pdf.ln(4)

# Subtitle (Teal/Green)
pdf.set_text_color(0, 150, 136)
pdf.set_font('Times', 'I', 14)
subtitle_text = "Dual-Model Deep Learning with Grad-CAM Explainability"
pdf.multi_cell(w=pdf.epw, h=6, text=subtitle_text, align='C')

pdf.ln(10)

# Draw Horizontal Lines
x_start = 38.1
x_end = 210 - 25.4
y_line = pdf.get_y()

# Thick Teal Line
pdf.set_draw_color(0, 150, 136)
pdf.set_line_width(1.5)
pdf.line(x_start, y_line, x_end, y_line)

pdf.ln(10)
y_line2 = pdf.get_y()

# Thin Gray Line
pdf.set_draw_color(150, 150, 150)
pdf.set_line_width(0.5)
pdf.line(x_start, y_line2, x_end, y_line2)

pdf.ln(15)

# Reset text color to black for metadata
pdf.set_text_color(0, 0, 0)
pdf.set_font('Times', 'B', 12)
pdf.write(6, "Cohort: ")
pdf.set_font('Times', '', 12)
pdf.write(6, "XTRAGRAD Advanced Cohort - Batch 1\n")

pdf.ln(2)

pdf.set_font('Times', 'B', 12)
pdf.write(6, "Focus Group: ")
pdf.set_font('Times', '', 12)
pdf.write(6, "Image Authentication & Explainable AI Team\n")

# Start Chapter 1 on a new page (Page 2)
pdf.add_page()
pdf.set_text_color(0, 0, 0)

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

in_code_block = False
for line in lines:
    line = line.strip()
    if line.startswith('```'):
        in_code_block = not in_code_block
        continue
        
    if in_code_block:
        continue
        
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
