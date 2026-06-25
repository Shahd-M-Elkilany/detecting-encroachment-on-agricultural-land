"""
Generate a Final PDF with pipeline flowchart and KEMET1 classifier results.
"""
from fpdf import FPDF
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = BASE_DIR / "FINAL_REPORT_WITH_FLOWCHART.pdf"

class FlowchartPDF(FPDF):
    def header(self):
        if self.page_no() == 1: return
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Food Security ML Pipeline - Full Report with Flowchart", 0, 1, "C")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    # Helper to draw a flowchart block
    def draw_block(self, x, y, w, h, text1, text2, bg_color=(25, 50, 110)):
        self.set_fill_color(*bg_color)
        self.rect(x, y, w, h, 'F')
        self.set_xy(x, y + (h/2) - 5)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.multi_cell(w, 5, text1, align='C')
        self.set_xy(x, y + (h/2))
        self.set_font('Helvetica', '', 8)
        self.multi_cell(w, 5, text2, align='C')

    # Helper to draw an arrow
    def draw_arrow(self, x1, y1, x2, y2, label="", label_y_offset=-2):
        self.set_draw_color(100, 100, 100)
        self.set_line_width(0.5)
        self.line(x1, y1, x2, y2)
        # Draw arrow head (simple) if vertical or horizontal
        if y1 == y2:  # horizontal
            direction = 1 if x2 > x1 else -1
            self.line(x2, y2, x2 - direction*2, y2 - 1.5)
            self.line(x2, y2, x2 - direction*2, y2 + 1.5)
        elif x1 == x2:  # vertical
            direction = 1 if y2 > y1 else -1
            self.line(x2, y2, x2 - 1.5, y2 - direction*2)
            self.line(x2, y2, x2 + 1.5, y2 - direction*2)
        
        if label:
            orig_x, orig_y = self.get_x(), self.get_y()
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(80, 80, 80)
            
            if x1 == x2: # vertical line
                self.set_xy(x1 + 2, y1 + (y2-y1)/2 + label_y_offset)
                self.multi_cell(40, 3, label)
            else: # horizontal line
                self.set_xy(x1 + (x2-x1)/2 - 15, y1 + label_y_offset)
                self.cell(30, 3, label, align="C")
            
            self.set_xy(orig_x, orig_y)

def create_pdf():
    pdf = FlowchartPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # -----------------------------
    # PAGE 1: TITLE & FLOWCHART
    # -----------------------------
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 10, "Food Security AI System", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Complete Pipeline Flow & Project Details", 0, 1, 'C')
    
    # Draw Flowchart (custom)
    pdf.ln(15)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(35, 75, 135)
    pdf.cell(0, 8, "End-to-End Pipeline Flowchart", 0, 1, 'C')
    pdf.ln(5)
    
    # Base layout params
    # 4 columns, 2 rows
    start_x = 10
    start_y = 65
    bw = 40  # box width
    bh = 15  # box height
    gap_x = 6
    gap_y = 40
    
    # Helper definitions
    # Row 1
    pdf.draw_block(start_x, start_y, bw, bh, "STEP 01", "Data Acquisition", (41, 128, 185))
    pdf.draw_block(start_x + bw + gap_x, start_y, bw, bh, "STEP 02", "Cloud Detection", (41, 128, 185))
    pdf.draw_block(start_x + 2*(bw + gap_x), start_y, bw, bh, "STEP 03", "Cloud Removal", (41, 128, 185))
    pdf.draw_block(start_x + 3*(bw + gap_x), start_y, bw, bh, "STEP 04", "Spectral Indices", (41, 128, 185))
    
    # Row 2 (we draw it right to left or just standard)
    row2_y = start_y + gap_y + bh
    pdf.draw_block(start_x, row2_y, bw, bh, "STEP 05", "Change Detection", (39, 174, 96))
    pdf.draw_block(start_x + bw + gap_x, row2_y, bw, bh, "STEP 06", "Agri. Segment.", (39, 174, 96))
    pdf.draw_block(start_x + 2*(bw + gap_x), row2_y, bw, bh, "STEP 07", "Building Detect.", (39, 174, 96))
    pdf.draw_block(start_x + 3*(bw + gap_x), row2_y, bw, bh, "STEP 08", "FINAL OUTPUT", (192, 57, 43))
    
    # Draw arrows
    # R1: 1 -> 2
    pdf.draw_arrow(start_x + bw, start_y + bh/2, start_x + bw + gap_x, start_y + bh/2, "GeoTIFFs", -6)
    # R1: 2 -> 3
    pdf.draw_arrow(start_x + 2*bw + gap_x, start_y + bh/2, start_x + 2*bw + 2*gap_x, start_y + bh/2, "Cloud Mask", -6)
    # R1: 3 -> 4
    pdf.draw_arrow(start_x + 3*bw + 2*gap_x, start_y + bh/2, start_x + 3*bw + 3*gap_x, start_y + bh/2, "Clean GeoTIFF", -6)
    
    # Crossing arrows
    # Raw T1 & T2 to Step 3
    pdf.draw_arrow(start_x + bw/2, start_y + bh, start_x + bw/2, start_y + bh + 15)
    pdf.draw_arrow(start_x + bw/2, start_y + bh + 15, start_x + 2*bw + 1.5*gap_x, start_y + bh + 15)
    pdf.draw_arrow(start_x + 2*bw + 1.5*gap_x, start_y + bh + 15, start_x + 2*bw + 1.5*gap_x, start_y + bh, "Raw T1/T2", -20)
    
    # R1: 4 -> R2: 5
    pdf.draw_arrow(start_x + 3.5*bw + 3*gap_x, start_y + bh, start_x + 3.5*bw + 3*gap_x, row2_y - 20)
    pdf.draw_arrow(start_x + 3.5*bw + 3*gap_x, row2_y - 20, start_x + 0.5*bw, row2_y - 20)
    pdf.draw_arrow(start_x + 0.5*bw, row2_y - 20, start_x + 0.5*bw, row2_y, "NDVI/NDBI maps", -15)
    
    # R2: 5 -> 8
    pdf.draw_arrow(start_x + 0.5*bw, row2_y + bh, start_x + 0.5*bw, row2_y + bh + 15)
    pdf.draw_arrow(start_x + 0.5*bw, row2_y + bh + 15, start_x + 3.5*bw + 3*gap_x, row2_y + bh + 15, "Change Map (H,W)", 1)
    pdf.draw_arrow(start_x + 3.5*bw + 3*gap_x, row2_y + bh + 15, start_x + 3.5*bw + 3*gap_x, row2_y + bh)
    
    # T1 Clean to 6
    pdf.draw_arrow(start_x + 2.5*bw + 2*gap_x, start_y + bh, start_x + 2.5*bw + 2*gap_x, row2_y - 10)
    pdf.draw_arrow(start_x + 2.5*bw + 2*gap_x, row2_y - 10, start_x + 1.5*bw + gap_x, row2_y - 10)
    pdf.draw_arrow(start_x + 1.5*bw + gap_x, row2_y - 10, start_x + 1.5*bw + gap_x, row2_y, "T1 Clean RGB", -10)
    
    # 6 to 7
    pdf.draw_arrow(start_x + 1.5*bw + gap_x, row2_y + bh, start_x + 1.5*bw + gap_x, row2_y + bh + 8)
    pdf.draw_arrow(start_x + 1.5*bw + gap_x, row2_y + bh + 8, start_x + 2.5*bw + 2*gap_x, row2_y + bh + 8, "Agri Mask", 1)
    pdf.draw_arrow(start_x + 2.5*bw + 2*gap_x, row2_y + bh + 8, start_x + 2.5*bw + 2*gap_x, row2_y + bh)
    
    # 7 to 8
    pdf.draw_arrow(start_x + 2*bw + gap_x + bw, row2_y + bh/2, start_x + 3*bw + 3*gap_x, row2_y + bh/2, "Building Mask", -6)
    
    # -----------------------------
    # PAGE 2: PROJECT DETAILS 
    # -----------------------------
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 10, "1. The One Dataset Per Step", 0, 1, 'L')
    pdf.ln(2)
    
    # Table headers
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(25, 50, 110)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(15, 6, "Step", 1, 0, 'C', fill=True)
    pdf.cell(45, 6, "Name", 1, 0, 'C', fill=True)
    pdf.cell(75, 6, "Dataset Used", 1, 0, 'C', fill=True)
    pdf.cell(55, 6, "Pretrained Model", 1, 1, 'C', fill=True)
    
    # Table Rows
    rows = [
        ("01", "Data Acquisition", "Sentinel-2 L2A (from GEE)", "Google Earth Engine API"),
        ("02", "Cloud Detection", "38-Cloud (Landsat clouds)", "U-Net + ResNet34 (ImageNet)"),
        ("03", "Cloud Removal", "No dataset (uses Step 02 mask)", "OpenCV Telea Inpainting"),
        ("04", "Spectral Indices", "No dataset (pure formulas)", "NumPy Math (NDVI, NDBI)"),
        ("05", "Change Detection", "LEVIR-CD (Change pairs)", "ChangeFormer (Siamese)"),
        ("06", "Agriculture Segment", "ADE20K (150 classes)", "SegFormer-B4 (from HF)"),
        ("07", "Building Detection", "SpaceNet v2 (Building footprints)", "SAM (Segment Anything ViT-B)"),
        ("08", "Final Output", "No dataset (uses previous masks)", "OpenCV + GeoPandas"),
    ]
    
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(40, 40, 40)
    for i, r in enumerate(rows):
        pdf.set_fill_color(240, 243, 250) if i%2==0 else pdf.set_fill_color(255, 255, 255)
        pdf.cell(15, 6, r[0], 1, 0, 'C', fill=True)
        pdf.cell(45, 6, f" {r[1]}", 1, 0, 'L', fill=True)
        pdf.cell(75, 6, f" {r[2]}", 1, 0, 'L', fill=True)
        pdf.cell(55, 6, f" {r[3]}", 1, 1, 'L', fill=True)
        
    pdf.ln(10)
    
    # -----------------------------
    # PAGE 3: STEP BY STEP DETAILS
    # -----------------------------
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 10, "2. Detailed Step Specifications & Expected Outputs", 0, 1, 'L')
    pdf.ln(2)
    
    steps_info = [
        ("Step 01 - Data Acquisition", 
         "Goal: Download multi-temporal satellite images (T1=before, T2=after).\n"
         "Dataset: Sentinel-2 L2A via Google Earth Engine.\n"
         "Expected Output: Two Multi-band GeoTIFFs (T1 and T2) at 10m resolution."),
         
        ("Step 02 - Cloud Detection", 
         "Goal: Detect which pixels are covered by clouds.\n"
         "Model: U-Net with ResNet34 encoder pretrained on ImageNet.\n"
         "Dataset: Fine-tunable via 38-Cloud dataset.\n"
         "Expected Output: Binary cloud mask (H,W) highlighting cloud coverage."),
         
        ("Step 03 - Cloud Removal", 
         "Goal: Reconstruct the surface below the clouds.\n"
         "Model: OpenCV Telea inpainting algorithm.\n"
         "Expected Output: Cloud-free T1 and T2 GeoTIFFs across all bands."),
         
        ("Step 04 - Spectral Indices", 
         "Goal: Compute vegetation (NDVI) and built-up (NDBI) indices.\n"
         "Model: Pure mathematical band combination via NumPy.\n"
         "Expected Output: Float32 arrays for NDVI, NDBI for the models directly."),
         
        ("Step 05 - Change Detection", 
         "Goal: Identify pixels that changed between the two time periods.\n"
         "Model: ChangeFormer (Siamese Transformer).\n"
         "Dataset: Pretrained on LEVIR-CD.\n"
         "Expected Output: Binary change map where 1 = changed, 0 = unchanged."),
         
        ("Step 06 - Agriculture Segmentation", 
         "Goal: Extract pixels classified as agricultural land in the Before (T1) image.\n"
         "Model: SegFormer-B4.\n"
         "Dataset: Pretrained on ADE20K (filtering class IDs 9, 29, 92, etc.).\n"
         "Expected Output: Binary agricultural mask (1 = farmland)."),
         
        ("Step 07 - Building Detection", 
         "Goal: Segment instances of buildings exclusively within the changed agricultural regions.\n"
         "Model: SAM (Segment Anything Model by Meta AI) + YOLOv8-seg.\n"
         "Dataset: Pretrained on SA-1B, optional SpaceNet v2 fine-tuning.\n"
         "Expected Output: Building masks and polygon bounding coordinates."),
         
        ("Step 08 - Final Output", 
         "Goal: Synthesize all masks into an overarching visualization and report.\n"
         "Color Logic: RED (buildings on farmland), YELLOW (changed vegetation), GREEN (stable agriculture).\n"
         "Expected Output: High-resolution colored GeoTIFF, PNG preview, GeoJSON polygons, and a JSON area statistics report.")
    ]
    
    for title, desc in steps_info:
        # Prevent page breaking halfway
        if pdf.get_y() > 240:
            pdf.add_page()
            
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(35, 75, 135)
        pdf.cell(0, 8, title, 0, 1, 'L')
        
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(4)
        
    # -----------------------------
    # PAGE 4: KEMET1 RF CLASSIFIER
    # -----------------------------
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 10, "3. KEMET1 Encroachment Classifier (v4 Final)", 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 5,
        "A Random Forest classifier trained on 75 Egyptian tile locations x 4 time periods. "
        "Features: 48-dimensional vector (36 per-band stats + 11 derived indices + 1 prior-label). "
        "Post-processing: majority temporal consistency filter suppresses seasonal false positives.")
    pdf.ln(6)

    # Final metrics table
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(35, 75, 135)
    pdf.cell(0, 8, "Final Model Performance (RF depth=8, leaf=3)", 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(25, 50, 110)
    pdf.set_text_color(255, 255, 255)
    for hdr, w in [("Metric", 70), ("Val", 35), ("Test", 35), ("Pooled", 35), ("Notes", 0)]:
        pdf.cell(w, 6, hdr, 1, 0, 'C', fill=True)
    pdf.ln()

    metrics = [
        ("AUC",          "0.963",  "0.988", "0.990", "Post temporal consistency"),
        ("Recall",        "1.000",  "1.000", "1.000", "Zero missed encroachments"),
        ("Precision",     "0.667",  "0.667", "0.667", "3 FP per split"),
        ("F2 score",      "0.909",  "0.909", "0.909", "beta=2 weights recall 2x"),
        ("FP count",      "3",      "3",     "-",     "Down from 22 in v3 baseline"),
        ("FN count",      "0",      "0",     "-",     "Perfect recall maintained"),
        ("Threshold",     "0.29",   "0.29",  "0.29",  "F2-optimal on val set"),
    ]
    pdf.set_font('Helvetica', '', 8)
    for i, (m, v, t, p, n) in enumerate(metrics):
        fill = (i % 2 == 0)
        c = (240, 243, 250) if fill else (255, 255, 255)
        pdf.set_fill_color(*c)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(70, 6, f" {m}", 1, 0, 'L', fill=True)
        for val in (v, t, p):
            pdf.cell(35, 6, val, 1, 0, 'C', fill=True)
        pdf.cell(0, 6, f" {n}", 1, 1, 'L', fill=True)
    pdf.ln(6)

    # Improvement journey
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(35, 75, 135)
    pdf.cell(0, 8, "Improvement Journey: v3 Baseline to v4 Final", 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(25, 50, 110)
    pdf.set_text_color(255, 255, 255)
    for hdr, w in [("Version", 80), ("Test AUC", 35), ("FP (test)", 35), ("FN", 25), ("Change", 0)]:
        pdf.cell(w, 6, hdr, 1, 0, 'C', fill=True)
    pdf.ln()

    versions = [
        ("v3 - RF depth=10 (baseline)",         "0.747", "22", "0", "Starting point"),
        ("v4 Step 1 - + t1_is_pos feature",     "0.907", "10", "0", "+32% AUC, -55% FP"),
        ("v4 Final - + temporal consistency",   "0.988", "3",  "0", "+32% AUC, -86% FP"),
    ]
    pdf.set_font('Helvetica', '', 8)
    for i, (ver, auc, fp, fn, ch) in enumerate(versions):
        fill = (i % 2 == 0)
        c = (240, 243, 250) if fill else (255, 255, 255)
        pdf.set_fill_color(*c)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(80, 6, f" {ver}", 1, 0, 'L', fill=True)
        pdf.cell(35, 6, auc, 1, 0, 'C', fill=True)
        pdf.cell(35, 6, fp,  1, 0, 'C', fill=True)
        pdf.cell(25, 6, fn,  1, 0, 'C', fill=True)
        pdf.cell(0,  6, f" {ch}", 1, 1, 'L', fill=True)
    pdf.ln(6)

    # Top features
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(35, 75, 135)
    pdf.cell(0, 8, "Top 10 Feature Importances", 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(25, 50, 110)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(10,  6, "Rank", 1, 0, 'C', fill=True)
    pdf.cell(80,  6, "Feature",     1, 0, 'L', fill=True)
    pdf.cell(30,  6, "Importance",  1, 0, 'C', fill=True)
    pdf.cell(0,   6, "Band / Type", 1, 1, 'L', fill=True)

    top_feats = [
        (1,  "t1_is_pos",         "0.0768", "Prior label (new feature)"),
        (2,  "MNDWI_T1_std",      "0.0444", "MNDWI"),
        (3,  "MNDWI_diff_mean",   "0.0398", "MNDWI"),
        (4,  "MNDWI_T2_mean",     "0.0354", "MNDWI"),
        (5,  "NDVI_T2_std",       "0.0326", "NDVI"),
        (6,  "BSI_T2_mean",       "0.0292", "BSI"),
        (7,  "frac_changed_5pct", "0.0288", "Derived"),
        (8,  "NDBI_T2_mean",      "0.0273", "NDBI"),
        (9,  "NDBI_T1_std",       "0.0267", "NDBI"),
        (10, "NDVI_T2_mean",      "0.0259", "NDVI"),
    ]
    pdf.set_font('Helvetica', '', 8)
    for i, (rank, name, imp, band) in enumerate(top_feats):
        fill = (i % 2 == 0)
        c = (240, 243, 250) if fill else (255, 255, 255)
        pdf.set_fill_color(*c)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(10,  6, str(rank),   1, 0, 'C', fill=True)
        pdf.cell(80,  6, f" {name}",  1, 0, 'L', fill=True)
        pdf.cell(30,  6, imp,         1, 0, 'C', fill=True)
        pdf.cell(0,   6, f" {band}",  1, 1, 'L', fill=True)

    pdf.ln(8)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5,
        "Temporal consistency post-processing: if >= 2 of 3 consecutive pairs for a tile "
        "score above threshold, all scores are multiplied by 0.6 (seasonal dampening). "
        "This suppresses seasonal drift FPs while preserving true encroachment detections "
        "which typically only have 1 pair (neg to pos) flagging positive.")

    pdf.output(str(OUTPUT_PDF))
    print(f"Created PDF report at {OUTPUT_PDF}")

if __name__ == "__main__":
    create_pdf()
