import customtkinter as ctk
from tkinter import ttk
from collections import Counter
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- LOGIC FUNCTIONS ---
def sdg_sort_key(sdg):
    if sdg.startswith("SDG"):
        return int(sdg.split(":")[0].split()[1])
    return 999

def group_titles_by_sdg(titles):
    grouped = {}
    for title in titles:
        sdg = classify(title)
        if sdg not in grouped:
            grouped[sdg] = []
        grouped[sdg].append(title)
    return grouped

def load_txt_data(file_path):
    data = []
    current_sdg = None
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line: continue
                if line.startswith("SDG"):
                    current_sdg = line
                elif "|" in line and current_sdg:
                    parts = line.split("|") 
                    title = parts[1].strip()
                    data.append((title, current_sdg))
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
    return data

# --- DATA LOADING ---
bscs_data = load_txt_data("bscs_sdg_data.txt")
bsit_data = load_txt_data("bsit_sdg_data.txt")  
bscs_titles = [t[0] for t in bscs_data]
bsit_titles = [t[0] for t in bsit_data]
all_titles = bsit_titles + bscs_titles

# --- AI MODEL SETUP ---
sdg_descriptions = { 
    "SDG 1: No Poverty": "poverty poor low income livelihood unemployment financial assistance aid social protection welfare subsidy beneficiaries 4ps pantawid pamilya cash transfer support services community development assistance", 
    "SDG 2: Zero Hunger": "food hunge fruit egg agricultural supply agriculture farming crops livestock nutrition food security irrigation harvest farm poultry feedmills feedmill", 
    "SDG 3: Good Health": "health cough  breast cancer dermatitis blister mri diabetes melanoma glaucoma bacterial herbal atopic lung membrane colon cancer pharmacy healthcare hospital clinic patient disease medical vaccination mental tuberculosis health wellness diagnosis treatment monitoring mellitus polyclinic medicine medicines eye dental", 
    "SDG 4: Quality Education": "education instruction basics educational quiz learning students school university teaching classroom e-learning LMS enrollment academic grading training course", 
    "SDG 5: Gender Equality": "gender women equality discrimination empowerment rights female violence inclusion feminism protection support", 
    "SDG 6: Clean Water": "water sanitation hygiene clean water wastewater supply filtration waterworks sewage drinking water purification monitoring", 
    "SDG 7: Clean Energy": "energy solar renewable solar wind electricity power grid battery sustainability energy consumption energy", 
    "SDG 8: Economic Growth": "jobs e-commerce bir tax employment economy business entrepreneurship productivity workforce labor income finance livelihood hiring", 
    "SDG 9: Industry & Innovation": "technology software application automation algorithm modification AI IoT engineering antenna development network database information RFID sensors sensor", 
    "SDG 10: Reduced Inequality": "impairment inequality inclusion disability accessibility poverty marginalized social equity support services inclusive", 
    "SDG 11: Sustainable Cities": "spatiotemporal city urban sandbox transport traffic housing infrastructure disaster evacuation smart city barangay navigation mapping GIS city parking municipality", 
    "SDG 12: Responsible Consumption": "waste garbage recycling plastic consumption production sustainability materials reuse segregation disposal", 
    "SDG 13: Climate Action": "climate change carbon emissions disaster flood typhoon environment monitoring hazard risk reduction prediction", 
    "SDG 14: Life Below Water": "ocean marine fish coral aquatic coastal water pollution fisheries underwater biodiversity mangrove reef",
    "SDG 15: Life on Land": "land forest rice seed field biodiversity wildlife soil agriculture plants animals ecosystem conservation farming monitoring crops", 
    "SDG 16: Peace & Justice": "justice secrecy stenography encryption identification cryptography misinformation security crime recognition spoofed deepfake law governance police legal records secure safety monitoring access control news", 
    "SDG 17: Partnerships": "partnership collaboration cooperation organizations integration shared multi agency coordination" }

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
sdg_vectors = vectorizer.fit_transform(sdg_descriptions.values())

def classify(title):
    title_lower = title.lower()
    if any(word in title_lower for word in ["4ps", "pantawid", "beneficiary", "donation", "financial assistance"]): return "SDG 1: No Poverty"
    if any(word in title_lower for word in ["egg", "agriculture", "farm"]): return "SDG 2: Zero Hunger"
    if any(word in title_lower for word in ["student", "grading", "enrollment", "basics", "lms"]): return "SDG 4: Quality Education"
    if any(word in title_lower for word in ["medical", "health", "eye", "vaccination", "dental", "hospital"]): return "SDG 3: Good Health"
    if any(word in title_lower for word in ["gender", "women", "female", "girls", "equality", "lgbt", "vaw"]): return "SDG 5: Gender Equality"
    if any(word in title_lower for word in ["job", "employment", "hiring", "business", "entrepreneur"]): return "SDG 8: Economic Growth"
    if any(word in title_lower for word in ["criminal"]): return "SDG 16: Peace & Justice"

    vec = vectorizer.transform([title])
    sim = cosine_similarity(vec, sdg_vectors)[0]
    max_index = sim.argmax()
    if sim[max_index] < 0.15: return "Unclassified"
    return list(sdg_descriptions.keys())[max_index]

# Pre-classify mapping
title_sdg_map = {title: classify(title) for title in all_titles}
counts = Counter(title_sdg_map.values())
max_count = max(counts.values()) if counts else 0
min_count = min(counts.values()) if counts else 0

def recommendation(sdg):
    c = counts.get(sdg, 0)
    if c == 0: return "No Data"
    if max_count == min_count: return "Balanced"
    score = (c - min_count) / (max_count - min_count)
    if score > 0.7: return "Overused"
    elif score < 0.3: return "Underexplored"
    return "Balanced"

# --- RECOMMENDATION LOGIC ---
def get_strategic_advice(sdg, status):
    if sdg == "Unclassified":
        sdg_list = "\n • ".join([s for s in sdg_descriptions.keys()])
        return (
            "Strategic Move: ALIGNMENT REQUIRED\n"
            "Your title doesn't clearly map to any SDG. To improve research impact, "
            "consider integrating keywords from these goals:\n • " + sdg_list
        )

    advice_map = {
        "Overused": (
            "Strategic Move: NICHE DOWN OR PIVOT\n"
            "This area is highly saturated. To be accepted, you must find a very specific "
            "gap or apply a unique technology that hasn't been used in previous titles."
        ),
        "Underexplored": (
            "Strategic Move: HIGH POTENTIAL AREA\n"
            "There is a lack of research here! This title has a high chance of being "
            "considered original and valuable to the department's SDG portfolio."
        ),
        "Balanced": (
            "Strategic Move: STRENGTHEN METHODOLOGY\n"
            "This is a steady research area. Ensure your technical implementation "
            "is robust to stand out from existing similar projects."
        )
    }
    return advice_map.get(status, "Enter a valid title for advice.")

# --- GUI SETUP ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("SDG Research Analyzer")
root.geometry("1200x850")

main_container = ctk.CTkFrame(root, fg_color="transparent")
main_container.pack(fill="both", expand=True)

# SIDEBAR
sidebar = ctk.CTkFrame(main_container, width=240, corner_radius=0)
sidebar.pack(side="left", fill="y")

sidebar_title = ctk.CTkLabel(sidebar, text="SDG Analyzer", font=ctk.CTkFont(size=20, weight="bold"))
sidebar_title.pack(pady=(30, 10))

try:
    logo_img = ctk.CTkImage(light_image=Image.open("logo.png"), size=(200, 200))
    logo_label = ctk.CTkLabel(sidebar, image=logo_img, text="")
    logo_label.pack(pady=20)
except:
    ctk.CTkLabel(sidebar, text="[Logo Placeholder]").pack(pady=20)

# SHOW WINDOW FUNCTIONS
def show_titles_window(title, titles):
    win = ctk.CTkToplevel(root)
    win.title(title)
    win.geometry("900x600")
    win.configure(fg_color="#1a1a1a")
    win.after(200, lambda: win.lift())

    header_frame = ctk.CTkFrame(win, fg_color="transparent")
    header_frame.pack(fill="x", padx=20, pady=(20, 10))
    
    ctk.CTkLabel(header_frame, text=title, font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
    ctk.CTkLabel(header_frame, text="Research Thesis Titles up until December 2025", font=ctk.CTkFont(size=14), text_color="gray").pack(anchor="w")

    style = ttk.Style()
    style.theme_use("default")
    style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", rowheight=35, borderwidth=0, font=("Segoe UI", 11))
    style.map("Treeview", background=[('selected', '#1f538d')])

    tree_container = ctk.CTkFrame(win, fg_color="#2b2b2b", corner_radius=15)
    tree_container.pack(fill="both", expand=True, padx=20, pady=20)

    tree = ttk.Treeview(tree_container, show="tree", style="Treeview")
    tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    scrollbar = ctk.CTkScrollbar(tree_container, command=tree.yview)
    scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)
    tree.configure(yscrollcommand=scrollbar.set)

    grouped = group_titles_by_sdg(titles)
    for sdg in sorted(grouped.keys(), key=sdg_sort_key):
        parent = tree.insert("", "end", text=f" 📁   {sdg}", open=False)
        for i, t in enumerate(grouped[sdg], start=1):
            tree.insert(parent, "end", text=f"      {i}. {t}")

def get_sdg_percentages():
    total_all = sum(counts.values())
    total_classified = sum(v for k, v in counts.items() if k != "Unclassified")
    sdg_percentages = {sdg: (count / total_classified) * 100 for sdg, count in counts.items() if sdg != "Unclassified"} if total_classified > 0 else {}
    unclassified_pct = (counts.get("Unclassified", 0) / total_all) * 100 if total_all > 0 else 0
    return sdg_percentages, total_classified, total_all, unclassified_pct

def show_percentages():
    win = ctk.CTkToplevel(root)
    win.title("SDG Research Distribution")
    win.geometry("550x700")
    win.configure(fg_color="#1a1a1a")
    win.after(200, lambda: win.lift())

    header_frame = ctk.CTkFrame(win, fg_color="transparent")
    header_frame.pack(fill="x", padx=30, pady=(30, 10))

    ctk.CTkLabel(header_frame, text="SDG Distribution", font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
    sdg_percentages, _, total_all, unp = get_sdg_percentages()
    ctk.CTkLabel(header_frame, text=f"Analyzing {total_all} total research titles", font=ctk.CTkFont(size=14), text_color="gray").pack(anchor="w")

    stats_container = ctk.CTkScrollableFrame(win, fg_color="#2b2b2b", corner_radius=15, label_text="Frequency Breakdown", label_font=ctk.CTkFont(weight="bold"))
    stats_container.pack(fill="both", expand=True, padx=30, pady=20)

    for sdg in sorted(sdg_percentages.keys(), key=sdg_sort_key):
        percentage = sdg_percentages[sdg]
        row = ctk.CTkFrame(stats_container, fg_color="transparent")
        row.pack(fill="x", pady=8, padx=10)
        ctk.CTkLabel(row, text=sdg, font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkLabel(row, text=f"{percentage:.1f}%", font=ctk.CTkFont(size=12, weight="bold"), text_color="#339af0").pack(side="right")
        progress = ctk.CTkProgressBar(stats_container, height=8, corner_radius=5)
        progress.set(percentage / 100)
        progress.pack(fill="x", padx=15, pady=(0, 15))

    footer = ctk.CTkFrame(win, fg_color="#333333", height=50, corner_radius=10)
    footer.pack(fill="x", padx=30, pady=(0, 30))
    ctk.CTkLabel(footer, text=f"Unclassified Data: {unp:.2f}%", font=ctk.CTkFont(size=13, slant="italic")).pack(pady=10)

# CONTENT AREA
content = ctk.CTkFrame(main_container, fg_color="transparent")
content.pack(side="right", fill="both", expand=True, padx=40)

header_label = ctk.CTkLabel(content, text="SDG Research Analyzer", font=ctk.CTkFont(size=32, weight="bold"))
header_label.pack(pady=(40, 20))

entry = ctk.CTkEntry(content, placeholder_text="Enter thesis title here...", width=550, height=45)
entry.pack(pady=10)

def analyze_input():
    print("Analyze clicked")  # debug

    title_text = entry.get().strip()
    if not title_text:
        result_label.configure(text="Please enter a title.", text_color="red")
        return
    
    sdg = classify(title_text)
    
    # Determine status
    if sdg == "Unclassified":
        status = "N/A"
        color = "#fab005"
        advice = get_strategic_advice(sdg, status)
    else:
        status = recommendation(sdg)
        color = {"Overused": "#ff6b6b", "Underexplored": "#51cf66", "Balanced": "#339af0"}.get(status, "white")
        advice = get_strategic_advice(sdg, status)
    
    # Update result label
    result_label.configure(text=f"Classified: {sdg}\nStatus: {status}", text_color=color)

    # Enable textbox
    similarity_box.configure(state="normal")
    similarity_box.delete("1.0", "end")

    # Split safely
    if "\n" in advice:
        header, body = advice.split("\n", 1)
    else:
        header = advice
        body = ""

    similarity_box.insert("end", f"➔ {header}\n", "highlight")
    similarity_box.insert("end", f"{body}\n")
    similarity_box.insert("end", "\n" + "─" * 60 + "\n\n")

    # Show similar titles
    if sdg != "Unclassified":
        similar_titles = [t for t, s in title_sdg_map.items() if s == sdg]
        if similar_titles:
            similarity_box.insert("end", f"Existing titles under {sdg}:\n\n", "highlight")
            for t in similar_titles:
                similarity_box.insert("end", f" • {t}\n\n")
        else:
            similarity_box.insert("end", "No existing titles found for this SDG.\n")

    similarity_box.see("1.0")
    similarity_box.configure(state="disabled")

    root.update_idletasks()

analyze_btn = ctk.CTkButton(content, text="Analyze Title", command=analyze_input, width=220, height=45, font=ctk.CTkFont(weight="bold"))
analyze_btn.pack(pady=10)

result_label = ctk.CTkLabel(content, text="Awaiting Input...", font=ctk.CTkFont(size=18, weight="bold"))
result_label.pack(pady=10)

similarity_frame = ctk.CTkFrame(content, corner_radius=15)
similarity_frame.pack(fill="both", expand=True, pady=(20, 40))

similarity_caption = ctk.CTkLabel(similarity_frame, text="Thesis Analysis & SDG Trends", font=ctk.CTkFont(size=16, weight="bold"))
similarity_caption.pack(pady=(15, 5))

similarity_box = ctk.CTkTextbox(similarity_frame, font=ctk.CTkFont(size=13), state="disabled", corner_radius=10)
similarity_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

# Sidebar Buttons
btn_options = {"width": 200, "height": 40, "font": ctk.CTkFont(weight="bold"), "corner_radius": 8}
ctk.CTkButton(sidebar, text="BSCS Titles", command=lambda: show_titles_window("BSCS Titles", bscs_titles), **btn_options).pack(pady=10, padx=20)
ctk.CTkButton(sidebar, text="BSIT Titles", command=lambda: show_titles_window("BSIT Titles", bsit_titles), **btn_options).pack(pady=10, padx=20)
ctk.CTkButton(sidebar, text="SDG Distribution %", command=show_percentages, **btn_options).pack(pady=10, padx=20)

root.after(0, lambda: root.state('zoomed'))
root.mainloop()