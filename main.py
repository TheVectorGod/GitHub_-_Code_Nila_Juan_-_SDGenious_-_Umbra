from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

app = Flask(__name__)
app.run(host="0.0.0.0", port=5000)
CORS(app) 

app.config["JWT_SECRET_KEY"] = "this-is-a-very-long-and-secure-32-byte-key-14457600234943576931" 
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# TEMP DATABASE
users = {}
saved_titles = {}
user_plans = {
    "free_user": "free",
    "premium_user": "premium"
}

# --- CORE LOGIC ---
sdg_descriptions = { 
    "SDG 1: No Poverty": "poverty poor low income livelihood unemployment financial assistance aid social protection welfare subsidy beneficiaries 4ps pantawid pamilya cash transfer support services community development assistance", 
    "SDG 2: Zero Hunger": "food hunge fruit egg agricultural supply agriculture farming crops livestock nutrition food security irrigation harvest farm poultry feedmills feedmill", 
    "SDG 3: Good Health": "health cough breast cancer dermatitis blister mri diabetes melanoma glaucoma bacterial herbal atopic lung membrane colon cancer pharmacy healthcare hospital clinic patient disease medical vaccination mental tuberculosis health wellness diagnosis treatment monitoring mellitus polyclinic medicine medicines eye dental", 
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
    "SDG 17: Partnerships": "partnership collaboration cooperation organizations integration shared multi agency coordination" 
}

# Initialize AI Model
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
sdg_vectors = vectorizer.fit_transform(sdg_descriptions.values())

def classify(title):
    vec = vectorizer.transform([title])
    sim = cosine_similarity(vec, sdg_vectors)[0]
    max_index = sim.argmax()
    if sim[max_index] < 0.15: return "Unclassified"
    return list(sdg_descriptions.keys())[max_index]

def get_classified_data(file_path):
    classified_groups = {}
    try:
        if not os.path.exists(file_path):
            return {"Unclassified": ["System Test: Unclassified Title 1"], "SDG 4: Quality Education": ["System Test: Title 2"]}
            
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if "|" in line:
                    parts = line.split("|") 
                    if len(parts) > 1:
                        title = parts[1].strip()
                        sdg = classify(title)
                        if sdg not in classified_groups:
                            classified_groups[sdg] = []
                        classified_groups[sdg].append(title)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return classified_groups

@app.route("/delete_title", methods=["POST"])
@jwt_required()
def delete_title():
    user = get_jwt_identity()
    data = request.json
    title_to_remove = data.get("title")

    if user in saved_titles and title_to_remove in saved_titles[user]:
        saved_titles[user].remove(title_to_remove)
        return jsonify({"msg": "Title removed successfully"}), 200
    
    return jsonify({"msg": "Title not found"}), 404

def get_real_market_counts():
    bscs_data = get_classified_data("bscs_sdg_data.txt")
    bsit_data = get_classified_data("bsit_sdg_data.txt")
    
    real_counts = {}
    for dataset in [bscs_data, bsit_data]:
        for sdg, titles in dataset.items():
            if sdg != "Unclassified":
                real_counts[sdg] = real_counts.get(sdg, 0) + len(titles)
                
    if not real_counts: return {}, 0, 0
    return real_counts, max(real_counts.values()), min(real_counts.values())

def get_status(sdg):
    market_counts, max_count, min_count = get_real_market_counts()
    c = market_counts.get(sdg, 0)
    if c == 0: return "Underexplored" 
    
    score = (c - min_count) / (max_count - min_count) if max_count != min_count else 0
    
    if score > 0.7: return "Overused"
    elif score < 0.3: return "Underexplored"
    return "Balanced"

def get_strategic_advice(status, sdg):
    if sdg == "Unclassified":
        return "Strategic Move: ALIGNMENT REQUIRED. Your title doesn't clearly map to any SDG. To improve research impact, consider integrating keywords from recognized UN goals."
        
    advice_map = {
        "Overused": "Strategic Move: NICHE DOWN OR PIVOT. This area is highly saturated. To be accepted, you must find a very specific gap or apply a unique technology.",
        "Underexplored": "Strategic Move: HIGH POTENTIAL AREA. There is a lack of research here! This title has a high chance of being considered original and valuable to the department.",
        "Balanced": "Strategic Move: STRENGTHEN METHODOLOGY. This is a steady research area. Ensure your technical implementation is robust to stand out."
    }
    return advice_map.get(status, "Enter a valid title for advice.")

# -------------------------
# AUTH ROUTES
# -------------------------
@app.route("/register", methods=["POST"])
def register():
    # Use get_json(silent=True) to prevent crashing on malformed JSON
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"msg": "Please provide both username and password"}), 400
        
    if username in users:
        return jsonify({"msg": "User already exists"}), 400

    # Hash password and store user
    users[username] = bcrypt.generate_password_hash(password).decode('utf-8')
    user_plans[username] = 'free'
    return jsonify({"msg": "Account created successfully!"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')

    if username in users and bcrypt.check_password_hash(users[username], password):
        token = create_access_token(identity=username)
        return jsonify({
            "access_token": token, 
            "plan": user_plans.get(username, 'free')
        }), 200
    
    return jsonify({"msg": "Invalid username or password"}), 401
# -------------------------
# MAIN APP ROUTES
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze_title():
    data = request.json
    title = data.get('title', '')
    
    sdg = classify(title)
    status = "N/A" if sdg == "Unclassified" else get_status(sdg)
    advice = get_strategic_advice(status, sdg)

    return jsonify({"sdg": sdg, "status": status, "advice": advice})

@app.route('/get_titles/<major>', methods=['GET'])
def get_titles(major):
    file_name = "bscs_sdg_data.txt" if major == "bscs" else "bsit_sdg_data.txt"
    grouped_data = get_classified_data(file_name)
    return jsonify(grouped_data)

@app.route("/save_title", methods=["POST"])
@jwt_required()
def save_title():
    user = get_jwt_identity()
    data = request.json
    title = data.get("title")
    if not title: return jsonify({"msg": "No title to save"}), 400

    if user not in saved_titles: saved_titles[user] = []
    saved_titles[user].append(title)
    return jsonify({"msg": "Saved to library!"}), 200

@app.route("/my_titles", methods=["GET"])
@jwt_required()
def get_my_titles():
    user = get_jwt_identity()
    return jsonify(saved_titles.get(user, []))

@app.route('/get_distribution', methods=['GET'])
def get_distribution():
    bscs_data = get_classified_data("bscs_sdg_data.txt")
    bsit_data = get_classified_data("bsit_sdg_data.txt")
    
    combined_counts = {}
    total_titles = 0
    
    for dataset in [bscs_data, bsit_data]:
        for sdg, titles in dataset.items():
            combined_counts[sdg] = combined_counts.get(sdg, 0) + len(titles)
            total_titles += len(titles)
    
    if total_titles == 0: return jsonify({})
    
    percentages = {sdg: round((count / total_titles) * 100, 1) for sdg, count in combined_counts.items()}
    return jsonify(percentages)

#Premium access
def check_premium(user):
    return user_plans.get(user, "free") == "premium"

@app.route("/similarity_check", methods=["POST"])
@jwt_required()
def similarity_check():
    user = get_jwt_identity()

    if not check_premium(user):
        return jsonify({"msg": "Premium required"}), 403

    data = request.json
    title = data.get("title")

    bscs_data = get_classified_data("bscs_sdg_data.txt")
    bsit_data = get_classified_data("bsit_sdg_data.txt")

    all_titles = []
    for dataset in [bscs_data, bsit_data]:
        for titles in dataset.values():
            all_titles.extend(titles)

    vec = vectorizer.transform([title])
    existing_vecs = vectorizer.transform(all_titles)

    similarities = cosine_similarity(vec, existing_vecs)[0]

    max_sim = max(similarities) if len(similarities) > 0 else 0

    risk = "Low"
    if max_sim > 0.7:
        risk = "High"
    elif max_sim > 0.4:
        risk = "Medium"

    return jsonify({
        "similarity_score": round(max_sim, 2),
        "plagiarism_risk": risk
    })

@app.route("/generate_title", methods=["POST"])
@jwt_required()
def generate_title():
    user = get_jwt_identity()

    if not check_premium(user):
        return jsonify({"msg": "Premium required"}), 403

    data = request.json
    sdg = data.get("sdg")

    tech_terms = [
        "AI-Based", "IoT-Enabled", "Machine Learning",
        "Web-Based", "Mobile Application", "Blockchain-Based"
    ]

    actions = [
        "Monitoring System", "Prediction Model",
        "Management System", "Decision Support System",
        "Analytics Platform"
    ]

    sdg_context = {
        "SDG 1: No Poverty": ["Livelihood", "Financial Aid", "Beneficiary"],
        "SDG 2: Zero Hunger": ["Food Supply", "Agriculture", "Nutrition"],
        "SDG 3: Good Health": ["Healthcare", "Disease", "Patient Monitoring"],
        "SDG 4: Quality Education": ["Student Performance", "Learning", "Academic"],
        "SDG 5: Gender Equality": ["Gender Violence", "Women Empowerment", "Equality"],
        "SDG 6: Clean Water": ["Water Quality", "Sanitation", "Water Supply"],
        "SDG 7: Clean Energy": ["Energy Consumption", "Solar Power", "Electricity"],
        "SDG 8: Economic Growth": ["Business", "Employment", "Entrepreneurship"],
        "SDG 9: Industry & Innovation": ["Automation", "Smart Systems", "Technology"],
        "SDG 10: Reduced Inequality": ["Accessibility", "Inclusion", "Disability Support"],
        "SDG 11: Sustainable Cities": ["Traffic", "Flood", "Urban Planning"],
        "SDG 12: Responsible Consumption": ["Waste Management", "Recycling", "Sustainability"],
        "SDG 13: Climate Action": ["Disaster", "Flood Prediction", "Climate Monitoring"],
        "SDG 14: Life Below Water": ["Marine Life", "Fisheries", "Water Pollution"],
        "SDG 15: Life on Land": ["Forest", "Agriculture", "Wildlife"],
        "SDG 16: Peace & Justice": ["Crime", "Security", "Legal Records"],
        "SDG 17: Partnerships": ["Collaboration", "Data Sharing", "Integration"]
    }

    context_list = sdg_context.get(sdg, ["System"])
    context = random.choice(context_list)

    title = f"{random.choice(tech_terms)} {context} {random.choice(actions)}"

    return jsonify({"generated_title": title})

@app.route("/research_insights", methods=["POST"])
@jwt_required()
def research_insights():
    user = get_jwt_identity()

    if not check_premium(user):
        return jsonify({"msg": "Premium required"}), 403

    data = request.json
    title = data.get("title")

    sdg = classify(title)
    status = get_status(sdg)

    approval = {
        "Overused": "Low",
        "Balanced": "Medium",
        "Underexplored": "High"
    }.get(status, "Unknown")

    trend = {
        "Overused": "Declining",
        "Balanced": "Stable",
        "Underexplored": "Rising"
    }.get(status, "Unknown")

    return jsonify({
        "sdg": sdg,
        "trend": trend,
        "approval_likelihood": approval
    })

@app.route("/upgrade", methods=["POST"])
@jwt_required()
def upgrade():
    current_user = get_jwt_identity()
    user_plans[current_user] = "premium"
    return jsonify({"msg": "Upgraded to premium successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
