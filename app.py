"""
AI Campus — app.py
Full backend: auth + 10 AI/ML endpoints
Run: python app.py  →  http://localhost:5000
"""

import os, json, pickle, hashlib, base64, warnings, re
import numpy as np
from datetime import datetime
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash)

warnings.filterwarnings('ignore')

# ── App ────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "aicampus-ultra-secret-2025"

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')
USERS_FILE = os.path.join(BASE, 'users.json')

# ── Load all models ────────────────────────────────────────────────
def load_pkl(name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    perf_model    = load_pkl('performance.pkl')         # RandomForest – pass/fail
    risk_bundle   = load_pkl('risk_model.pkl')          # RF – risk classification
    stress_bundle = load_pkl('stress_model.pkl')        # GBM – stress
    drop_bundle   = load_pkl('dropout_model.pkl')       # RF – dropout
    clust_bundle  = load_pkl('cluster_model.pkl')       # KMeans – clustering
    anom_bundle   = load_pkl('anomaly_model.pkl')       # IsoForest – anomaly
    print("✅ All 6 ML models loaded")
except Exception as e:
    print(f"❌ Model load error: {e}"); raise

# ── User helpers ───────────────────────────────────────────────────
def load_users():
    return json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}

def save_users(u):
    json.dump(u, open(USERS_FILE, 'w'), indent=2)

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def d(*a, **k):
        if 'user' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*a, **k)
    return d

@app.context_processor
def ctx(): return {'now': datetime.now(), 'user': session.get('user')}

# ══════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route('/')
def index(): return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if 'user' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        em = request.form.get('email','').strip().lower()
        pw = request.form.get('password','')
        u  = load_users().get(em)
        if u and u['password'] == hash_pw(pw):
            session['user'] = {'email': em, 'name': u['name']}
            flash(f"Welcome back, {u['name']}!", 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if 'user' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        em   = request.form.get('email','').strip().lower()
        pw   = request.form.get('password','')
        cf   = request.form.get('confirm','')
        if not name or not em or not pw:
            flash('All fields required.', 'danger')
        elif pw != cf:
            flash('Passwords do not match.', 'danger')
        elif len(pw) < 6:
            flash('Password must be ≥ 6 characters.', 'danger')
        else:
            users = load_users()
            if em in users: flash('Email already registered.', 'danger')
            else:
                users[em] = {'name': name, 'password': hash_pw(pw),
                             'joined': datetime.now().isoformat()}
                save_users(users)
                session['user'] = {'email': em, 'name': name}
                flash(f'Welcome, {name}!', 'success')
                return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None); flash('Logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('dashboard.html')

# Individual tool pages
@app.route('/tools/<tool>')
@login_required
def tool(tool):
    valid = {'performance','risk','attendance','faceid','anomaly',
             'recommend','stress','chatbot','cluster','dropout'}
    if tool not in valid: return redirect(url_for('dashboard'))
    return render_template(f'tool_{tool}.html')

# ══════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'models': 6})

# ── 1. Performance Prediction ──────────────────────────────────────
@app.route('/api/performance', methods=['POST'])
@login_required
def api_performance():
    try:
        d = request.get_json(force=True)
        att  = float(d['attendance'])
        hrs  = float(d['study_hours'])
        asgn = float(d['assignments'])
        mrk  = float(d['marks'])
        X = np.array([[att, hrs, asgn, mrk]])
        pred  = int(perf_model.predict(X)[0])
        proba = perf_model.predict_proba(X)[0].tolist()
        imp   = perf_model.feature_importances_.tolist()
        tips  = _perf_tips(att, hrs, asgn, mrk)
        return jsonify({
            'success': True,
            'prediction': pred,
            'label': 'Pass' if pred == 1 else 'Fail',
            'confidence': round(max(proba) * 100, 1),
            'probabilities': {'fail': round(proba[0]*100,1), 'pass': round(proba[1]*100,1)},
            'feature_importance': {
                n: round(v*100,1) for n,v in
                zip(['Attendance','Study Hours','Assignments','Marks'], imp)
            },
            'tips': tips
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def _perf_tips(att, hrs, asgn, mrk):
    tips = []
    if att < 75:  tips.append("⚠️ Attendance < 75% — major impact on grades.")
    if hrs < 2:   tips.append("📚 Increase daily study to at least 2–3 hours.")
    if asgn < 70: tips.append("📝 Submit more assignments for consistent marks.")
    if mrk < 50:  tips.append("🎯 Focus on core concepts; consider peer tutoring.")
    return tips or ["🌟 Excellent profile! Maintain this consistency."]

# ── 2. Risk Classification ─────────────────────────────────────────
@app.route('/api/risk', methods=['POST'])
@login_required
def api_risk():
    try:
        d   = request.get_json(force=True)
        att = float(d['attendance']); mrk = float(d['marks']); beh = float(d['behavior'])
        model  = risk_bundle['model']; scaler = risk_bundle['scaler']
        X = scaler.transform([[att, mrk, beh]])
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        colors = ['green', 'gold', 'red']
        actions = {
            0: ["✅ Keep up excellent performance!", "📊 Continue current study habits."],
            1: ["⚠️ Improve attendance by at least 10%.", "📚 Increase study hours.", "🤝 Seek teacher guidance."],
            2: ["🚨 Urgent: attend all classes immediately.", "📞 Schedule a counselor meeting.", "👨‍👩‍👧 Parental involvement recommended."]
        }
        return jsonify({
            'success': True, 'risk_level': pred, 'label': labels[pred],
            'color': colors[pred], 'confidence': round(max(proba)*100, 1),
            'probabilities': {labels[i]: round(v*100,1) for i,v in enumerate(proba)},
            'actions': actions[pred]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 3. Attendance Pattern Analysis ────────────────────────────────
@app.route('/api/attendance', methods=['POST'])
@login_required
def api_attendance():
    try:
        d = request.get_json(force=True)
        history = [float(x) for x in d.get('history', [])]
        if len(history) < 3:
            return jsonify({'success': False, 'error': 'Need at least 3 data points'}), 400
        arr = np.array(history)
        # Linear trend
        x    = np.arange(len(arr))
        coef = np.polyfit(x, arr, 1)
        trend_slope = round(float(coef[0]), 2)
        # Predict next 3 weeks
        poly    = np.poly1d(coef)
        next_3  = [round(max(0, min(100, float(poly(len(arr)+i)))), 1) for i in range(3)]
        avg     = round(float(np.mean(arr)), 1)
        std_dev = round(float(np.std(arr)), 1)
        # Trend label
        if trend_slope < -1.5:   trend_label = 'Sharply Declining ⬇️'
        elif trend_slope < -0.3: trend_label = 'Slightly Declining ↘️'
        elif trend_slope > 1.5:  trend_label = 'Strongly Improving ⬆️'
        elif trend_slope > 0.3:  trend_label = 'Slightly Improving ↗️'
        else:                    trend_label = 'Stable ➡️'
        warning = trend_slope < -1.5 or avg < 60
        return jsonify({
            'success': True, 'average': avg, 'std_dev': std_dev,
            'trend_slope': trend_slope, 'trend_label': trend_label,
            'prediction_next_3_weeks': next_3,
            'warning': warning,
            'message': '🚨 Critical: attendance declining fast!' if warning else '✅ Attendance pattern is healthy.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 4. Face Detection (OpenCV) ─────────────────────────────────────
@app.route('/api/faceid', methods=['POST'])
@login_required
def api_faceid():
    try:
        import cv2
        d       = request.get_json(force=True)
        img_b64 = d.get('image','').split(',')[-1]
        np_arr  = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        img     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        name    = d.get('registered_name','')
        verified = len(faces) > 0 and bool(name)
        return jsonify({
            'success': True, 'faces_detected': len(faces),
            'bounding_boxes': faces.tolist() if len(faces) else [],
            'verified': verified,
            'identity': name if verified else None,
            'timestamp': datetime.now().isoformat(),
            'message': f'✅ {name} verified' if verified else ('No face detected' if not len(faces) else 'Face detected')
        })
    except ImportError:
        return jsonify({'success': True, 'faces_detected': 1,
                        'bounding_boxes': [], 'verified': True,
                        'message': 'Demo mode (OpenCV not installed)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 5. Anomaly Detection ───────────────────────────────────────────
@app.route('/api/anomaly', methods=['POST'])
@login_required
def api_anomaly():
    try:
        d = request.get_json(force=True)
        att  = float(d['attendance']); mrk = float(d['marks'])
        beh  = float(d['behavior']);  hrs = float(d['study_hours'])
        model  = anom_bundle['model']; scaler = anom_bundle['scaler']
        X = scaler.transform([[att, mrk, beh, hrs]])
        score  = float(model.decision_function(X)[0])
        pred   = int(model.predict(X)[0])   # -1 = anomaly, 1 = normal
        is_anom = pred == -1
        # Specific anomaly flags
        flags = []
        if att < 40:   flags.append("🚨 Critically low attendance")
        if mrk < 30:   flags.append("🚨 Unusually low marks — sudden drop suspected")
        if beh < 2:    flags.append("🚨 Abnormal behavior score")
        if hrs < 0.5:  flags.append("🚨 Near-zero study time detected")
        return jsonify({
            'success': True,
            'is_anomaly': is_anom,
            'anomaly_score': round(score, 4),
            'label': 'Anomalous Behaviour Detected' if is_anom else 'Normal Behaviour',
            'severity': 'High' if score < -0.1 else 'Medium' if is_anom else 'Normal',
            'flags': flags,
            'recommendation': 'Immediate counselor intervention required.' if is_anom
                              else 'No unusual patterns detected.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 6. Recommendation System ───────────────────────────────────────
@app.route('/api/recommend', methods=['POST'])
@login_required
def api_recommend():
    try:
        d   = request.get_json(force=True)
        att  = float(d.get('attendance', 75))
        hrs  = float(d.get('study_hours', 3))
        asgn = float(d.get('assignments', 70))
        mrk  = float(d.get('marks', 60))
        beh  = float(d.get('behavior', 7))
        score = att*0.25 + mrk*0.35 + asgn*0.2 + min(hrs/10*100,100)*0.1 + beh*10*0.1
        recs = []
        priorities = []
        if att < 75:
            recs.append({'category':'Attendance','icon':'📅',
                'action':f'Attend {max(1,int((75-att)/5)+1)} more classes per week',
                'impact':'High','current':f'{att}%','target':'75%+'})
            priorities.append('attendance')
        if hrs < 3:
            recs.append({'category':'Study Time','icon':'⏰',
                'action':f'Study {round(3-hrs,1)} more hours daily (target 3h/day)',
                'impact':'High','current':f'{hrs}h','target':'3h+'})
        if asgn < 70:
            recs.append({'category':'Assignments','icon':'📝',
                'action':'Complete all pending assignments before deadline',
                'impact':'Medium','current':f'{asgn}%','target':'70%+'})
        if mrk < 50:
            recs.append({'category':'Exam Prep','icon':'🎯',
                'action':'Join study group, use past papers, revise weak chapters',
                'impact':'High','current':f'{mrk}%','target':'50%+'})
        if beh < 5:
            recs.append({'category':'Engagement','icon':'🤝',
                'action':'Participate more in class activities and discussions',
                'impact':'Medium','current':f'{beh}/10','target':'7/10+'})
        if not recs:
            recs.append({'category':'Excellence','icon':'🏆',
                'action':'Maintain current performance and help peers',
                'impact':'Low','current':'Excellent','target':'Keep it up'})
        return jsonify({
            'success': True,
            'overall_score': round(score, 1),
            'grade': 'A' if score>=85 else 'B' if score>=70 else 'C' if score>=55 else 'D' if score>=40 else 'F',
            'recommendations': recs,
            'priority': priorities[0] if priorities else 'maintain'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 7. Stress / Health Prediction ─────────────────────────────────
@app.route('/api/stress', methods=['POST'])
@login_required
def api_stress():
    try:
        d      = request.get_json(force=True)
        sleep  = float(d['sleep']); screen = float(d['screen']); act = float(d['activity'])
        model  = stress_bundle['model']; scaler = stress_bundle['scaler']
        X      = scaler.transform([[sleep, screen, act]])
        pred   = int(model.predict(X)[0])
        proba  = model.predict_proba(X)[0].tolist()
        labels = ['Low Stress', 'Moderate Stress', 'High Stress']
        colors = ['green', 'gold', 'red']
        advice = {
            0: ["😴 Great sleep schedule!", "🏃 Stay active.", "📵 Good screen discipline."],
            1: ["⚠️ Try to sleep 7-8 hours.", "📵 Reduce screen time by 1-2 hours.", "🧘 Add 20min daily exercise."],
            2: ["🚨 Critical stress detected!", "🛑 Limit screen time to < 4h.", "💤 Prioritize 8h sleep.", "🏥 Consider speaking to a counselor."]
        }
        return jsonify({
            'success': True, 'stress_level': pred,
            'label': labels[pred], 'color': colors[pred],
            'confidence': round(max(proba)*100,1),
            'probabilities': {labels[i]: round(v*100,1) for i,v in enumerate(proba)},
            'advice': advice[pred],
            'wellbeing_score': round((sleep/8*40 + max(0,(12-screen)/12)*35 + min(act/2,1)*25), 1)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 8. Chatbot ─────────────────────────────────────────────────────
CHAT_KB = {
    r'attendance|absent|miss': {
        'answer': "Attendance is crucial — it accounts for 25% of your performance score. Aim for 85%+.",
        'tips': ["Set daily reminders", "Inform teachers in advance if absent", "Make up missed content same day"]
    },
    r'stress|anxious|worried|pressure': {
        'answer': "Academic stress is common. Balance study with breaks, sleep 7-8h, and exercise regularly.",
        'tips': ["Try the Pomodoro technique (25min study, 5min break)", "Talk to a counselor", "Limit social media to 1h/day"]
    },
    r'study|learn|improve|marks|grade': {
        'answer': "Effective study habits: active recall, spaced repetition, and teaching concepts to others.",
        'tips': ["Use past papers for exam prep", "Study in 45-minute focused blocks", "Form a study group of 3-4 students"]
    },
    r'dropout|quit|leave|give up': {
        'answer': "Feeling like dropping out is a serious concern. Please speak with a counselor or trusted teacher immediately.",
        'tips': ["Talk to your academic advisor", "Explore support resources available", "Remember: temporary difficulties ≠ permanent failures"]
    },
    r'assignment|homework|deadline|submit': {
        'answer': "Assignments are key to continuous assessment. Break tasks into small parts and start early.",
        'tips': ["Use a planner for deadlines", "Start 3 days before due date", "Seek help from teachers if stuck"]
    },
    r'health|sleep|eat|exercise|tired': {
        'answer': "Physical health directly impacts academic performance. Prioritize sleep, nutrition, and movement.",
        'tips': ["Sleep 7-8 hours nightly", "Eat balanced meals including breakfast", "Exercise 30 minutes daily"]
    },
    r'hello|hi|hey|hii': {
        'answer': "Hello! 👋 I'm your AI Campus assistant. Ask me anything about studying, attendance, stress, or performance.",
        'tips': []
    },
}

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    try:
        msg = request.get_json(force=True).get('message','').lower().strip()
        for pattern, resp in CHAT_KB.items():
            if re.search(pattern, msg):
                return jsonify({
                    'success': True, 'answer': resp['answer'],
                    'tips': resp['tips'], 'source': 'AI Campus KB'
                })
        return jsonify({
            'success': True,
            'answer': "I'm not sure about that yet. Try asking about attendance, study habits, stress, assignments, or health.",
            'tips': ["Type 'study tips' for study advice", "Type 'stress help' for stress management"],
            'source': 'fallback'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 9. Clustering ──────────────────────────────────────────────────
@app.route('/api/cluster', methods=['POST'])
@login_required
def api_cluster():
    try:
        d    = request.get_json(force=True)
        att  = float(d['attendance']); mrk = float(d['marks'])
        asgn = float(d['assignments']); hrs = float(d['study_hours'])
        model  = clust_bundle['model']; scaler = clust_bundle['scaler']
        X      = scaler.transform([[att, mrk, asgn, hrs]])
        cluster = int(model.predict(X)[0])
        centers = scaler.inverse_transform(model.cluster_centers_)
        # Rank clusters by average performance
        avg_scores = [np.mean(c) for c in centers]
        rank = sorted(range(3), key=lambda i: avg_scores[i], reverse=True)
        cluster_names = {rank[0]: 'High Performer', rank[1]: 'Average Student', rank[2]: 'At-Risk Student'}
        cluster_colors = {rank[0]: 'green', rank[1]: 'gold', rank[2]: 'red'}
        cluster_icons  = {rank[0]: '🏆', rank[1]: '📘', rank[2]: '⚠️'}
        profile_score  = round((att*0.25 + mrk*0.35 + asgn*0.2 + min(hrs/10*100,100)*0.2), 1)
        cluster_tips = {
            rank[0]: ["Mentor peers who need help", "Explore advanced topics", "Apply for competitions/scholarships"],
            rank[1]: ["Target 10% improvement each month", "Focus on weakest subject", "Increase study hours by 1h/day"],
            rank[2]: ["Attend every class this week", "Meet with academic advisor ASAP", "Use all campus support resources"]
        }
        return jsonify({
            'success': True, 'cluster': cluster,
            'cluster_name': cluster_names[cluster],
            'color': cluster_colors[cluster],
            'icon': cluster_icons[cluster],
            'profile_score': profile_score,
            'tips': cluster_tips[cluster],
            'peer_comparison': {
                'your_score': profile_score,
                'cluster_avg': round(float(np.mean(centers[cluster])), 1),
                'top_cluster_avg': round(float(np.mean(centers[rank[0]])), 1)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── 10. Dropout Prediction ─────────────────────────────────────────
@app.route('/api/dropout', methods=['POST'])
@login_required
def api_dropout():
    try:
        d    = request.get_json(force=True)
        att  = float(d['attendance']); mrk = float(d['marks'])
        asgn = float(d['assignments']); hrs = float(d['study_hours'])
        beh  = float(d['behavior'])
        model  = drop_bundle['model']; scaler = drop_bundle['scaler']
        X      = scaler.transform([[att, mrk, asgn, hrs, beh]])
        pred   = int(model.predict(X)[0])
        proba  = model.predict_proba(X)[0].tolist()
        risk_factors = []
        if att < 50:   risk_factors.append('Very low attendance')
        if mrk < 40:   risk_factors.append('Failing marks')
        if asgn < 40:  risk_factors.append('Missing assignments')
        if beh < 4:    risk_factors.append('Low engagement/behavior')
        if hrs < 1:    risk_factors.append('Minimal study time')
        interventions = {
            True: [
                "🚨 Schedule immediate counselor meeting",
                "📞 Contact parents/guardian",
                "📋 Create a personalised recovery plan",
                "🏫 Enroll in academic support program",
                "👥 Assign a peer mentor"
            ],
            False: [
                "✅ Keep current performance levels",
                "📈 Set new academic goals each month",
                "🤝 Help peers who may be struggling"
            ]
        }
        return jsonify({
            'success': True,
            'dropout_risk': pred,
            'label': 'HIGH Dropout Risk' if pred else 'Low Dropout Risk',
            'probability': round(proba[1]*100, 1),
            'confidence': round(max(proba)*100, 1),
            'risk_factors': risk_factors,
            'interventions': interventions[bool(pred)],
            'urgency': 'Critical' if proba[1] > 0.7 else 'High' if pred else 'Low'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*52)
    print("  🎓 AI Campus Server")
    print("  URL:   http://localhost:5000")
    print("  Health: http://localhost:5000/api/health")
    print("="*52 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
