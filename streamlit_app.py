# ================================================================
# Loan Approval Intelligence — Streamlit + Gemini AI Agent
# Light SaaS UI · st.tabs navigation · native bordered containers
# Fuzzy engine (FIS + GA) lives in fuzzy_engine.py — unchanged.
# ================================================================

import os
import json
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go            # never reassign `go`
from plotly.subplots import make_subplots
from google import genai
from google.genai import types

st.set_page_config(page_title="Loan Approval Intelligence",
                   page_icon="●", layout="wide", initial_sidebar_state="collapsed")

PAGE="#F4F6F8"; CARD="#FFFFFF"; BORDER="#E3E8EE"
INK="#16242B"; MUTED="#67767F"; ACCENT="#137A54"; ACC_SOFT="#E7F3EE"
GREEN="#15935F"; AMBER="#C98A1E"; RED="#D14343"
FONT="Inter, -apple-system, 'Segoe UI', system-ui, sans-serif"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp {{ background:{PAGE}; }}
html, body, [class*="css"] {{ font-family:{FONT}; color:{INK}; font-size:15px; }}
/* centre the page */
.main .block-container {{ padding-top:.35rem; padding-bottom:2rem; max-width:1240px;
    margin-left:auto !important; margin-right:auto !important; }}
#MainMenu, header, footer {{ visibility:hidden; }}
[data-testid="stSidebar"] {{ display:none; }}

/* brand bar */
.brandbar {{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }}
.blogo {{ width:40px; height:40px; border-radius:11px; background:{ACCENT};
         display:flex; align-items:center; justify-content:center; font-weight:800; color:#fff; font-size:18px; }}
.brandbar .t {{ font-size:18px; font-weight:800; color:{INK}; line-height:1.1; letter-spacing:-.3px; }}
.brandbar .s {{ font-size:11px; color:{MUTED}; letter-spacing:1px; text-transform:uppercase; font-weight:600; }}

/* tabs -> clean segmented nav */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{CARD}; border:1px solid {BORDER};
    border-radius:14px; padding:6px; box-shadow:0 1px 2px rgba(16,42,34,.04); }}
.stTabs [data-baseweb="tab-list"] button {{ height:40px; border-radius:10px; padding:0 20px;
    background:transparent; }}
.stTabs [data-baseweb="tab-list"] button p {{ font-size:14px; font-weight:600; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ background:{ACCENT} !important; }}
.stTabs [aria-selected="true"] p {{ color:#fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:14px; }}

/* page header */
.eyebrow {{ font-size:11.5px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase; color:{ACCENT}; margin-bottom:4px; }}
.h-title {{ font-size:26px; font-weight:800; letter-spacing:-.5px; color:{INK}; margin:0 0 6px; }}
.h-desc {{ font-size:14px; color:{MUTED}; margin-bottom:14px; max-width:820px; line-height:1.55; }}

/* native bordered container -> card */
[data-testid="stVerticalBlockBorderWrapper"] {{ background:{CARD}; border:1px solid {BORDER} !important;
    border-radius:16px; box-shadow:0 1px 2px rgba(16,42,34,.04); }}
.ch {{ font-size:16px; font-weight:700; color:{INK}; margin:2px 0 2px; }}
.cs {{ font-size:13px; color:{MUTED}; margin-bottom:14px; }}

/* metric cards */
.mwrap {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:16px; }}
.mcard {{ flex:1; min-width:160px; background:{CARD}; border:1px solid {BORDER};
         border-radius:14px; padding:16px 18px; box-shadow:0 1px 2px rgba(16,42,34,.04); }}
.mcard .k {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.9px; font-weight:700; }}
.mcard .v {{ font-size:25px; font-weight:800; color:{INK}; margin-top:6px; letter-spacing:-.5px; }}
.mcard .v .ac {{ color:{ACCENT}; }}
.mcard .d {{ font-size:12px; color:{MUTED}; margin-top:3px; }}

/* form */
[data-testid="stWidgetLabel"] p {{ color:{INK} !important; font-weight:600 !important; font-size:13px !important; }}
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextInput input {{
    background:#FBFCFD !important; color:{INK} !important; border:1px solid {BORDER} !important;
    border-radius:10px !important; font-size:14.5px !important; }}
.stNumberInput button {{ background:#FBFCFD !important; border:1px solid {BORDER} !important; color:{INK} !important; }}
.stButton > button, .stFormSubmitButton > button {{ background:{ACCENT}; color:#fff; border:none;
    border-radius:11px; font-weight:700; padding:11px 0; font-size:15px; }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ filter:brightness(1.08); }}
div[data-testid="column"] .stButton > button {{ background:#FFFFFF; color:{INK};
    border:1px solid {BORDER}; font-weight:600; font-size:13px; padding:10px 12px; }}
div[data-testid="column"] .stButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT}; filter:none; }}

/* badge / reasons / pipeline / trace */
.badge {{ border-radius:13px; padding:16px 18px; border:1px solid; display:flex;
         align-items:center; justify-content:space-between; margin-bottom:6px; }}
.badge .lab {{ font-size:21px; font-weight:800; }}
.badge .sc {{ font-size:14px; font-weight:700; }}
.approve {{ background:{ACC_SOFT}; border-color:#BFE3D2; color:{GREEN}; }}
.review  {{ background:#FBF2DF; border-color:#EEDCAE; color:{AMBER}; }}
.reject  {{ background:#FBE7E7; border-color:#F1C4C4; color:{RED}; }}
.reason {{ display:flex; align-items:flex-start; gap:9px; padding:8px 0; border-bottom:1px solid {BORDER}; font-size:14px; }}
.reason:last-child {{ border-bottom:none; }}
.rdot {{ width:8px; height:8px; border-radius:50%; margin-top:6px; flex:none; }}
.rpos {{ background:{GREEN}; }} .rneg {{ background:{RED}; }} .rneu {{ background:{AMBER}; }}
.pipe {{ display:flex; align-items:center; flex-wrap:wrap; gap:4px; }}
.pstep {{ display:flex; align-items:center; gap:8px; background:#F7FAF8; border:1px solid #DCEAE3;
         border-radius:10px; padding:8px 13px; }}
.pnum {{ width:20px; height:20px; border-radius:50%; background:{ACCENT}; color:#fff;
        font-size:11px; font-weight:700; display:flex; align-items:center; justify-content:center; }}
.pstep .pl {{ font-size:13px; font-weight:600; color:{INK}; }}
.psep {{ color:#C4D3CB; }}
.trace {{ background:#F7FAF8; border:1px solid #DCEAE3; border-left:3px solid {ACCENT};
         border-radius:10px; padding:11px 14px; margin:8px 0; font-size:13px; color:{INK}; }}
.trace .tn {{ color:{ACCENT}; font-weight:700; font-family:'SF Mono',Menlo,monospace; }}
.trace code {{ background:#EAF2EE; padding:1px 6px; border-radius:5px; font-size:12px; }}
.callout {{ background:{ACC_SOFT}; border:1px solid #BFE3D2; border-radius:11px;
           padding:12px 15px; font-size:13.5px; color:#0E5D40; margin-bottom:16px; line-height:1.5; }}
.kv {{ font-size:14px; line-height:1.95; }}

/* compact hero + agent timeline */
.hero {{ background:{CARD}; border:1px solid {BORDER}; border-radius:18px; padding:16px 18px; margin-bottom:14px;
        box-shadow:0 1px 3px rgba(16,42,34,.05); }}
.agent-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:18px; padding:18px 22px; margin:8px 0 16px;
        box-shadow:0 1px 3px rgba(16,42,34,.05); }}
.agent-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }}
.agent-title {{ font-size:17px; font-weight:800; color:{INK}; }}
.agent-status {{ font-size:12px; color:{ACCENT}; font-weight:700; background:{ACC_SOFT}; border:1px solid #BFE3D2; padding:6px 10px; border-radius:999px; }}
.timeline {{ position:relative; display:grid; grid-template-columns:repeat(6,1fr); gap:14px; align-items:start; }}
.timeline:before {{ content:""; position:absolute; left:6%; right:6%; top:25px; height:3px; background:#DCEAE3; border-radius:99px; }}
.timeline:after {{ content:""; position:absolute; left:6%; top:25px; height:3px; width:var(--fill,0%); background:{ACCENT}; border-radius:99px; transition:width .45s ease; }}
.titem {{ position:relative; z-index:1; text-align:center; min-width:0; }}
.tcircle {{ width:52px; height:52px; border-radius:999px; margin:0 auto 8px; display:flex; align-items:center; justify-content:center;
          background:#F7FAF8; border:2px solid #DCEAE3; color:{MUTED}; font-weight:800; box-shadow:0 0 0 6px {CARD}; }}
.titem.done .tcircle {{ background:{ACCENT}; border-color:{ACCENT}; color:white; }}
.titem.active .tcircle {{ background:{ACCENT}; border-color:{ACCENT}; color:white; animation:pulse 1.1s infinite; }}
.tlabel {{ font-size:13px; font-weight:800; color:{INK}; white-space:nowrap; }}
.tdesc {{ font-size:11.5px; color:{MUTED}; line-height:1.25; margin-top:3px; }}
@keyframes pulse {{ 0% {{ box-shadow:0 0 0 6px {CARD},0 0 0 8px rgba(19,122,84,.20); }} 70% {{ box-shadow:0 0 0 6px {CARD},0 0 0 17px rgba(19,122,84,0); }} 100% {{ box-shadow:0 0 0 6px {CARD},0 0 0 8px rgba(19,122,84,0); }} }}
.info-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:10px 0 12px; }}
.infobox {{ background:#FBFCFD; border:1px solid {BORDER}; border-radius:12px; padding:10px 12px; }}
.infobox .ik {{ font-size:10.5px; letter-spacing:.9px; text-transform:uppercase; font-weight:800; color:{MUTED}; }}
.infobox .iv {{ font-size:16px; font-weight:800; color:{INK}; margin-top:3px; }}
@media (max-width: 900px) {{ .timeline {{ grid-template-columns:repeat(3,1fr); }} .timeline:before,.timeline:after{{display:none}} .info-grid{{grid-template-columns:1fr}} }}

</style>
""", unsafe_allow_html=True)


# ---------------- engine ----------------
@st.cache_resource(show_spinner="Running GA optimisation (first load only, ~15s)…")
def load_engine():
    import fuzzy_engine as fe
    return fe

fe = load_engine()
ga = fe.ga_result


# ---------------- tools ----------------
def compute_risk_detailed(income, credit, ratio, default, emp):
    default_val = 1 if str(default).lower() in ("yes", "y", "1", "true") else 0
    fe.risk_sim.input["income"]  = float(income)
    fe.risk_sim.input["credit"]  = float(credit)
    fe.risk_sim.input["ratio"]   = float(ratio)
    fe.risk_sim.input["default"] = default_val
    fe.risk_sim.input["emp_exp"] = float(emp)
    fe.risk_sim.compute()
    score = float(fe.risk_sim.output["risk"])
    decision = "APPROVE" if score < 40 else ("REVIEW" if score < 70 else "REJECT")
    md = fe.membership_degree
    deg = {
        "income_high": round(md(float(income), fe.best_mfs["inc_u"], fe.best_mfs["inc"]["high"]), 2),
        "income_low":  round(md(float(income), fe.best_mfs["inc_u"], fe.best_mfs["inc"]["low"]), 2),
        "ratio_high":  round(md(float(ratio),  fe.best_mfs["rat_u"], fe.best_mfs["rat"]["high"]), 2),
        "ratio_low":   round(md(float(ratio),  fe.best_mfs["rat_u"], fe.best_mfs["rat"]["low"]), 2),
        "emp_experienced": round(md(float(emp), fe.best_mfs["emp_u"], fe.best_mfs["emp"]["experienced"]), 2),
        "emp_junior":      round(md(float(emp), fe.best_mfs["emp_u"], fe.best_mfs["emp"]["junior"]), 2),
        "credit_risk": round(fe.ctos_risk(float(credit)), 2),
        "defaulted":   default_val,
    }
    return {"risk_score": round(score, 1), "decision": decision, "membership_degrees": deg,
            "rule_weights": {"ratio": fe.W_RATIO, "credit": fe.W_CREDIT,
                             "income": fe.W_INCOME, "default": fe.W_DEFAULT, "emp": fe.W_EMP}}

def lookup_ctos_category(credit):
    c = float(credit)
    cat = ("Very Poor" if c <= 449 else "Poor" if c <= 549 else
           "Fair" if c <= 649 else "Good" if c <= 749 else "Excellent")
    return {"credit_score": c, "category": cat, "risk_value": round(fe.ctos_risk(c), 2)}

def reasons_from_degrees(d):
    out = []
    cr = d["credit_risk"]
    if cr <= 0.25:   out.append(("pos", "Credit score is strong — low credit risk"))
    elif cr >= 0.55: out.append(("neg", "Credit score is weak — high credit risk"))
    else:            out.append(("neu", "Credit score is moderate"))
    if d["ratio_high"] >= 0.5:   out.append(("neg", "Loan-to-income ratio sits in the high-risk band"))
    elif d["ratio_low"] >= 0.5:  out.append(("pos", "Loan-to-income ratio is comfortably low"))
    else:                        out.append(("neu", "Loan-to-income ratio is moderate"))
    if d["emp_junior"] >= 0.5:        out.append(("neg", "Employment experience is still junior"))
    elif d["emp_experienced"] >= 0.5: out.append(("pos", "Employment experience is solid"))
    else:                             out.append(("neu", "Employment experience is mid-level"))
    if d["defaulted"] == 1: out.append(("neg", "Has a previous loan default on record"))
    else:                   out.append(("pos", "No previous loan default"))
    return out


# ---------------- gemini agent ----------------
TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="calculate_loan_risk",
        description=("Run the GA-optimised fuzzy logic engine on one applicant. Returns risk score "
                     "0-100, decision (APPROVE/REVIEW/REJECT) and membership degrees. Call when you "
                     "have all 5 variables. May be called MULTIPLE times for what-if."),
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "income": types.Schema(type=types.Type.NUMBER, description="Annual income RM"),
            "credit": types.Schema(type=types.Type.NUMBER, description="CTOS score 300-850"),
            "ratio":  types.Schema(type=types.Type.NUMBER, description="Loan fraction of income 0-1"),
            "default":types.Schema(type=types.Type.STRING, description="'Yes' or 'No'"),
            "emp":    types.Schema(type=types.Type.NUMBER, description="Employment years"),
        }, required=["income", "credit", "ratio", "default", "emp"]),
    ),
    types.FunctionDeclaration(
        name="lookup_ctos_category",
        description="Look up the CTOS credit category for a credit score, as extra evidence.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "credit": types.Schema(type=types.Type.NUMBER, description="CTOS score")},
            required=["credit"]),
    ),
])

def execute_tool(name, args):
    if name == "calculate_loan_risk":
        return compute_risk_detailed(args["income"], args["credit"], args["ratio"],
                                     args["default"], args["emp"])
    if name == "lookup_ctos_category":
        return lookup_ctos_category(args["credit"])
    return {"error": f"unknown tool {name}"}

SYSTEM = """You are a Malaysian loan officer AI agent. You assess loan applications using a
fuzzy-logic risk engine that you call as a tool.

Follow this loop and briefly show each step:
1. OBSERVE  - note which of the 5 variables you have: income (RM/year), credit (300-850),
   loan ratio (0-1), previous default (Yes/No), employment experience (years).
2. COLLECT  - if a variable is missing, ask ONE concise question. You may call
   lookup_ctos_category to add credit evidence.
3. REASON   - when you have all 5, decide to call the risk engine.
4. CALL TOOL- call calculate_loan_risk. If REJECT or REVIEW, try a what-if: call again with a
   small realistic change to see if it would flip to a better decision.
5. EXPLAIN  - explain using the membership_degrees the tool returned (do NOT invent reasons).
6. ACT      - give the final decision and, if not APPROVE, 2-3 concrete improvement steps.

Reply in the user's language. Be concise."""

MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]

def get_client():
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    return genai.Client(api_key=key) if key else None

def run_agent(client, history, user_msg, render):
    contents = history + [types.Content(role="user", parts=[types.Part(text=user_msg)])]
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM, tools=[TOOLS],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    model = st.session_state.get("gem_model", MODELS[0])
    try:
        for _ in range(8):
            resp = client.models.generate_content(model=model, contents=contents, config=cfg)
            cand = resp.candidates[0]; parts = cand.content.parts or []
            for p in parts:
                if getattr(p, "text", None): render("text", p.text.strip())
            fcs = [p.function_call for p in parts if getattr(p, "function_call", None)]
            if fcs:
                contents.append(cand.content); rp = []
                for fc in fcs:
                    args = dict(fc.args); render("tool", (fc.name, args))
                    result = execute_tool(fc.name, args); render("result", (fc.name, result))
                    rp.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
                contents.append(types.Content(role="user", parts=rp)); continue
            else:
                contents.append(cand.content); break
        return contents
    except Exception as e:
        render("error", str(e))
        return history   # keep prior history intact on failure


# ---------------- plotly (light) ----------------
def _light(fig, h=None):
    fig.update_layout(template="plotly_white", paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT, color=MUTED, size=13), margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12)),
        hoverlabel=dict(bgcolor="#fff", bordercolor=BORDER, font_size=12))
    if h: fig.update_layout(height=h)
    fig.update_xaxes(gridcolor="#EEF1F4", zerolinecolor="#E3E8EE", linecolor="#E3E8EE")
    fig.update_yaxes(gridcolor="#EEF1F4", zerolinecolor="#E3E8EE", linecolor="#E3E8EE")
    return fig

def risk_gauge(score, decision):
    col = {"APPROVE": GREEN, "REVIEW": AMBER, "REJECT": RED}[decision]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        number=dict(font=dict(size=40, color=col), suffix="<span style='font-size:14px;color:#67767F'> /100</span>"),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(size=10)),
            bar=dict(color=col, thickness=0.3), bgcolor="#F4F6F8", borderwidth=0,
            steps=[dict(range=[0, 40], color="rgba(21,147,95,.12)"),
                   dict(range=[40, 70], color="rgba(201,138,30,.12)"),
                   dict(range=[70, 100], color="rgba(209,67,67,.12)")],
            threshold=dict(line=dict(color=col, width=3), thickness=0.8, value=score))))
    return _light(fig, h=210)

def convergence_figure():
    fit = fe.ga_result["fit_history"]; x = list(range(len(fit)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fit, mode="lines", name="Best fitness",
        line=dict(color=ACCENT, width=3), fill="tozeroy", fillcolor="rgba(19,122,84,.08)",
        hovertemplate="Gen %{x}<br>Fitness %{y:.1f}<extra></extra>"))
    fig.update_layout(title=dict(text="GA Convergence — lower is better", font=dict(color=INK, size=15)))
    fig.update_yaxes(range=[min(fit) * 0.97, max(fit) * 1.01])
    return _light(fig, h=340)

def mf_figure():
    specs = [
        (fe.inc_u, {"Low": fe.income_fuzzy['low'].mf, "Medium": fe.income_fuzzy['medium'].mf,
                    "High": fe.income_fuzzy['high'].mf}, "Annual Income · GA-tuned", [RED, AMBER, GREEN]),
        (fe.credit_fuzzy.universe,
         {"Very Poor": fe.credit_fuzzy['very_poor'].mf, "Poor": fe.credit_fuzzy['poor'].mf,
          "Fair": fe.credit_fuzzy['fair'].mf, "Good": fe.credit_fuzzy['good'].mf,
          "Excellent": fe.credit_fuzzy['excellent'].mf},
         "Credit Score · fixed (CTOS)", ["#8B2020", RED, AMBER, GREEN, "#0E5D40"]),
        (fe.rat_u, {"Low": fe.ratio_fuzzy['low'].mf, "Medium": fe.ratio_fuzzy['medium'].mf,
                    "High": fe.ratio_fuzzy['high'].mf}, "Loan-to-Income Ratio · GA-tuned", [GREEN, AMBER, RED]),
        (fe.default_fuzzy.universe,
         {"No Default": fe.default_fuzzy['no'].mf, "Defaulted": fe.default_fuzzy['yes'].mf},
         "Previous Default · fixed (binary)", [GREEN, RED]),
        (fe.emp_u, {"Junior": fe.emp_fuzzy['junior'].mf, "Mid-Level": fe.emp_fuzzy['mid'].mf,
                    "Experienced": fe.emp_fuzzy['experienced'].mf}, "Employment Experience · GA-tuned", [RED, AMBER, GREEN]),
        (fe.risk.universe, {"Low Risk": fe.risk['low'].mf, "Medium Risk": fe.risk['medium'].mf,
                            "High Risk": fe.risk['high'].mf}, "Output · Risk Score", [GREEN, AMBER, RED]),
    ]
    fig = make_subplots(rows=3, cols=2, subplot_titles=[s[2] for s in specs],
                        vertical_spacing=0.13, horizontal_spacing=0.09)
    for i, (u, mfs, _t, colors) in enumerate(specs):
        r, c = i // 2 + 1, i % 2 + 1
        for (lbl, vals), col in zip(mfs.items(), colors):
            fig.add_trace(go.Scatter(x=list(u), y=list(vals), mode="lines", name=lbl,
                line=dict(color=col, width=2.4), legendgroup=f"g{i}", showlegend=True,
                hovertemplate=f"{lbl}<br>x %{{x:.2f}}<br>mu %{{y:.2f}}<extra></extra>"), row=r, col=c)
    fig.update_layout(template="plotly_white", paper_bgcolor=CARD, plot_bgcolor=CARD, height=850,
        font=dict(family=FONT, color=MUTED, size=11), margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#EEF1F4", linecolor="#E3E8EE")
    fig.update_yaxes(gridcolor="#EEF1F4", linecolor="#E3E8EE", range=[-0.05, 1.1])
    for ann in fig.layout.annotations:
        ann.font = dict(color=INK, size=13, family=FONT)
    return fig

PCFG = {"displayModeBar": False, "responsive": True}




def render_agent_process(active=0, mode="idle"):
    steps = [
        ("Observe", "Read applicant inputs"),
        ("Collect", "Check CTOS evidence"),
        ("Reason", "Validate required data"),
        ("Call Engine", "Run fuzzy risk tool"),
        ("Explain", "Use membership degrees"),
        ("Act", "Return decision & advice"),
    ]
    fill = 0 if active <= 1 else min(100, (active-1) / (len(steps)-1) * 88)
    status = "Ready" if mode == "idle" else ("Processing" if mode == "running" else "Completed")
    html = f'<div class="agent-card"><div class="agent-head"><div><div class="agent-title">AI Agent Process</div><div class="cs" style="margin:2px 0 0;">Agentic loop integrated into Evaluate Applicant</div></div><div class="agent-status">{status}</div></div>'
    html += f'<div class="timeline" style="--fill:{fill}%">'
    for i, (lab, desc) in enumerate(steps, start=1):
        cls = "done" if (mode == "done" or i < active) else ("active" if i == active and mode == "running" else "")
        mark = "✓" if (mode == "done" or i < active) else str(i)
        html += f'<div class="titem {cls}"><div class="tcircle">{mark}</div><div class="tlabel">{lab}</div><div class="tdesc">{desc}</div></div>'
    html += '</div></div>'
    return html

# ================================================================
# LAYOUT
# ================================================================
st.markdown('<div class="brandbar"><div class="blogo">L</div>'
            '<div><div class="t">Loan Approval Intelligence</div>'
            '<div class="s">Fuzzy Logic · Genetic Algorithm · AI Agent</div></div></div>',
            unsafe_allow_html=True)

def header(eyebrow, title, desc):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="h-title">{title}</div>'
                f'<div class="h-desc">{desc}</div>', unsafe_allow_html=True)

tab_assess, tab_ga, tab_mf = st.tabs(
    ["Assess", "GA Results", "Membership Functions"])


# ---------------- ASSESS ----------------
with tab_assess:
    st.markdown('<div class="hero"><div class="eyebrow">Risk Assessment</div><div class="h-title">Evaluate Applicant</div>'
                '<div class="h-desc">The AI agent is now integrated into the main evaluation flow: it observes inputs, checks CTOS evidence, calls the GA-optimised fuzzy engine, explains the fuzzy membership degrees, and returns the final decision.</div></div>',
                unsafe_allow_html=True)

    process_slot = st.empty()
    process_slot.markdown(render_agent_process(active=0, mode="idle"), unsafe_allow_html=True)

    col_in, col_out = st.columns([0.95, 1.05], gap="large")

    with col_in:
        with st.container(border=True):
            st.markdown('<div class="ch">Applicant Information</div>'
                        '<div class="cs">Five inputs used by the fuzzy inference engine.</div>',
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            income = a.number_input("Annual income (RM)", 0, 1_000_000, 65000, step=1000)
            credit = b.number_input("Credit score (CTOS)", 300, 850, 720)
            ratio  = a.number_input("Loan as % of income (0–1)", 0.0, 1.0, 0.18, step=0.01)
            emp    = b.number_input("Employment experience (yrs)", 0.0, 50.0, 4.0, step=0.5)
            default = st.selectbox("Previous loan default", ["No", "Yes"])
            evaluate_clicked = st.button("Evaluate Applicant", use_container_width=True, type="primary")
            st.markdown('<div class="cs" style="margin:12px 0 0;">CTOS bands: 300–449 Very Poor · '
                        '450–549 Poor · 550–649 Fair · 650–749 Good · 750–850 Excellent</div>',
                        unsafe_allow_html=True)

    with col_out:
        with st.container(border=True):
            st.markdown('<div class="ch">Assessment Result</div>'
                        '<div class="cs">Decision, risk gauge, tool trace, and reasons drawn from the model.</div>',
                        unsafe_allow_html=True)
            if evaluate_clicked:
                step_text = ["Observe", "Collect", "Reason", "Call Engine", "Explain", "Act"]
                for i in range(1, 7):
                    process_slot.markdown(render_agent_process(active=i, mode="running"), unsafe_allow_html=True)
                    time.sleep(0.16)
                process_slot.markdown(render_agent_process(active=6, mode="done"), unsafe_allow_html=True)

                ctos_info = lookup_ctos_category(credit)
                r = compute_risk_detailed(income, credit, ratio, default, emp)
                d = r["membership_degrees"]

                cls = {"APPROVE": "approve", "REVIEW": "review", "REJECT": "reject"}[r["decision"]]
                st.markdown(f'<div class="badge {cls}"><span class="lab">{r["decision"]}</span>'
                            f'<span class="sc">Risk {r["risk_score"]} / 100</span></div>',
                            unsafe_allow_html=True)

                st.markdown(f'<div class="info-grid">'
                            f'<div class="infobox"><div class="ik">CTOS Evidence</div><div class="iv">{credit} · {ctos_info["category"]}</div></div>'
                            f'<div class="infobox"><div class="ik">Loan Ratio</div><div class="iv">{ratio:.2f}</div></div>'
                            f'<div class="infobox"><div class="ik">Employment</div><div class="iv">{emp:.1f} years</div></div>'
                            f'</div>', unsafe_allow_html=True)

                st.plotly_chart(risk_gauge(r["risk_score"], r["decision"]),
                                use_container_width=True, config=PCFG)

                st.markdown(f'<div class="trace"><span class="tn">collect</span> CTOS {credit} → '
                            f'<b>{ctos_info["category"]}</b> credit band</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="trace"><span class="tn">call tool</span> calculate_loan_risk '
                            f'<code>income={income}, credit={credit}, ratio={ratio}, default={default}, emp={emp}</code></div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="trace"><span class="tn">explain</span> membership degrees '
                            f'<code>income_high={d["income_high"]}, ratio_high={d["ratio_high"]}, emp_junior={d["emp_junior"]}, credit_risk={d["credit_risk"]}</code></div>',
                            unsafe_allow_html=True)

                st.markdown('<div style="font-weight:800;font-size:14px;margin:10px 0 4px;">AI explanation</div>',
                            unsafe_allow_html=True)
                cmap = {"pos": "rpos", "neg": "rneg", "neu": "rneu"}
                html = "".join(f'<div class="reason"><span class="rdot {cmap[k]}"></span>{t}</div>'
                               for k, t in reasons_from_degrees(r["membership_degrees"]))
                st.markdown(html, unsafe_allow_html=True)

                if r["decision"] in ("REVIEW", "REJECT"):
                    suggestion = "Reduce the loan-to-income ratio, improve the CTOS score, or build longer employment stability before reapplying."
                else:
                    suggestion = "Applicant profile is acceptable. Maintain credit quality and repayment discipline."
                st.markdown(f'<div class="callout"><b>Action advice:</b> {suggestion}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align:center;color:#8A99A3;padding:58px 10px;font-size:14px;">'
                            'Waiting for applicant input.<br>Click '
                            '<b style="color:#137A54">Evaluate Applicant</b> to run the integrated AI agent process.</div>',
                            unsafe_allow_html=True)


# ---------------- GA RESULTS ----------------
with tab_ga:
    header("Data-Driven Optimisation", "Genetic Algorithm",
           "The GA tunes the fuzzy membership-function breakpoints by minimising the error between "
           "the model's risk output and the real loan-status labels.")

    st.markdown(
        f'<div class="mwrap">'
        f'<div class="mcard"><div class="k">Initial Fitness</div><div class="v">{ga["init_fitness"]:.0f}</div>'
        f'<div class="d">generation 0</div></div>'
        f'<div class="mcard"><div class="k">Final Fitness</div><div class="v"><span class="ac">{ga["best_fitness"]:.0f}</span></div>'
        f'<div class="d">best individual</div></div>'
        f'<div class="mcard"><div class="k">Improvement</div><div class="v"><span class="ac">{ga["improvement_pct"]:.1f}%</span></div>'
        f'<div class="d">lower = better fit</div></div>'
        f'<div class="mcard"><div class="k">Reference Data</div><div class="v">{len(fe.REF_STATUS):,}</div>'
        f'<div class="d">{fe.n_approved:,} approved · {fe.n_rejected:,} rejected</div></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="callout">Fitness dropped <b>{ga["improvement_pct"]:.1f}%</b> '
        f'({ga["init_fitness"]:.0f} → {ga["best_fitness"]:.0f}) over {fe.GA_GENS} generations — the '
        f'GA-tuned membership functions match the reference loan data far better than the initial '
        f'hand-set functions. Lower fitness = closer to ground truth.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="ch">Convergence</div>'
                    f'<div class="cs">Population {fe.GA_POP} · {fe.GA_GENS} generations · two-point crossover · '
                    f'adaptive mutation · tournament k={fe.GA_TOURNAMENT} · fitness = Σ(loan_status − fuzzy_risk)²</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(convergence_figure(), use_container_width=True, config=PCFG)

    with st.container(border=True):
        st.markdown('<div class="ch">Optimised MF Breakpoints</div>'
                    '<div class="cs">Evolved parameters for the three GA-tuned variables. '
                    'Credit and default are fixed, not evolved.</div>', unsafe_allow_html=True)
        st.dataframe({
            "Variable": ["Annual Income (RM)", "Loan Ratio (0–1)", "Employment Exp (yrs)"],
            "a · Low→Mid": [f"{fe.inc_a:,.0f}", f"{fe.rat_a:.4f}", f"{fe.emp_a:.2f}"],
            "b · Mid peak": [f"{fe.inc_b:,.0f}", f"{fe.rat_b:.4f}", f"{fe.emp_b:.2f}"],
            "c · Mid→High": [f"{fe.inc_c:,.0f}", f"{fe.rat_c:.4f}", f"{fe.emp_c:.2f}"],
        }, use_container_width=True, hide_index=True)


# ---------------- MEMBERSHIP FUNCTIONS ----------------
with tab_mf:
    header("Fuzzy Sets", "Membership Functions",
           "The fuzzy sets the inference engine uses. Three variables are tuned by the GA; credit and "
           "default are fixed by standard. Hover any curve to read its membership value.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="ch" style="color:#137A54;">GA-tuned variables</div>'
                        '<div class="cs">Data-driven · breakpoints evolved by the genetic algorithm</div>'
                        '<div class="kv">• Annual Income<br>• Loan-to-Income Ratio<br>• Employment Experience</div>',
                        unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="ch">Fixed variables</div>'
                        '<div class="cs">Knowledge-driven · set by standard, not evolved</div>'
                        '<div class="kv">• CTOS Credit Score<br>• Previous Loan Default</div>',
                        unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="ch">All membership functions</div>'
                    '<div class="cs">Hover any curve to read its exact membership value.</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(mf_figure(), use_container_width=True, config=PCFG)
