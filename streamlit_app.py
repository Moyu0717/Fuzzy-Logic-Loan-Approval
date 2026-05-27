# ================================================================
# Loan Approval Intelligence — Streamlit + Gemini AI Agent
# Light SaaS UI · sidebar navigation · interactive Plotly.
# Fuzzy engine (FIS + GA) lives in fuzzy_engine.py — unchanged.
# ================================================================

import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go            # NOTE: never reassign `go`
from plotly.subplots import make_subplots
from google import genai
from google.genai import types

st.set_page_config(page_title="Loan Approval Intelligence",
                   page_icon="●", layout="wide", initial_sidebar_state="expanded")

# ---- design tokens (light SaaS) ----
PAGE    = "#F4F6F8"
SIDEBAR = "#102A22"
CARD    = "#FFFFFF"
BORDER  = "#E3E8EE"
INK     = "#16242B"
MUTED   = "#67767F"
ACCENT  = "#137A54"
ACC_SOFT= "#E7F3EE"
GREEN   = "#15935F"
AMBER   = "#C98A1E"
RED     = "#D14343"
FONT    = "Inter, -apple-system, 'Segoe UI', system-ui, sans-serif"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {{ background:{PAGE}; }}
html, body, [class*="css"] {{ font-family:{FONT}; color:{INK}; font-size:15px; }}
.main .block-container {{ padding-top:2rem; padding-bottom:3rem; max-width:1180px; }}
#MainMenu, header, footer {{ visibility:hidden; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background:{SIDEBAR}; border-right:none; }}
[data-testid="stSidebar"] * {{ color:#D7E4DD; }}
.sb-brand {{ display:flex; align-items:center; gap:11px; padding:6px 4px 18px; }}
.sb-logo {{ width:38px; height:38px; border-radius:10px; background:{ACCENT};
           display:flex; align-items:center; justify-content:center;
           font-weight:800; color:#fff; font-size:17px; }}
.sb-brand .t {{ font-size:15px; font-weight:700; color:#fff; line-height:1.1; }}
.sb-brand .s {{ font-size:10.5px; color:#7FA295; letter-spacing:1px; text-transform:uppercase; }}
.sb-section {{ font-size:10.5px; color:#6E8C80; letter-spacing:1.3px; text-transform:uppercase;
              font-weight:700; margin:8px 4px 4px; }}

/* radio -> nav list */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:3px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    background:transparent; border-radius:10px; padding:10px 13px; margin:0;
    transition:background .12s ease; cursor:pointer; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:rgba(255,255,255,.06); }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size:14.5px; font-weight:600; color:#CFE0D8; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background:{ACCENT}; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color:#fff; }}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] div:first-child {{ display:none; }}

/* ---------- page header ---------- */
.eyebrow {{ font-size:11.5px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase;
           color:{ACCENT}; margin-bottom:4px; }}
.h-title {{ font-size:30px; font-weight:800; letter-spacing:-.6px; color:{INK}; margin:0 0 6px; }}
.h-desc {{ font-size:15px; color:{MUTED}; margin-bottom:22px; max-width:720px; line-height:1.55; }}

/* ---------- cards ---------- */
.card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:16px;
        padding:22px 24px; box-shadow:0 1px 2px rgba(16,42,34,.04); margin-bottom:16px; }}
.card h3 {{ margin:0 0 3px; font-size:17px; font-weight:700; color:{INK}; }}
.card .sub {{ font-size:13.5px; color:{MUTED}; margin-bottom:18px; }}

/* metric cards */
.mwrap {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:18px; }}
.mcard {{ flex:1; min-width:160px; background:{CARD}; border:1px solid {BORDER};
         border-radius:14px; padding:16px 18px; box-shadow:0 1px 2px rgba(16,42,34,.04); }}
.mcard .k {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.9px; font-weight:700; }}
.mcard .v {{ font-size:26px; font-weight:800; color:{INK}; margin-top:6px; letter-spacing:-.5px; }}
.mcard .v .ac {{ color:{ACCENT}; }}
.mcard .d {{ font-size:12px; color:{MUTED}; margin-top:3px; }}

/* form controls */
[data-testid="stWidgetLabel"] p {{ color:{INK} !important; font-weight:600 !important; font-size:13px !important; }}
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextInput input {{
    background:#FBFCFD !important; color:{INK} !important; border:1px solid {BORDER} !important;
    border-radius:10px !important; font-size:14.5px !important; }}
.stNumberInput button {{ background:#FBFCFD !important; border:1px solid {BORDER} !important; color:{INK} !important; }}

.stButton > button {{ background:{ACCENT}; color:#fff; border:none; border-radius:11px;
    font-weight:700; padding:12px 0; font-size:15px; transition:filter .12s ease; }}
.stButton > button:hover {{ filter:brightness(1.08); }}

/* decision badge */
.badge {{ border-radius:14px; padding:18px 20px; border:1px solid; display:flex;
         align-items:center; justify-content:space-between; }}
.badge .lab {{ font-size:22px; font-weight:800; letter-spacing:.3px; }}
.badge .sc {{ font-size:14px; font-weight:700; }}
.approve {{ background:{ACC_SOFT}; border-color:#BFE3D2; color:{GREEN}; }}
.review  {{ background:#FBF2DF; border-color:#EEDcae; color:{AMBER}; }}
.reject  {{ background:#FBE7E7; border-color:#F1C4C4; color:{RED}; }}

/* reason list */
.reason {{ display:flex; align-items:flex-start; gap:9px; padding:8px 0;
          border-bottom:1px solid {BORDER}; font-size:14px; color:{INK}; }}
.reason:last-child {{ border-bottom:none; }}
.rdot {{ width:8px; height:8px; border-radius:50%; margin-top:6px; flex:none; }}
.rpos {{ background:{GREEN}; }} .rneg {{ background:{RED}; }} .rneu {{ background:{AMBER}; }}

/* pipeline */
.pipe {{ display:flex; gap:0; align-items:center; flex-wrap:wrap; margin:4px 0 6px; }}
.pstep {{ display:flex; align-items:center; gap:8px; background:#F7FAF8; border:1px solid #DCEAE3;
         border-radius:10px; padding:9px 14px; }}
.pnum {{ width:20px; height:20px; border-radius:50%; background:{ACCENT}; color:#fff;
        font-size:11px; font-weight:700; display:flex; align-items:center; justify-content:center; }}
.pstep .pl {{ font-size:13px; font-weight:600; color:{INK}; }}
.psep {{ color:#C4D3CB; padding:0 8px; font-size:14px; }}

/* agent trace */
.trace {{ background:#F7FAF8; border:1px solid #DCEAE3; border-left:3px solid {ACCENT};
         border-radius:10px; padding:11px 14px; margin:8px 0; font-size:13px; color:{INK}; }}
.trace .tn {{ color:{ACCENT}; font-weight:700; font-family:'SF Mono',Menlo,monospace; }}
.trace code {{ background:#EAF2EE; padding:1px 6px; border-radius:5px; font-size:12px; }}

/* example chips override (secondary look) */
div[data-testid="column"] .stButton > button {{ background:#FFFFFF; color:{INK};
    border:1px solid {BORDER}; font-weight:600; font-size:13px; padding:10px 12px; }}
div[data-testid="column"] .stButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT}; filter:none; }}

.stDataFrame {{ border:1px solid {BORDER}; border-radius:12px; }}
hr {{ border-color:{BORDER}; }}
.callout {{ background:{ACC_SOFT}; border:1px solid #BFE3D2; border-radius:11px;
           padding:12px 15px; font-size:13.5px; color:#0E5D40; margin:6px 0 14px; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------
# Load fuzzy engine ONCE
# ----------------------------------------------------------------
@st.cache_resource(show_spinner="Running GA optimisation (first load only, ~15s)…")
def load_engine():
    import fuzzy_engine as fe
    return fe

fe = load_engine()
ga = fe.ga_result


# ----------------------------------------------------------------
# TOOLS — wrap the existing engine
# ----------------------------------------------------------------
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
    """Plain-language reasons derived from the model's own membership degrees."""
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


# ----------------------------------------------------------------
# Gemini agent
# ----------------------------------------------------------------
TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="calculate_loan_risk",
        description=("Run the GA-optimised fuzzy logic engine on one applicant. Returns risk "
                     "score 0-100, decision (APPROVE/REVIEW/REJECT) and membership degrees. "
                     "Call when you have all 5 variables. May be called MULTIPLE times for what-if."),
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
4. CALL TOOL- call calculate_loan_risk. If the decision is REJECT or REVIEW, try a what-if:
   call again with a small realistic change to see if it would flip to a better decision.
5. EXPLAIN  - explain using the membership_degrees the tool returned (do NOT invent reasons).
6. ACT      - give the final decision and, if not APPROVE, 2-3 concrete improvement steps.

Reply in the user's language. Be concise."""

def get_client():
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    return genai.Client(api_key=key) if key else None

def run_agent(client, history, user_msg, render):
    contents = history + [types.Content(role="user", parts=[types.Part(text=user_msg)])]
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=[TOOLS],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    for _ in range(8):
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=contents, config=cfg)
        cand = resp.candidates[0]
        parts = cand.content.parts or []
        for p in parts:
            if getattr(p, "text", None):
                render("text", p.text.strip())
        fcs = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if fcs:
            contents.append(cand.content)
            rp = []
            for fc in fcs:
                args = dict(fc.args)
                render("tool", (fc.name, args))
                result = execute_tool(fc.name, args)
                render("result", (fc.name, result))
                rp.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=rp))
            continue
        else:
            contents.append(cand.content)
            break
    return contents


# ----------------------------------------------------------------
# Plotly (light theme)
# ----------------------------------------------------------------
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
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(font=dict(size=42, color=col), suffix="<span style='font-size:15px;color:#67767F'> /100</span>"),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(size=10)),
            bar=dict(color=col, thickness=0.3), bgcolor="#F4F6F8", borderwidth=0,
            steps=[dict(range=[0, 40], color="rgba(21,147,95,.12)"),
                   dict(range=[40, 70], color="rgba(201,138,30,.12)"),
                   dict(range=[70, 100], color="rgba(209,67,67,.12)")],
            threshold=dict(line=dict(color=col, width=3), thickness=0.8, value=score))))
    return _light(fig, h=215)

def convergence_figure():
    fit = fe.ga_result["fit_history"]; x = list(range(len(fit)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fit, mode="lines", name="Best fitness",
        line=dict(color=ACCENT, width=3), fill="tozeroy", fillcolor="rgba(19,122,84,.08)",
        hovertemplate="Gen %{x}<br>Fitness %{y:.1f}<extra></extra>"))
    fig.update_layout(title=dict(text="GA Convergence — lower is better",
                                 font=dict(color=INK, size=15)))
    fig.update_yaxes(range=[min(fit) * 0.97, max(fit) * 1.01])
    return _light(fig, h=350)

def mf_figure():
    specs = [
        (fe.inc_u, {"Low": fe.income_fuzzy['low'].mf, "Medium": fe.income_fuzzy['medium'].mf,
                    "High": fe.income_fuzzy['high'].mf},
         f"Annual Income · GA-tuned", [RED, AMBER, GREEN]),
        (fe.credit_fuzzy.universe,
         {"Very Poor": fe.credit_fuzzy['very_poor'].mf, "Poor": fe.credit_fuzzy['poor'].mf,
          "Fair": fe.credit_fuzzy['fair'].mf, "Good": fe.credit_fuzzy['good'].mf,
          "Excellent": fe.credit_fuzzy['excellent'].mf},
         "Credit Score · fixed (CTOS)", ["#8B2020", RED, AMBER, GREEN, "#0E5D40"]),
        (fe.rat_u, {"Low": fe.ratio_fuzzy['low'].mf, "Medium": fe.ratio_fuzzy['medium'].mf,
                    "High": fe.ratio_fuzzy['high'].mf},
         "Loan-to-Income Ratio · GA-tuned", [GREEN, AMBER, RED]),
        (fe.default_fuzzy.universe,
         {"No Default": fe.default_fuzzy['no'].mf, "Defaulted": fe.default_fuzzy['yes'].mf},
         "Previous Default · fixed (binary)", [GREEN, RED]),
        (fe.emp_u, {"Junior": fe.emp_fuzzy['junior'].mf, "Mid-Level": fe.emp_fuzzy['mid'].mf,
                    "Experienced": fe.emp_fuzzy['experienced'].mf},
         "Employment Experience · GA-tuned", [RED, AMBER, GREEN]),
        (fe.risk.universe, {"Low Risk": fe.risk['low'].mf, "Medium Risk": fe.risk['medium'].mf,
                            "High Risk": fe.risk['high'].mf},
         "Output · Risk Score", [GREEN, AMBER, RED]),
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


# ----------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-brand"><div class="sb-logo">L</div>'
                '<div><div class="t">Loan Approval</div><div class="s">Fuzzy · GA · Agent</div></div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["Assess", "AI Agent", "GA Results", "Membership Functions"],
                    label_visibility="collapsed")


def header(eyebrow, title, desc):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="h-title">{title}</div>'
                f'<div class="h-desc">{desc}</div>', unsafe_allow_html=True)


# ================================================================
# PAGE: ASSESS
# ================================================================
if page == "Assess":
    header("Risk Assessment", "Evaluate Applicant",
           "Enter the five inputs the GA-optimised fuzzy engine evaluates to produce a risk score and decision.")
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="card"><h3>Applicant Information</h3>'
                    '<div class="sub">Five inputs the fuzzy inference system reads.</div>',
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        income = a.number_input("Annual income (RM)", 0, 1_000_000, 65000, step=1000)
        credit = b.number_input("Credit score (CTOS)", 300, 850, 720)
        ratio  = a.number_input("Loan as % of income (0–1)", 0.0, 1.0, 0.18, step=0.01)
        emp    = b.number_input("Employment experience (yrs)", 0.0, 50.0, 4.0, step=0.5)
        default = st.selectbox("Previous loan default", ["No", "Yes"])
        evaluate_clicked = st.button("Evaluate Risk", use_container_width=True)
        st.markdown('<div class="sub" style="margin:14px 0 0;">CTOS bands &nbsp; 300–449 Very Poor · '
                    '450–549 Poor · 550–649 Fair · 650–749 Good · 750–850 Excellent</div></div>',
                    unsafe_allow_html=True)

    with col_out:
        if evaluate_clicked:
            r = compute_risk_detailed(income, credit, ratio, default, emp)
            cls = {"APPROVE": "approve", "REVIEW": "review", "REJECT": "reject"}[r["decision"]]
            st.markdown('<div class="card"><h3>Assessment Result</h3>'
                        '<div class="sub">Decision, risk gauge, and the reasons drawn from the model.</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="badge {cls}"><span class="lab">{r["decision"]}</span>'
                        f'<span class="sc">Risk {r["risk_score"]} / 100</span></div>',
                        unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(r["risk_score"], r["decision"]),
                            use_container_width=True, config=PCFG)
            st.markdown('<div style="font-weight:700;font-size:14px;margin:4px 0 6px;">Main reasons</div>',
                        unsafe_allow_html=True)
            cmap = {"pos": "rpos", "neg": "rneg", "neu": "rneu"}
            html = ""
            for kind, text in reasons_from_degrees(r["membership_degrees"]):
                html += f'<div class="reason"><span class="rdot {cmap[kind]}"></span>{text}</div>'
            st.markdown(html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><h3>Assessment Result</h3>'
                        '<div class="sub">Decision, risk gauge, and reasons.</div>'
                        '<div style="text-align:center;color:#9AA8B0;padding:40px 10px;font-size:14px;">'
                        'Waiting for applicant input.<br>Click '
                        '<b style="color:#137A54">Evaluate Risk</b> to generate a decision.'
                        '</div></div>', unsafe_allow_html=True)


# ================================================================
# PAGE: AI AGENT
# ================================================================
elif page == "AI Agent":
    header("Agentic Reasoning", "AI Agent",
           "A reasoning agent that calls the fuzzy engine as a tool. It observes the input, "
           "gathers missing evidence, runs the engine, tests what-if scenarios, then explains and decides.")

    st.markdown('<div class="card"><h3>How the agent works</h3>'
                '<div class="sub">Every request follows the same six-step loop.</div>'
                '<div class="pipe">'
                '<div class="pstep"><span class="pnum">1</span><span class="pl">Observe</span></div>'
                '<span class="psep">→</span>'
                '<div class="pstep"><span class="pnum">2</span><span class="pl">Collect</span></div>'
                '<span class="psep">→</span>'
                '<div class="pstep"><span class="pnum">3</span><span class="pl">Reason</span></div>'
                '<span class="psep">→</span>'
                '<div class="pstep"><span class="pnum">4</span><span class="pl">Call engine</span></div>'
                '<span class="psep">→</span>'
                '<div class="pstep"><span class="pnum">5</span><span class="pl">Explain</span></div>'
                '<span class="psep">→</span>'
                '<div class="pstep"><span class="pnum">6</span><span class="pl">Act</span></div>'
                '</div></div>', unsafe_allow_html=True)

    client = get_client()
    if client is None:
        st.warning("No GEMINI_API_KEY found. Add it under Manage app → Settings → Secrets "
                   "(deployed) or as an environment variable (local). Free key at aistudio.google.com.")

    if "gem_history" not in st.session_state: st.session_state.gem_history = []
    if "chat_log" not in st.session_state: st.session_state.chat_log = []
    if "queued" not in st.session_state: st.session_state.queued = None

    st.markdown('<div class="card"><h3>Try an example</h3>'
                '<div class="sub">Click to send, or type your own below.</div>', unsafe_allow_html=True)
    ex = ["income 80k, credit 600, ratio 0.45, no default, 1 year",
          "RM120k salary, CTOS 760, borrowing 15%, no defaults, 8 years",
          "credit 540, defaulted before, ratio 0.5, income 40k, 2 years"]
    ec = st.columns(len(ex))
    for i, e in enumerate(ex):
        if ec[i].button(e, key=f"ex{i}", use_container_width=True):
            st.session_state.queued = e
    st.markdown("</div>", unsafe_allow_html=True)

    for role, text in st.session_state.chat_log:
        with st.chat_message(role):
            st.markdown(text, unsafe_allow_html=True)

    typed = st.chat_input("Describe an applicant…", disabled=(client is None))
    prompt = typed or st.session_state.queued
    st.session_state.queued = None

    if prompt and client is not None:
        st.session_state.chat_log.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            box = st.container(); buf = {"text": []}
            def render(kind, payload):
                if kind == "text":
                    buf["text"].append(payload); box.markdown(payload)
                elif kind == "tool":
                    name, args = payload
                    box.markdown(f'<div class="trace"><span class="tn">call</span> {name} '
                                 f'<code>{json.dumps(args, ensure_ascii=False)}</code></div>',
                                 unsafe_allow_html=True)
                elif kind == "result":
                    name, res = payload
                    if name == "calculate_loan_risk":
                        d = res["membership_degrees"]
                        box.markdown(f'<div class="trace"><span class="tn">result</span> score '
                                     f'<b>{res["risk_score"]}</b> → <b>{res["decision"]}</b> · '
                                     f'ratio_high {d["ratio_high"]} · emp_junior {d["emp_junior"]} · '
                                     f'credit_risk {d["credit_risk"]}</div>', unsafe_allow_html=True)
                    else:
                        box.markdown(f'<div class="trace"><span class="tn">result</span> '
                                     f'<code>{json.dumps(res, ensure_ascii=False)}</code></div>',
                                     unsafe_allow_html=True)
            with st.spinner("Agent reasoning…"):
                st.session_state.gem_history = run_agent(
                    client, st.session_state.gem_history, prompt, render)
            st.session_state.chat_log.append(("assistant", "\n\n".join(buf["text"]) or "_done_"))


# ================================================================
# PAGE: GA RESULTS
# ================================================================
elif page == "GA Results":
    header("Data-Driven Optimisation", "Genetic Algorithm",
           "The GA tunes the fuzzy membership-function breakpoints by minimising the error "
           "between the model's risk output and the real loan-status labels.")

    st.markdown(
        f'<div class="mwrap">'
        f'<div class="mcard"><div class="k">Initial Fitness</div><div class="v">{ga["init_fitness"]:.0f}</div>'
        f'<div class="d">generation 0</div></div>'
        f'<div class="mcard"><div class="k">Final Fitness</div><div class="v"><span class="ac">'
        f'{ga["best_fitness"]:.0f}</span></div><div class="d">best individual</div></div>'
        f'<div class="mcard"><div class="k">Improvement</div><div class="v"><span class="ac">'
        f'{ga["improvement_pct"]:.1f}%</span></div><div class="d">lower = better fit</div></div>'
        f'<div class="mcard"><div class="k">Reference Data</div><div class="v">{len(fe.REF_STATUS):,}</div>'
        f'<div class="d">{fe.n_approved:,} approved · {fe.n_rejected:,} rejected</div></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="callout">Fitness dropped <b>{ga["improvement_pct"]:.1f}%</b> '
        f'({ga["init_fitness"]:.0f} → {ga["best_fitness"]:.0f}) over {fe.GA_GENS} generations — '
        f'the GA-tuned membership functions match the reference loan data far better than the '
        f'initial hand-set functions. Lower fitness = closer to ground truth.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Convergence</h3>'
                f'<div class="sub">Population {fe.GA_POP} · {fe.GA_GENS} generations · '
                f'two-point crossover · adaptive mutation · tournament k={fe.GA_TOURNAMENT} · '
                f'fitness = Σ(loan_status − fuzzy_risk)²</div>', unsafe_allow_html=True)
    st.plotly_chart(convergence_figure(), use_container_width=True, config=PCFG)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Optimised MF Breakpoints</h3>'
                '<div class="sub">Evolved parameters for the three GA-tuned variables. '
                'Credit and default are fixed, not evolved.</div>', unsafe_allow_html=True)
    st.dataframe({
        "Variable": ["Annual Income (RM)", "Loan Ratio (0–1)", "Employment Exp (yrs)"],
        "a · Low→Mid": [f"{fe.inc_a:,.0f}", f"{fe.rat_a:.4f}", f"{fe.emp_a:.2f}"],
        "b · Mid peak": [f"{fe.inc_b:,.0f}", f"{fe.rat_b:.4f}", f"{fe.emp_b:.2f}"],
        "c · Mid→High": [f"{fe.inc_c:,.0f}", f"{fe.rat_c:.4f}", f"{fe.emp_c:.2f}"],
    }, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# PAGE: MEMBERSHIP FUNCTIONS
# ================================================================
elif page == "Membership Functions":
    header("Fuzzy Sets", "Membership Functions",
           "The fuzzy sets the inference engine uses. Three variables are tuned by the GA; "
           "credit and default are fixed by standard. Hover any curve to read its membership value.")

    c1, c2 = st.columns(2, gap="large")
    c1.markdown('<div class="card" style="margin-bottom:0;"><h3 style="color:#137A54;">GA-tuned variables</h3>'
                '<div class="sub" style="margin:0;">Data-driven · breakpoints evolved by the genetic algorithm</div>'
                '<div style="margin-top:10px;font-size:14px;line-height:1.9;">'
                '• Annual Income<br>• Loan-to-Income Ratio<br>• Employment Experience</div></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="card" style="margin-bottom:0;"><h3>Fixed variables</h3>'
                '<div class="sub" style="margin:0;">Knowledge-driven · set by standard, not evolved</div>'
                '<div style="margin-top:10px;font-size:14px;line-height:1.9;">'
                '• CTOS Credit Score<br>• Previous Loan Default</div></div>',
                unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card"><h3>All membership functions</h3>'
                '<div class="sub">Green-coloured high/low sets indicate favourable regions.</div>',
                unsafe_allow_html=True)
    st.plotly_chart(mf_figure(), use_container_width=True, config=PCFG)
    st.markdown("</div>", unsafe_allow_html=True)
