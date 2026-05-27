# ================================================================
# Loan Approval Intelligence — Streamlit + Gemini AI Agent
# Fuzzy engine (FIS + GA) lives in fuzzy_engine.py — unchanged.
# UI: premium dark theme · interactive Plotly charts · no emojis.
# Tabs: Assess · AI Agent · GA Results · Membership Functions
# ================================================================

import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from google.genai import types

st.set_page_config(page_title="Loan Approval Intelligence",
                   page_icon="●", layout="wide")

# ---- design tokens ----
BG      = "#0A0E17"
SURFACE = "#111726"
SURF2   = "#0E1422"
BORDER  = "#1E2740"
TEXT    = "#E6EBF4"
MUTED   = "#8A97AD"
ACCENT  = "#5B9DFF"
GREEN   = "#34D399"
AMBER   = "#FBBF24"
RED     = "#F87171"
FONT    = "Inter, -apple-system, 'Segoe UI', system-ui, sans-serif"

# ----------------------------------------------------------------
# Global CSS — premium dark, strong contrast, no default Streamlit chrome
# ----------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {{ background:{BG}; }}
.main .block-container {{ padding-top:1.1rem; padding-bottom:3rem; max-width:1240px; }}
html, body, [class*="css"] {{ font-family:{FONT}; color:{TEXT}; }}

#MainMenu, header, footer {{ visibility:hidden; }}

/* hero */
.hero {{ position:relative; overflow:hidden; border:1px solid {BORDER};
        background:
          radial-gradient(120% 140% at 0% 0%, rgba(91,157,255,.16) 0%, transparent 55%),
          radial-gradient(120% 140% at 100% 0%, rgba(52,211,153,.12) 0%, transparent 50%),
          {SURFACE};
        padding:26px 30px; border-radius:18px; }}
.hero h1 {{ font-size:27px; font-weight:800; letter-spacing:-.6px; margin:0; color:#fff; }}
.hero p  {{ font-size:13px; color:{MUTED}; margin:7px 0 0; max-width:760px; line-height:1.5; }}
.hero .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
             background:{GREEN}; margin-right:9px; box-shadow:0 0 0 4px rgba(52,211,153,.18); }}

/* stat strip */
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 4px; }}
.stat {{ flex:1; min-width:150px; background:{SURFACE}; border:1px solid {BORDER};
        border-radius:13px; padding:13px 16px; }}
.stat .k {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.7px; font-weight:600; }}
.stat .v {{ font-size:20px; font-weight:700; color:{TEXT}; margin-top:3px; }}
.stat .v .ac {{ color:{ACCENT}; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {BORDER}; padding-bottom:0; }}
.stTabs [data-baseweb="tab"] {{ height:42px; background:transparent; border:none;
        color:{MUTED}; font-weight:600; font-size:14px; padding:0 6px; }}
.stTabs [aria-selected="true"] {{ color:{TEXT}; border-bottom:2px solid {ACCENT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:transparent; }}

/* cards */
.card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px;
        padding:22px 24px; margin-bottom:4px; }}
.card h3 {{ margin:0 0 4px; font-size:16px; font-weight:700; color:{TEXT}; }}
.card .sub {{ font-size:12.5px; color:{MUTED}; margin-bottom:16px; }}

/* form labels — force readable contrast */
[data-testid="stWidgetLabel"] p, .stNumberInput label, .stSelectbox label,
.stTextInput label, .stTextArea label {{ color:#C5D0E2 !important; font-weight:600 !important;
        font-size:12.5px !important; }}
.stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background:{SURF2} !important; color:{TEXT} !important; border:1px solid {BORDER} !important;
        border-radius:10px !important; }}
.stNumberInput button {{ background:{SURF2} !important; border:1px solid {BORDER} !important; }}

/* primary button */
.stButton > button {{ background:linear-gradient(135deg,{ACCENT},#7C5BFF); color:#fff;
        border:none; border-radius:11px; font-weight:700; padding:11px 0; font-size:14px;
        transition:transform .08s ease, box-shadow .2s ease; }}
.stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 8px 22px rgba(91,157,255,.32); }}

/* decision badge */
.badge {{ border-radius:14px; padding:20px; text-align:center; border:1px solid; }}
.badge .lab {{ font-size:24px; font-weight:800; letter-spacing:.4px; }}
.badge .sc {{ font-size:13px; opacity:.85; margin-top:3px; font-weight:600; }}
.approve {{ background:rgba(52,211,153,.10); border-color:rgba(52,211,153,.35); color:{GREEN}; }}
.review  {{ background:rgba(251,191,36,.10); border-color:rgba(251,191,36,.35); color:{AMBER}; }}
.reject  {{ background:rgba(248,113,113,.10); border-color:rgba(248,113,113,.35); color:{RED}; }}

/* degree rows */
.deg {{ display:flex; justify-content:space-between; align-items:center;
       padding:9px 0; border-bottom:1px solid {BORDER}; font-size:13px; }}
.deg:last-child {{ border-bottom:none; }}
.deg .name {{ color:{MUTED}; }}
.chip {{ background:{SURF2}; border:1px solid {BORDER}; border-radius:7px;
        padding:2px 9px; font-family:'SF Mono',Menlo,monospace; font-size:12px; color:{TEXT};
        margin-left:6px; }}

/* pipeline strip */
.pipe {{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 18px; }}
.step {{ background:{SURF2}; border:1px solid {BORDER}; border-radius:9px;
        padding:7px 13px; font-size:12px; color:{MUTED}; font-weight:600; }}
.step b {{ color:{ACCENT}; }}
.arrow {{ color:{BORDER}; align-self:center; font-size:13px; }}

/* example chips */
.exwrap {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:6px; }}

/* chat tool trace */
.trace {{ background:{SURF2}; border:1px solid {BORDER}; border-left:3px solid {ACCENT};
         border-radius:9px; padding:10px 13px; margin:7px 0; font-family:'SF Mono',Menlo,monospace;
         font-size:12px; color:#B7C3D8; }}
.trace .tn {{ color:{ACCENT}; font-weight:700; }}
hr {{ border-color:{BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------
# Load fuzzy engine ONCE (GA runs first load, then cached)
# ----------------------------------------------------------------
@st.cache_resource(show_spinner="Running GA optimisation (first load only, ~15s)…")
def load_engine():
    import fuzzy_engine as fe
    return fe

fe = load_engine()


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
    return {"risk_score": round(score, 1), "decision": decision,
            "membership_degrees": deg,
            "rule_weights": {"ratio": fe.W_RATIO, "credit": fe.W_CREDIT,
                             "income": fe.W_INCOME, "default": fe.W_DEFAULT, "emp": fe.W_EMP}}


def lookup_ctos_category(credit):
    c = float(credit)
    cat = ("Very Poor" if c <= 449 else "Poor" if c <= 549 else
           "Fair" if c <= 649 else "Good" if c <= 749 else "Excellent")
    return {"credit_score": c, "category": cat, "risk_value": round(fe.ctos_risk(c), 2)}


# ----------------------------------------------------------------
# Gemini tool declarations + agent loop
# ----------------------------------------------------------------
TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="calculate_loan_risk",
        description=("Run the GA-optimised fuzzy logic engine on one applicant. Returns risk "
                     "score 0-100, decision (APPROVE/REVIEW/REJECT) and membership degrees. "
                     "Call when you have all 5 variables. May be called MULTIPLE times for "
                     "what-if scenarios."),
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "income": types.Schema(type=types.Type.NUMBER, description="Annual income RM"),
            "credit": types.Schema(type=types.Type.NUMBER, description="CTOS score 300-850"),
            "ratio":  types.Schema(type=types.Type.NUMBER, description="Loan as fraction of income 0-1"),
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
   call again with a small realistic change (lower the loan ratio, +1 year experience) to
   see if it would flip to a better decision.
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
        resp = client.models.generate_content(
            model="gemini-2.0-flash", contents=contents, config=cfg)
        cand = resp.candidates[0]
        parts = cand.content.parts or []
        for p in parts:
            if getattr(p, "text", None):
                render("text", p.text.strip())
        fcs = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if fcs:
            contents.append(cand.content)
            resp_parts = []
            for fc in fcs:
                args = dict(fc.args)
                render("tool", (fc.name, args))
                result = execute_tool(fc.name, args)
                render("result", (fc.name, result))
                resp_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=resp_parts))
            continue
        else:
            contents.append(cand.content)
            break
    return contents


# ----------------------------------------------------------------
# Plotly theming + figures (interactive, not static images)
# ----------------------------------------------------------------
def _layout(fig, h=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=12),
        margin=dict(l=10, r=10, t=46, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_size=12),
    )
    if h: fig.update_layout(height=h)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig

def risk_gauge(score, decision):
    col = {"APPROVE": GREEN, "REVIEW": AMBER, "REJECT": RED}[decision]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(font=dict(size=40, color=col), suffix="<span style='font-size:15px;color:#8A97AD'> /100</span>"),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(size=10)),
            bar=dict(color=col, thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 40], color="rgba(52,211,153,.14)"),
                   dict(range=[40, 70], color="rgba(251,191,36,.14)"),
                   dict(range=[70, 100], color="rgba(248,113,113,.14)")],
            threshold=dict(line=dict(color=col, width=3), thickness=0.78, value=score)),
    ))
    return _layout(fig, h=230)

def convergence_figure():
    fit = fe.ga_result["fit_history"]
    x = list(range(len(fit)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fit, mode="lines", name="Best fitness",
        line=dict(color=ACCENT, width=2.6),
        fill="tozeroy", fillcolor="rgba(91,157,255,.10)",
        hovertemplate="Gen %{x}<br>Fitness %{y:.1f}<extra></extra>"))
    fig.update_layout(title=dict(text="GA Convergence  ·  lower is better",
                                 font=dict(color=TEXT, size=15)))
    fig.update_yaxes(range=[min(fit) * 0.97, max(fit) * 1.01])
    return _layout(fig, h=360)

def mf_figure():
    specs = [
        (fe.inc_u, {"Low": fe.income_fuzzy['low'].mf, "Medium": fe.income_fuzzy['medium'].mf,
                    "High": fe.income_fuzzy['high'].mf},
         f"Annual Income · GA  (a={fe.inc_a:,.0f}  b={fe.inc_b:,.0f}  c={fe.inc_c:,.0f})",
         [RED, AMBER, GREEN]),
        (fe.credit_fuzzy.universe,
         {"Very Poor": fe.credit_fuzzy['very_poor'].mf, "Poor": fe.credit_fuzzy['poor'].mf,
          "Fair": fe.credit_fuzzy['fair'].mf, "Good": fe.credit_fuzzy['good'].mf,
          "Excellent": fe.credit_fuzzy['excellent'].mf},
         "Credit Score · CTOS (fixed)", ["#8B0000", RED, AMBER, GREEN, "#1B7A3D"]),
        (fe.rat_u, {"Low": fe.ratio_fuzzy['low'].mf, "Medium": fe.ratio_fuzzy['medium'].mf,
                    "High": fe.ratio_fuzzy['high'].mf},
         f"Loan-to-Income Ratio · GA  (a={fe.rat_a:.3f}  b={fe.rat_b:.3f}  c={fe.rat_c:.3f})",
         [GREEN, AMBER, RED]),
        (fe.default_fuzzy.universe,
         {"No Default": fe.default_fuzzy['no'].mf, "Defaulted": fe.default_fuzzy['yes'].mf},
         "Previous Default · binary (fixed)", [GREEN, RED]),
        (fe.emp_u, {"Junior": fe.emp_fuzzy['junior'].mf, "Mid-Level": fe.emp_fuzzy['mid'].mf,
                    "Experienced": fe.emp_fuzzy['experienced'].mf},
         f"Employment Experience · GA  (a={fe.emp_a:.2f}  b={fe.emp_b:.2f}  c={fe.emp_c:.2f})",
         [RED, AMBER, GREEN]),
        (fe.risk.universe, {"Low Risk": fe.risk['low'].mf, "Medium Risk": fe.risk['medium'].mf,
                            "High Risk": fe.risk['high'].mf},
         "Output · Risk Score", [GREEN, AMBER, RED]),
    ]
    titles = [s[2] for s in specs]
    fig = make_subplots(rows=3, cols=2, subplot_titles=titles,
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    for i, (u, mfs, _t, colors) in enumerate(specs):
        r, c = i // 2 + 1, i % 2 + 1
        showleg = True
        for (lbl, vals), col in zip(mfs.items(), colors):
            fig.add_trace(go.Scatter(
                x=list(u), y=list(vals), mode="lines", name=lbl,
                line=dict(color=col, width=2.2), legendgroup=f"g{i}",
                showlegend=showleg, fill="tozeroy",
                fillcolor=col.replace(")", ",.08)").replace("#", "rgba(") if False else None,
                hovertemplate=f"{lbl}<br>x %{{x:.2f}}<br>μ %{{y:.2f}}<extra></extra>"),
                row=r, col=c)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=820,
                      font=dict(family=FONT, color=MUTED, size=11),
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, range=[-0.05, 1.1])
    for ann in fig.layout.annotations:
        ann.font = dict(color=ACCENT, size=12.5, family=FONT)
    return fig

PLOTLY_CFG = {"displayModeBar": False, "responsive": True}


# ----------------------------------------------------------------
# HERO + STAT STRIP
# ----------------------------------------------------------------
ga = fe.ga_result
st.markdown(
    '<div class="hero"><h1>Loan Approval Intelligence</h1>'
    '<p><span class="dot"></span>GA-optimised Fuzzy Logic risk engine (Arslan &amp; Kaya, 2001) '
    'paired with a Gemini AI agent that calls the engine as a tool. '
    'CTOS credit standard · five-variable model · trained on 45,000 records.</p></div>',
    unsafe_allow_html=True)

st.markdown(
    f'<div class="stats">'
    f'<div class="stat"><div class="k">GA Fitness</div><div class="v">{ga["init_fitness"]:.0f} '
    f'<span class="ac">&rarr;</span> {ga["best_fitness"]:.0f}</div></div>'
    f'<div class="stat"><div class="k">Improvement</div><div class="v"><span class="ac">'
    f'{ga["improvement_pct"]:.1f}%</span></div></div>'
    f'<div class="stat"><div class="k">Generations</div><div class="v">{fe.GA_GENS}</div></div>'
    f'<div class="stat"><div class="k">Reference Rows</div><div class="v">{len(fe.REF_STATUS):,}</div></div>'
    f'</div>', unsafe_allow_html=True)

tab_assess, tab_agent, tab_ga, tab_mf = st.tabs(
    ["Assess", "AI Agent", "GA Results", "Membership Functions"])


# ---------------- TAB 1: ASSESS ----------------
with tab_assess:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="card"><h3>Applicant Information</h3>'
                    '<div class="sub">Enter the five inputs the fuzzy engine evaluates.</div>',
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        income = a.number_input("Annual income (RM)", 0, 1_000_000, 65000, step=1000)
        credit = b.number_input("Credit score (CTOS)", 300, 850, 720)
        ratio  = a.number_input("Loan as % of income (0–1)", 0.0, 1.0, 0.18, step=0.01)
        emp    = b.number_input("Employment experience (yrs)", 0.0, 50.0, 4.0, step=0.5)
        default = st.selectbox("Previous loan default", ["No", "Yes"])
        go = st.button("Evaluate Risk", use_container_width=True, type="primary")
        st.markdown('<div class="sub" style="margin-top:14px;">CTOS bands &nbsp; '
                    '300–449 Very Poor · 450–549 Poor · 550–649 Fair · 650–749 Good · '
                    '750–850 Excellent</div></div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="card"><h3>Assessment Result</h3>'
                    '<div class="sub">Decision, risk gauge, and the membership degrees the model used.</div>',
                    unsafe_allow_html=True)
        if go:
            r = compute_risk_detailed(income, credit, ratio, default, emp)
            cls = {"APPROVE": "approve", "REVIEW": "review", "REJECT": "reject"}[r["decision"]]
            st.markdown(f'<div class="badge {cls}"><div class="lab">{r["decision"]}</div>'
                        f'<div class="sc">Risk Score {r["risk_score"]} / 100</div></div>',
                        unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(r["risk_score"], r["decision"]),
                            use_container_width=True, config=PLOTLY_CFG)
            d = r["membership_degrees"]
            rows = [
                ("Income", f'high {d["income_high"]}', f'low {d["income_low"]}'),
                ("Loan ratio", f'high {d["ratio_high"]}', f'low {d["ratio_low"]}'),
                ("Experience", f'experienced {d["emp_experienced"]}', f'junior {d["emp_junior"]}'),
                ("Credit / default", f'credit-risk {d["credit_risk"]}', f'defaulted {d["defaulted"]}'),
            ]
            html = ""
            for name, c1, c2 in rows:
                html += (f'<div class="deg"><span class="name">{name}</span>'
                         f'<span><span class="chip">{c1}</span><span class="chip">{c2}</span></span></div>')
            st.markdown(html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#8A97AD;font-size:13px;padding:18px 0;">'
                        'Enter applicant details and click <b style="color:#C5D0E2">Evaluate Risk</b>.'
                        '</div></div>', unsafe_allow_html=True)


# ---------------- TAB 2: AI AGENT ----------------
with tab_agent:
    st.markdown('<div class="card"><h3>AI Agent</h3>'
                '<div class="sub">A reasoning agent that calls the fuzzy engine as a tool — '
                'it observes the input, gathers missing evidence, runs the engine, tests '
                'what-if scenarios, then explains and decides.</div>'
                '<div class="pipe">'
                '<span class="step"><b>1</b> Observe</span><span class="arrow">→</span>'
                '<span class="step"><b>2</b> Collect</span><span class="arrow">→</span>'
                '<span class="step"><b>3</b> Reason</span><span class="arrow">→</span>'
                '<span class="step"><b>4</b> Call engine</span><span class="arrow">→</span>'
                '<span class="step"><b>5</b> Explain</span><span class="arrow">→</span>'
                '<span class="step"><b>6</b> Act</span>'
                '</div></div>', unsafe_allow_html=True)

    client = get_client()
    if client is None:
        st.warning("No GEMINI_API_KEY found. Add it under Manage app → Settings → Secrets "
                   "(deployed) or as an environment variable (local). "
                   "Get a free key at aistudio.google.com.")

    if "gem_history" not in st.session_state:
        st.session_state.gem_history = []
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "queued" not in st.session_state:
        st.session_state.queued = None

    # example prompts
    st.markdown('<div class="sub" style="margin:2px 0 6px;">Quick examples</div>', unsafe_allow_html=True)
    ex = ["income 80k, credit 600, ratio 0.45, no default, 1 year",
          "RM120k salary, CTOS 760, borrowing 15%, no defaults, 8 years",
          "credit 540, defaulted before, ratio 0.5, income 40k, 2 years"]
    ec = st.columns(len(ex))
    for i, e in enumerate(ex):
        if ec[i].button(e, key=f"ex{i}", use_container_width=True):
            st.session_state.queued = e

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
            box = st.container()
            buf = {"text": []}
            def render(kind, payload):
                if kind == "text":
                    buf["text"].append(payload); box.markdown(payload)
                elif kind == "tool":
                    name, args = payload
                    box.markdown(f'<div class="trace"><span class="tn">call</span> {name} '
                                 f'· {json.dumps(args, ensure_ascii=False)}</div>',
                                 unsafe_allow_html=True)
                elif kind == "result":
                    name, res = payload
                    if name == "calculate_loan_risk":
                        d = res["membership_degrees"]
                        box.markdown(
                            f'<div class="trace"><span class="tn">result</span> score '
                            f'<b>{res["risk_score"]}</b> → <b>{res["decision"]}</b> · '
                            f'ratio_high {d["ratio_high"]} · emp_junior {d["emp_junior"]} · '
                            f'credit_risk {d["credit_risk"]}</div>', unsafe_allow_html=True)
                    else:
                        box.markdown(f'<div class="trace"><span class="tn">result</span> '
                                     f'{json.dumps(res, ensure_ascii=False)}</div>',
                                     unsafe_allow_html=True)
            with st.spinner("Agent reasoning…"):
                st.session_state.gem_history = run_agent(
                    client, st.session_state.gem_history, prompt, render)
            st.session_state.chat_log.append(("assistant", "\n\n".join(buf["text"]) or "_done_"))


# ---------------- TAB 3: GA RESULTS ----------------
with tab_ga:
    st.markdown('<div class="card"><h3>Genetic Algorithm — Membership-Function Optimisation</h3>'
                f'<div class="sub">Population {fe.GA_POP} · {fe.GA_GENS} generations · '
                f'two-point crossover {fe.GA_CR*100:.0f}% · adaptive mutation · '
                f'tournament k={fe.GA_TOURNAMENT} · fitness = Σ(loan_status − fuzzy_risk)²</div>',
                unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Initial fitness", f"{ga['init_fitness']:.1f}")
    g2.metric("Final fitness", f"{ga['best_fitness']:.1f}", f"−{ga['improvement_pct']:.1f}%",
              delta_color="inverse")
    g3.metric("Approved (ref)", f"{fe.n_approved:,}")
    g4.metric("Rejected (ref)", f"{fe.n_rejected:,}")

    st.plotly_chart(convergence_figure(), use_container_width=True, config=PLOTLY_CFG)
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


# ---------------- TAB 4: MEMBERSHIP FUNCTIONS ----------------
with tab_mf:
    st.markdown('<div class="card"><h3>Membership Functions</h3>'
                '<div class="sub">The fuzzy sets the inference engine uses. GA-labelled charts '
                'were tuned by the genetic algorithm; credit and default are fixed. '
                'Hover any curve to read its membership value.</div>', unsafe_allow_html=True)
    st.plotly_chart(mf_figure(), use_container_width=True, config=PLOTLY_CFG)
    st.markdown("</div>", unsafe_allow_html=True)
