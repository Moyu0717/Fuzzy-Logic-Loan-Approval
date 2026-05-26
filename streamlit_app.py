# ================================================================
# Loan Approval Intelligence — Streamlit + Gemini AI Agent
# ----------------------------------------------------------------
# Fuzzy engine (FIS + GA) lives in fuzzy_engine.py — unchanged.
# This file: polished UI + a Gemini tool-calling agent whose loop is
#   observe -> collect -> reason -> call tools -> explain -> act
# Tabs: Assess · AI Agent · GA Results · Membership Functions
# ================================================================

import os
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from google import genai
from google.genai import types

st.set_page_config(page_title="Loan Approval Intelligence",
                   page_icon="🏦", layout="wide")

# ----------------------------------------------------------------
# Styling
# ----------------------------------------------------------------
st.markdown("""
<style>
  .stApp { background: #0e1117; }
  .main .block-container { padding-top: 1.4rem; max-width: 1280px; }
  .hero { background: linear-gradient(135deg,#0d3b66 0%,#1b5e20 100%);
          padding: 22px 28px; border-radius: 16px; margin-bottom: 6px; }
  .hero h1 { color:#fff; font-size:30px; font-weight:800; margin:0; letter-spacing:-.5px; }
  .hero p  { color:#cfe3ff; font-size:13px; margin:6px 0 0; }
  .pill { display:inline-block; background:#16203a; border:1px solid #243352;
          color:#cfe3ff; padding:6px 12px; border-radius:999px;
          font-size:12px; margin-right:8px; }
  .pill b { color:#7fd1ff; }
  .badge { font-size:26px; font-weight:800; padding:14px 18px; border-radius:14px;
           text-align:center; margin:4px 0; }
  .approve { background:#0d2e16; color:#3ddc84; border:1px solid #1c5c33; }
  .review  { background:#33260a; color:#ffc14d; border:1px solid #6b5012; }
  .reject  { background:#350f12; color:#ff6b6b; border:1px solid #6b1d22; }
  h2, h3 { color:#e8eefc !important; }
  .stTabs [data-baseweb="tab-list"] { gap:4px; }
  .stTabs [data-baseweb="tab"] { background:#141a28; border-radius:10px 10px 0 0;
          padding:8px 18px; color:#9fb0c8; }
  .stTabs [aria-selected="true"] { background:#1d2740; color:#7fd1ff; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------
# Load fuzzy engine ONCE (GA runs on first load, then cached)
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
# Plot helpers (matplotlib, dark theme)
# ----------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#141a28", "axes.facecolor": "#141a28",
    "axes.edgecolor": "#39466b", "axes.labelcolor": "#cfe3ff",
    "text.color": "#e8eefc", "xtick.color": "#9fb0c8", "ytick.color": "#9fb0c8",
    "grid.color": "#222d44", "legend.edgecolor": "#39466b",
})

def mf_figure():
    inc_u = fe.inc_u; rat_u = fe.rat_u; emp_u = fe.emp_u
    plots = [
        (inc_u, {"Low": fe.income_fuzzy['low'].mf, "Medium": fe.income_fuzzy['medium'].mf,
                 "High": fe.income_fuzzy['high'].mf},
         f"Annual Income [GA]  a={fe.inc_a:,.0f} b={fe.inc_b:,.0f} c={fe.inc_c:,.0f}",
         "Income (RM)", ["#ff6b6b", "#ffc14d", "#3ddc84"]),
        (fe.credit_fuzzy.universe,
         {"Very Poor": fe.credit_fuzzy['very_poor'].mf, "Poor": fe.credit_fuzzy['poor'].mf,
          "Fair": fe.credit_fuzzy['fair'].mf, "Good": fe.credit_fuzzy['good'].mf,
          "Excellent": fe.credit_fuzzy['excellent'].mf},
         "Credit Score [CTOS - Fixed]", "Credit Score",
         ["#8b0000", "#ff6b6b", "#ffc14d", "#3ddc84", "#1b7a3d"]),
        (rat_u, {"Low": fe.ratio_fuzzy['low'].mf, "Medium": fe.ratio_fuzzy['medium'].mf,
                 "High": fe.ratio_fuzzy['high'].mf},
         f"Loan-to-Income Ratio [GA]  a={fe.rat_a:.3f} b={fe.rat_b:.3f} c={fe.rat_c:.3f}",
         "Ratio (0-1)", ["#3ddc84", "#ffc14d", "#ff6b6b"]),
        (fe.default_fuzzy.universe,
         {"No Default": fe.default_fuzzy['no'].mf, "Defaulted": fe.default_fuzzy['yes'].mf},
         "Previous Default [Binary - Fixed]", "0 = No   1 = Yes", ["#3ddc84", "#ff6b6b"]),
        (emp_u, {"Junior": fe.emp_fuzzy['junior'].mf, "Mid-Level": fe.emp_fuzzy['mid'].mf,
                 "Experienced": fe.emp_fuzzy['experienced'].mf},
         f"Employment Experience [GA]  a={fe.emp_a:.2f} b={fe.emp_b:.2f} c={fe.emp_c:.2f}",
         "Years", ["#ff6b6b", "#ffc14d", "#3ddc84"]),
        (fe.risk.universe, {"Low Risk": fe.risk['low'].mf, "Medium Risk": fe.risk['medium'].mf,
                            "High Risk": fe.risk['high'].mf},
         "OUTPUT: Risk Score", "Risk Score (0-100)", ["#3ddc84", "#ffc14d", "#ff6b6b"]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    axes = axes.ravel()
    for ax, (u, mfs, title, xlab, colors) in zip(axes, plots):
        for (lbl, vals), c in zip(mfs.items(), colors):
            ax.plot(u, vals, color=c, lw=2, label=lbl)
            ax.fill_between(u, vals, alpha=0.12, color=c)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#7fd1ff")
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel("mu", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7, facecolor="#141a28", labelcolor="#cfe3ff")
        ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=2.0)
    return fig

def convergence_figure():
    fit = fe.ga_result["fit_history"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(range(len(fit)), fit, color="#3ddc84", lw=2, label="Best fitness")
    ax.set_title("GA Convergence (lower = better)", fontsize=11, fontweight="bold", color="#7fd1ff")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness  S(actual - fuzzy)^2")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, facecolor="#141a28", labelcolor="#cfe3ff")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------
# HERO
# ----------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🏦 Loan Approval Intelligence</h1>'
    '<p>GA-optimised Fuzzy Logic engine (Arslan &amp; Kaya, 2001) + a Gemini AI Agent '
    'that calls the engine as a <i>tool</i> · CTOS standard · 5-variable model</p></div>',
    unsafe_allow_html=True)

ga = fe.ga_result
st.markdown(
    f'<div style="margin:10px 0 14px;">'
    f'<span class="pill">GA Fitness <b>{ga["init_fitness"]:.1f} &rarr; {ga["best_fitness"]:.1f}</b></span>'
    f'<span class="pill">Improvement <b>{ga["improvement_pct"]:.1f}%</b></span>'
    f'<span class="pill">Generations <b>{fe.GA_GENS}</b></span>'
    f'<span class="pill">Dataset <b>{len(fe.REF_STATUS):,} rows</b></span>'
    f'</div>', unsafe_allow_html=True)


tab_assess, tab_agent, tab_ga, tab_mf = st.tabs(
    ["  📊  Assess  ", "  🤖  AI Agent  ", "  🧬  GA Results  ", "  📈  Membership Functions  "])


# ---------------- TAB 1: ASSESS ----------------
with tab_assess:
    c_in, c_out = st.columns([1, 1])
    with c_in:
        st.subheader("Applicant Information")
        a, b = st.columns(2)
        income = a.number_input("Annual income (RM)", 0, 1_000_000, 65000, step=1000)
        credit = b.number_input("Credit score (CTOS)", 300, 850, 720)
        ratio  = a.number_input("Loan % of income (0-1)", 0.0, 1.0, 0.18, step=0.01)
        emp    = b.number_input("Employment experience (yrs)", 0.0, 50.0, 4.0, step=0.5)
        default = st.selectbox("Previous loan default", ["No", "Yes"])
        go = st.button("Evaluate Risk", use_container_width=True, type="primary")
        st.caption("CTOS: 300-449 Very Poor · 450-549 Poor · 550-649 Fair · "
                   "650-749 Good · 750-850 Excellent")

    with c_out:
        st.subheader("Assessment Result")
        if go:
            r = compute_risk_detailed(income, credit, ratio, default, emp)
            cls = {"APPROVE": "approve", "REVIEW": "review", "REJECT": "reject"}[r["decision"]]
            icon = {"APPROVE": "✅", "REVIEW": "⚠️", "REJECT": "❌"}[r["decision"]]
            st.markdown(f'<div class="badge {cls}">{icon} {r["decision"]}<br>'
                        f'<span style="font-size:16px;">Risk Score {r["risk_score"]} / 100</span></div>',
                        unsafe_allow_html=True)
            st.markdown("**Membership degrees** (what the model actually saw)")
            d = r["membership_degrees"]
            st.markdown(
                f"- Income &rarr; high `{d['income_high']}` · low `{d['income_low']}`\n"
                f"- Ratio &rarr; high `{d['ratio_high']}` · low `{d['ratio_low']}`\n"
                f"- Experience &rarr; experienced `{d['emp_experienced']}` · junior `{d['emp_junior']}`\n"
                f"- Credit risk `{d['credit_risk']}` · Defaulted `{d['defaulted']}`")
            st.progress(min(int(r["risk_score"]), 100) / 100)
        else:
            st.info("Enter applicant details and click **Evaluate Risk**.")


# ---------------- TAB 2: AI AGENT ----------------
with tab_agent:
    st.subheader("🤖 AI Agent")
    st.caption('The agent observes -> reasons -> calls the fuzzy engine as a tool -> explains -> acts. '
               'Try: "income 80k, credit 600, ratio 0.45, no default, 1 year"')

    client = get_client()
    if client is None:
        st.warning("No GEMINI_API_KEY found. Add it under **Manage app -> Settings -> Secrets** "
                   "(deployed) or as an environment variable (local). "
                   "Get a free key at aistudio.google.com.")

    if "gem_history" not in st.session_state:
        st.session_state.gem_history = []
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    for role, text in st.session_state.chat_log:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("Describe an applicant…", disabled=(client is None))
    if prompt:
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
                    box.markdown(f"🔧 **calling** `{name}` -> `{json.dumps(args, ensure_ascii=False)}`")
                elif kind == "result":
                    name, res = payload
                    if name == "calculate_loan_risk":
                        box.markdown(f"↳ score **{res['risk_score']}** -> **{res['decision']}**")
                        box.json(res["membership_degrees"])
                    else:
                        box.json(res)
            with st.spinner("Agent thinking…"):
                st.session_state.gem_history = run_agent(
                    client, st.session_state.gem_history, prompt, render)
            st.session_state.chat_log.append(("assistant", "\n\n".join(buf["text"]) or "_(done)_"))


# ---------------- TAB 3: GA RESULTS ----------------
with tab_ga:
    st.subheader("🧬 Genetic Algorithm — MF Parameter Optimisation")
    st.caption(f"Population={fe.GA_POP} · Generations={fe.GA_GENS} · "
               f"Crossover={fe.GA_CR*100:.0f}% (two-point) · Adaptive mutation · "
               f"Tournament k={fe.GA_TOURNAMENT} · Fitness = Sum(loan_status - fuzzy_risk)^2")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Initial fitness", f"{ga['init_fitness']:.1f}")
    m2.metric("Final fitness", f"{ga['best_fitness']:.1f}", f"-{ga['improvement_pct']:.1f}%")
    m3.metric("Approved (ref)", f"{fe.n_approved:,}")
    m4.metric("Rejected (ref)", f"{fe.n_rejected:,}")

    st.pyplot(convergence_figure())

    st.markdown("**Optimised MF Breakpoints**")
    st.table({
        "Variable": ["Annual Income (RM)", "Loan Ratio (0-1)", "Employment Exp (yrs)"],
        "a (Low->Mid)": [f"{fe.inc_a:,.0f}", f"{fe.rat_a:.4f}", f"{fe.emp_a:.2f}"],
        "b (Mid peak)": [f"{fe.inc_b:,.0f}", f"{fe.rat_b:.4f}", f"{fe.emp_b:.2f}"],
        "c (Mid->High)": [f"{fe.inc_c:,.0f}", f"{fe.rat_c:.4f}", f"{fe.emp_c:.2f}"],
    })
    st.caption("Credit Score uses fixed CTOS categories; Previous Default is binary — "
               "neither is GA-evolved.")


# ---------------- TAB 4: MEMBERSHIP FUNCTIONS ----------------
with tab_mf:
    st.subheader("📈 Membership Functions")
    st.caption("Green-titled = GA-optimised · others fixed. These are the fuzzy sets the "
               "inference engine uses.")
    st.pyplot(mf_figure())
